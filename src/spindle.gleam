//// Local embeddings with a model held in an external process.
////
//// Each engine belongs to its creating Gleam process. The synchronous API
//// admits one request at a time; a weft state machine owns the Port
//// and observes helper exit before acknowledging an orderly stop. Model execution and
//// the helper's stdin watchdog live outside the BEAM VM.

import gleam/erlang/port.{type Port}
import gleam/erlang/process.{type Pid, type Subject}
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import gleam/string
import spindle/internal/port as native
import spindle/protocol
import weft/state_machine as sm

/// Errors distinguish failed requests from unconfirmed native teardown.
pub type Error {
  /// A configuration or input violated the public contract.
  InvalidInput(String)

  /// The caller is not the process that created this engine.
  WrongOwner

  /// Native startup, inference, or protocol validation failed.
  HelperFailed(String)

  /// The request timed out and the helper's exit was observed.
  TimedOut

  /// Teardown was requested, but native exit was not observed in time.
  DrainUnconfirmed

  /// The owner machine died; this error does not assert native drain.
  Unavailable
}

/// A process-owned engine with immutable model bounds.
pub opaque type Engine {
  Engine(client: Client, dimensions: Int, context_tokens: Int, helper_pid: Int)
}

/// Client identifies the machine and the only process allowed to call it.
/// Ownership makes the synchronous API the admission boundary: callers cannot
/// build an unbounded work queue by sharing an engine across processes.
type Client {
  Client(subject: Subject(Command), pid: Pid, owner: Pid)
}

/// Reply is a single terminal answer addressed to one synchronous call.
type Reply =
  Subject(Result(Answer, Error))

/// Command combines API requests, selected Port events, and lifecycle signals.
/// The machine owns the selector, so native messages never reach the caller.
type Command {
  /// Wait for the model greeting, postponing while the helper loads.
  Ready(reply: Reply)

  /// Submit an already formatted batch after the model is ready.
  Execute(operation: protocol.Operation, texts: List(String), reply: Reply)

  /// Retire the engine and acknowledge only an observed native exit.
  Stop(reply: Reply)

  /// Deliver bytes or exit status from this machine's Port only.
  FromPort(native.Event)

  /// Retire the helper when its creating process exits.
  OwnerDied

  /// End the drain wait without claiming the native process has exited.
  DrainExpired
}

/// Answer separates model metadata, operation results, and observed shutdown.
type Answer {
  /// Startup metadata comes from a validated native greeting.
  ModelReady(dimensions: Int, context_tokens: Int, helper_pid: Int)

  /// Vectors preserve the submitted input order.
  Embeddings(List(List(Float)))

  /// Counts include the embedding tokenizer's special tokens.
  Counts(List(Int))

  /// The native process exited with status zero.
  Stopped
}

/// Model contains the bounds established by the one accepted startup greeting.
/// It exists only in phases where requests can be admitted or validated.
type Model {
  Model(dimensions: Int, context_tokens: Int)
}

/// Request retains the identity and shape needed to validate exactly one reply.
type Request {
  Request(id: Int, operation: protocol.Operation, count: Int, reply: Reply)
}

/// Phase carries only the data valid at that point in the helper's lifetime.
/// A running request always has a loaded model. Draining has no request to
/// resume: its terminal reply was sent before entering that phase.
type Phase {
  /// The native reader is live, but the model greeting has not arrived.
  Loading

  /// The model is loaded and no batch is in flight.
  Idle(model: Model)

  /// One submitted batch owns the next operation response.
  Running(model: Model, request: Request)

  /// Admission is closed; only native exit can acknowledge a successful stop.
  Draining(reason: String, waiter: Option(Reply))
}

/// Data holds transport state that survives phase changes. Keeping it separate
/// prevents a fragmented frame or request counter update from resetting a
/// state timeout or replaying postponed events.
type Data {
  Data(port: Port, helper_pid: Int, buffer: BitArray, next_id: Int)
}

/// Load a model in a dedicated helper and wait for its versioned greeting.
/// The helper path must be absolute. The initial runtime uses CPU inference.
///
/// ## Examples
///
/// ```gleam
/// start("/opt/spindle/spindle-helper", "/models/embeddinggemma.gguf", 60_000)
/// ```
pub fn start(
  helper: String,
  model: String,
  within: Int,
) -> Result(Engine, Error) {
  use _ <- result.try(valid_timeout(within))
  let owner = process.self()
  let started =
    sm.new_with_initialiser(1000, fn(subject) {
      use opened <- result.try(native.open(helper, model))
      let #(port, helper_pid) = opened
      // The machine owns this Port directly. Its exit closes stdin even
      // when an untrappable kill bypasses every Gleam callback.
      let monitor = process.monitor(owner)
      let selector =
        process.new_selector()
        |> process.select(subject)
        |> process.select_record(port, 1, fn(message) {
          FromPort(native.event(message))
        })
        |> process.select_specific_monitor(monitor, fn(_) { OwnerDied })
      sm.initialised(Loading, Data(port, helper_pid, <<>>, 1))
      |> sm.selecting(selector)
      |> sm.returning(subject)
      |> Ok
    })
    |> sm.on_event(handle)
    |> sm.on_enter(entered)
    |> sm.unlinked
    |> sm.start
  use started <- result.try(
    started
    |> result.map_error(fn(error) {
      HelperFailed("could not start helper owner: " <> string.inspect(error))
    }),
  )
  let client = Client(started.data, started.pid, owner)
  use answer <- result.try(ask(client, within, Ready))
  case answer {
    ModelReady(dimensions, context, pid) ->
      Ok(Engine(client, dimensions, context, pid))
    _ -> Error(HelperFailed("invalid startup answer"))
  }
}

/// Return the loaded model's output dimensions.
///
/// ## Examples
///
/// ```gleam
/// dimensions(engine)
/// ```
pub fn dimensions(engine: Engine) -> Int {
  engine.dimensions
}

/// Return the loaded model's effective token limit, including special tokens.
///
/// ## Examples
///
/// ```gleam
/// context_tokens(engine)
/// ```
pub fn context_tokens(engine: Engine) -> Int {
  engine.context_tokens
}

/// Return the native PID for diagnostics and lifecycle verification.
///
/// ## Examples
///
/// ```gleam
/// helper_pid(engine)
/// ```
pub fn helper_pid(engine: Engine) -> Int {
  engine.helper_pid
}

/// Embed already formatted inputs in their original order.
/// A timeout terminates this engine; callers explicitly start a replacement.
///
/// ## Examples
///
/// ```gleam
/// embed(engine, [query("durable agent memory")], 30_000)
/// ```
pub fn embed(
  engine: Engine,
  texts: List(String),
  within: Int,
) -> Result(List(List(Float)), Error) {
  use answer <- result.try(
    ask(engine.client, within, fn(reply) {
      Execute(protocol.Embed, texts, reply)
    }),
  )
  case answer {
    Embeddings(values) -> Ok(values)
    _ -> Error(HelperFailed("invalid embedding answer"))
  }
}

/// Count tokens with precisely the native embedding tokenizer settings.
/// Inputs beyond the context limit fail instead of being silently truncated.
///
/// ## Examples
///
/// ```gleam
/// count_tokens(engine, [query("durable agent memory")], 30_000)
/// ```
pub fn count_tokens(
  engine: Engine,
  texts: List(String),
  within: Int,
) -> Result(List(Int), Error) {
  use answer <- result.try(
    ask(engine.client, within, fn(reply) {
      Execute(protocol.CountTokens, texts, reply)
    }),
  )
  case answer {
    Counts(values) -> Ok(values)
    _ -> Error(HelperFailed("invalid token-count answer"))
  }
}

/// Stop an engine and wait for its native process exit.
/// Unavailable or unconfirmed teardown never reports successful drain.
///
/// ## Examples
///
/// ```gleam
/// stop(engine, 5000)
/// ```
pub fn stop(engine: Engine, within: Int) -> Result(Nil, Error) {
  ask(engine.client, within, Stop) |> result.map(fn(_) { Nil })
}

/// Format an EmbeddingGemma retrieval query.
/// This template is model-specific and must also identify the stored profile.
///
/// ## Examples
///
/// ```gleam
/// query("how did we fix the lease race?")
/// ```
pub fn query(text: String) -> String {
  "task: search result | query: " <> text
}

/// Format an EmbeddingGemma document with its source title.
///
/// ## Examples
///
/// ```gleam
/// document("decision", "Keep the writer lease with the session owner.")
/// ```
pub fn document(title: String, text: String) -> String {
  "title: " <> title <> " | text: " <> text
}

/// valid_timeout bounds caller waits before a machine or request is created.
fn valid_timeout(within: Int) -> Result(Nil, Error) {
  case within > 0 && within <= 300_000 {
    True -> Ok(Nil)
    False -> Error(InvalidInput("timeout must be 1 to 300000 milliseconds"))
  }
}

/// Received keeps a machine death distinct from a terminal API reply.
type Received {
  Replied(Result(Answer, Error))
  Died
}

/// ask enforces creator ownership and monitors the machine for each call.
/// Caller timeout requests teardown; it never implies that inference stopped.
fn ask(
  client: Client,
  within: Int,
  command: fn(Subject(Result(Answer, Error))) -> Command,
) -> Result(Answer, Error) {
  use _ <- result.try(valid_timeout(within))
  case process.self() == client.owner {
    False -> Error(WrongOwner)
    True -> {
      let reply = process.new_subject()
      let monitor = process.monitor(client.pid)
      let selector =
        process.new_selector()
        |> process.select_map(reply, Replied)
        |> process.select_specific_monitor(monitor, fn(_) { Died })
      process.send(client.subject, command(reply))
      let outcome = case process.selector_receive(selector, within) {
        Ok(Replied(result)) -> result
        Ok(Died) -> Error(Unavailable)
        Error(Nil) -> cancel_and_drain(client, reply)
      }
      process.demonitor_process(monitor)
      outcome
    }
  }
}

/// cancel_and_drain retires a timed-out engine and waits for native exit.
/// The machine settles the original call before the stop acknowledgement, so
/// a confirmed acknowledgement also permits disposal of that queued reply.
fn cancel_and_drain(
  client: Client,
  original: Subject(Result(Answer, Error)),
) -> Result(Answer, Error) {
  let reply = process.new_subject()
  process.send(client.subject, Stop(reply))
  case process.receive(reply, 5000) {
    Ok(Ok(Stopped)) -> {
      // The machine settles the original request before acknowledging exit.
      let _ = process.receive(original, 0)
      Error(TimedOut)
    }

    // The machine settles the original call before either drain outcome.
    Ok(Error(error)) -> {
      let _ = process.receive(original, 0)
      Error(error)
    }
    _ -> Error(DrainUnconfirmed)
  }
}

/// handle dispatches lifecycle events against the phase in which they are
/// legal. Only Loading postpones Ready; entering either Idle or Draining
/// replays that call, so startup cannot strand a caller after model failure.
fn handle(
  phase: Phase,
  data: Data,
  command: Command,
) -> sm.Next(Phase, Data, Command) {
  case phase, command {
    Loading, Ready(_) -> sm.keep(data) |> sm.postpone
    Idle(model), Ready(reply) | Running(model, _), Ready(reply) -> {
      process.send(
        reply,
        Ok(ModelReady(model.dimensions, model.context_tokens, data.helper_pid)),
      )
      sm.keep(data)
    }
    Draining(reason, _), Ready(reply) -> {
      process.send(reply, Error(HelperFailed(reason)))
      sm.keep(data)
    }

    // Admission requires both a loaded model and the absence of a request.
    Idle(model), Execute(operation, texts, reply) ->
      execute(model, data, operation, texts, reply)
    Loading, Execute(_, _, reply)
    | Running(_, _), Execute(_, _, reply)
    | Draining(_, _), Execute(_, _, reply)
    -> {
      process.send(reply, Error(Unavailable))
      sm.keep(data)
    }

    // Every path into draining settles the operation before transferring
    // custody to the stop waiter. Native stdout can no longer finish work.
    _, Stop(reply) -> drain(phase, data, Some(reply), "engine stopped")
    _, OwnerDied -> drain(phase, data, None, "engine creator exited")
    _, FromPort(native.Exited(status)) -> exited(phase, status)

    // A response racing cancellation has no remaining consumer. Ignoring it
    // preserves the drain waiter and does not restart its deadline.
    Draining(_, _), FromPort(native.Bytes(_))
    | Draining(_, _), FromPort(native.Invalid)
    -> sm.keep(data)
    _, FromPort(native.Invalid) -> fail(phase, data, "unexpected port message")
    _, FromPort(native.Bytes(bytes)) -> {
      case protocol.feed(data.buffer, bytes) {
        Error(reason) -> fail(phase, data, reason)
        Ok(#(None, buffer)) -> sm.keep(Data(..data, buffer: buffer))
        Ok(#(Some(response), buffer)) ->
          respond(phase, Data(..data, buffer: buffer), response)
      }
    }

    // Closing the Port requests EOF teardown but is not an exit observation.
    // The machine stops too, so an unresponsive helper cannot retain an owner
    // process indefinitely. The caller is told the proof is incomplete.
    Draining(_, waiter), DrainExpired -> {
      settle(waiter, Error(DrainUnconfirmed))
      native.close(data.port)
      sm.stop()
    }
    Loading, DrainExpired
    | Idle(_), DrainExpired
    | Running(_, _), DrainExpired
    -> sm.keep(data)
  }
}

/// entered owns the shutdown effect and its deadline. A named timeout
/// survives replacement of the stop waiter, which changes the phase value.
/// Neither a repeated stop nor late stdout can extend the drain window.
/// The machine owns the Port directly; machine death also closes its stdin.
fn entered(
  from: Phase,
  to: Phase,
  data: Data,
) -> sm.Enter(Phase, Data, Command) {
  case from, to {
    Draining(_, _), Draining(_, _) -> sm.keep(data)
    _, Draining(_, _) -> {
      let _ = native.send(data.port, protocol.shutdown())
      sm.keep(data)
      |> sm.with_named_timeout(
        name: "drain",
        after: 5000,
        sending: DrainExpired,
      )
    }
    _, Loading | _, Idle(_) | _, Running(_, _) -> sm.keep(data)
  }
}

/// execute validates and submits a batch before transferring reply ownership
/// to Running. Encoding errors leave the model usable; a transport failure
/// retires it because native admission can no longer be established.
fn execute(
  model: Model,
  data: Data,
  operation: protocol.Operation,
  texts: List(String),
  reply: Reply,
) -> sm.Next(Phase, Data, Command) {
  case protocol.request(operation, data.next_id, texts) {
    Error(reason) -> {
      process.send(reply, Error(InvalidInput(reason)))
      sm.keep(data)
    }
    Ok(bytes) -> {
      case native.send(data.port, bytes) {
        Error(reason) -> {
          process.send(reply, Error(HelperFailed(reason)))
          fail(Idle(model), data, reason)
        }
        Ok(Nil) -> {
          let request =
            Request(data.next_id, operation, list.length(texts), reply)
          let next_id = case data.next_id {
            4_294_967_295 -> 1
            n -> n + 1
          }
          sm.transition(Running(model, request), Data(..data, next_id: next_id))
        }
      }
    }
  }
}

/// respond accepts a greeting only during Loading and operation output only
/// for the current request. A malformed or unsolicited response retires the
/// engine; a correlated native request error leaves the loaded model usable.
fn respond(
  phase: Phase,
  data: Data,
  response: protocol.Response,
) -> sm.Next(Phase, Data, Command) {
  case phase, response {
    Loading, protocol.Ready(dimensions, context) ->
      sm.transition(Idle(Model(dimensions, context)), data)
    Running(model, Request(expected, protocol.Embed, count, reply)),
      protocol.Vectors(id, dimensions, vectors)
      if id == expected
    -> {
      case model.dimensions == dimensions && list.length(vectors) == count {
        True -> {
          process.send(reply, Ok(Embeddings(vectors)))
          sm.transition(Idle(model), data)
        }
        False -> fail(phase, data, "embedding shape differs from loaded model")
      }
    }
    Running(model, Request(expected, protocol.CountTokens, count, reply)),
      protocol.TokenCounts(id, counts)
      if id == expected
    -> {
      case count == list.length(counts) {
        True -> {
          process.send(reply, Ok(Counts(counts)))
          sm.transition(Idle(model), data)
        }
        False -> fail(phase, data, "token count differs from request")
      }
    }
    Running(model, Request(expected, _, _, reply)), protocol.Failure(id, reason)
      if id == expected
    -> {
      process.send(reply, Error(HelperFailed(reason)))
      sm.transition(Idle(model), data)
    }
    _, protocol.Failure(0, reason) -> fail(phase, data, reason)
    _, _ -> fail(phase, data, "unsolicited or mismatched native response")
  }
}

/// drain settles the in-flight request before entering terminal teardown.
/// A repeated stop replaces only the acknowledgement recipient; owner death
/// has no recipient. Neither transition can make the engine usable again.
fn drain(
  phase: Phase,
  data: Data,
  waiter: Option(Reply),
  reason: String,
) -> sm.Next(Phase, Data, Command) {
  case phase {
    Running(_, request) ->
      process.send(request.reply, Error(HelperFailed(reason)))
    Loading | Idle(_) | Draining(_, _) -> Nil
  }
  // Retain the first failure for a postponed startup call. A later stop
  // changes the acknowledgement recipient, not the cause of teardown.
  let first_reason = case phase {
    Draining(first, _) -> first
    Loading | Idle(_) | Running(_, _) -> reason
  }
  sm.transition(Draining(first_reason, waiter), data)
}

/// exited is the only path that reports an observed native stop. Pending work
/// fails before the stop acknowledgement is sent, preserving message ordering
/// for the caller's timed-out reply cleanup.
fn exited(phase: Phase, status: Int) -> sm.Next(Phase, Data, Command) {
  case phase {
    Running(_, request) ->
      process.send(request.reply, Error(HelperFailed("native helper exited")))
    Draining(_, waiter) ->
      settle(waiter, case status {
        0 -> Ok(Stopped)
        _ -> Error(HelperFailed("native shutdown failed"))
      })
    Loading | Idle(_) -> Nil
  }
  sm.stop()
}

/// settle sends an acknowledgement only when a live call requested one.
fn settle(waiter: Option(Reply), answer: Result(Answer, Error)) -> Nil {
  case waiter {
    None -> Nil
    Some(reply) -> process.send(reply, answer)
  }
}

/// fail retires an engine without replacing an existing drain waiter. Once
/// draining has begun, later protocol failures cannot change its ownership.
fn fail(
  phase: Phase,
  data: Data,
  reason: String,
) -> sm.Next(Phase, Data, Command) {
  case phase {
    Draining(_, _) -> sm.keep(data)
    Loading | Idle(_) | Running(_, _) -> drain(phase, data, None, reason)
  }
}
