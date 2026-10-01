//// Local embeddings with a model held in an external process.
////
//// Each engine belongs to its creating Gleam process. The synchronous API
//// admits one request at a time; a weft state machine owns the BEAM Port and
//// observes helper exit before acknowledging an orderly stop. Model execution
//// and the helper's independent stdin reader live outside the BEAM VM.
////
//// ## Flow
////
//// `start` opens the helper inside the machine's initialiser, then
//// `await_command` sends `Ready`. `handle_event` postpones that call in
//// `Loading`; `accept_response` validates the greeting and enters `Idle`,
//// where weft replays the call with model bounds.
////
//// `embed` and `count_tokens` use `await_command` to enforce creator ownership
//// and watch machine death. `handle_event` admits work only from `Idle`;
//// `submit_batch` encodes and sends it before entering `Running`.
//// `accept_response` correlates the reply with that request and returns to
//// `Idle` after sending its terminal answer.
////
//// `stop`, creator death, and protocol failure reach `begin_drain`.
//// `enter_phase` sends shutdown once and arms the named drain deadline.
//// `observe_exit` acknowledges an observed native exit; `DrainExpired`
//// closes the Port and stops with `DrainUnconfirmed`. A caller deadline uses
//// `retire_timed_out_engine` to request this teardown and wait separately.
////
//// ## State and event transitions
////
//// The table covers every `Command` and `native.Event` constructor in
//// `handle_event`. “Keep” preserves the phase; response dispatch is described
//// by `accept_response`. `begin_drain` preserves the first teardown reason.
////
//// | Event | Loading | Idle | Running | Draining |
//// |---|---|---|---|---|
//// | `Ready` | Postpone. | Reply metadata; keep. | Reply metadata; keep. | Reply first failure; keep. |
//// | `Execute` | Reply unavailable; keep. | `submit_batch`. | Reply unavailable; keep. | Reply unavailable; keep. |
//// | `Stop` | Begin drain with waiter. | Begin drain with waiter. | Settle request; begin drain with waiter. | Replace waiter; keep deadline. |
//// | `OwnerDied` | Begin drain without waiter. | Begin drain without waiter. | Settle request; begin drain without waiter. | Remove waiter; keep deadline. |
//// | `FromPort(Exited)` | Stop machine. | Stop machine. | Fail request; stop machine. | Acknowledge status; stop machine. |
//// | `FromPort(Bytes)` | Feed framing; accept response. | Feed framing; accept response. | Feed framing; accept response. | Ignore; keep. |
//// | `FromPort(Invalid)` | Begin drain. | Begin drain. | Fail request; begin drain. | Ignore; keep. |
//// | `DrainExpired` | Keep. | Keep. | Keep. | Reply unconfirmed; close Port; stop machine. |
////
//// In `submit_batch`, encoding errors reply `InvalidInput` and keep `Idle`;
//// send failure replies `HelperFailed` and begins drain; successful submission
//// enters `Running`. For `FromPort(Bytes)`, an incomplete frame keeps the
//// phase, a framing error begins drain, and a complete frame reaches the
//// response table at `accept_response`.
////
//// `Engine` carries immutable startup metadata. `Phase` carries lifecycle
//// information; `Data` carries transport information. Updating a partial
//// frame or request counter therefore cannot replay postponed startup calls.
//// See `docs/architecture.md` for the process boundaries and reading order.

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

  /// The request timed out and native exit with status zero was observed.
  TimedOut

  /// Teardown was requested, but native exit was not observed in time.
  DrainUnconfirmed

  /// The owner machine died; this error does not assert native drain.
  Unavailable
}

/// A process-owned engine with immutable model bounds.
pub opaque type Engine {
  /// Construction follows a validated greeting; callers cannot forge ownership.
  Engine(
    /// The machine address and its sole permitted caller.
    client: Client,
    /// The loaded model's output width, between 1 and 4,096 components.
    dimensions: Int,
    /// The advertised input limit, including special tokens.
    context_tokens: Int,
    /// The OS process identity captured when the Port was opened.
    helper_pid: Int,
  )
}

/// Client identifies the machine and the only process allowed to call it.
/// Ownership makes the synchronous API the admission boundary: callers cannot
/// build an unbounded work queue by sharing an engine across processes.
type Client {
  /// The starter records its identity before the machine is spawned.
  Client(
    /// Typed mailbox selected by the machine alongside Port and monitor events.
    subject: Subject(Command),
    /// The machine identity monitored during each synchronous call.
    pid: Pid,
    /// The creating process whose identity gates every public operation.
    owner: Pid,
  )
}

/// Reply is a single terminal answer addressed to one synchronous call.
type Reply =
  Subject(Result(Answer, Error))

/// Command combines API requests, selected Port events, and lifecycle signals.
/// The machine owns the selector, so native messages never reach the caller.
type Command {
  /// Wait for the model greeting, postponing while the helper loads.
  Ready(
    /// The startup call awaiting validated model metadata or terminal failure.
    reply: Reply,
  )

  /// Submit an already formatted batch after the model is ready.
  Execute(
    /// The wire operation whose response shape must match the request.
    operation: protocol.Operation,
    /// Inputs already formatted for the caller's chosen model.
    texts: List(String),
    /// The sole recipient of the batch's terminal answer.
    reply: Reply,
  )

  /// Retire the engine and acknowledge only an observed native exit.
  Stop(
    /// The acknowledgement recipient, installed after any batch is settled.
    reply: Reply,
  )

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
  ModelReady(
    /// Validated width of each embedding vector.
    dimensions: Int,
    /// Validated input token bound supplied by the greeting.
    context_tokens: Int,
    /// Native PID from the Port, independent of the greeting payload.
    helper_pid: Int,
  )

  /// Vectors preserve the submitted input order.
  Embeddings(List(List(Float)))

  /// Counts include the embedding tokenizer's special tokens.
  Counts(List(Int))

  /// The native process exited with status zero.
  Stopped
}

/// A per-call selector distinguishes a reply from loss of the machine.
type Received {
  /// The machine sent the terminal answer for this call's subject.
  Replied(Result(Answer, Error))

  /// The monitored machine exited; native teardown remains unconfirmed.
  Died
}

/// Model contains the bounds established by the one accepted startup greeting.
/// It exists only in phases where requests can be admitted or validated.
type Model {
  /// These bounds survive successful requests and correlated native errors.
  Model(
    /// Width checked against each embedding response.
    dimensions: Int,
    /// Effective input limit reported to the creating process.
    context_tokens: Int,
  )
}

/// Request retains the identity and shape needed to validate exactly one reply.
type Request {
  /// Created only after the encoded batch has been sent to the Port.
  Request(
    /// Nonzero u32 identity used to reject stale or unsolicited output.
    id: Int,
    /// Determines whether vectors or token counts are legal.
    operation: protocol.Operation,
    /// Input count that the result must preserve exactly.
    count: Int,
    /// Owned until response, failure, drain, or native exit settles the call.
    reply: Reply,
  )
}

/// Phase carries only the data valid at that point in the helper's lifetime.
/// A running request always has a loaded model. Draining has no request to
/// resume: its terminal reply was sent before entering that phase.
type Phase {
  /// The native reader is live, but the model greeting has not arrived.
  Loading

  /// The model is loaded and no batch is in flight.
  Idle(
    /// Validated model bounds available before any batch is sent.
    model: Model,
  )

  /// One submitted batch owns the next operation response.
  Running(
    /// Bounds retained while the helper processes the batch.
    model: Model,
    /// Exactly one in-flight request whose terminal answer is still owed.
    request: Request,
  )

  /// Admission is closed; only native exit can acknowledge a successful stop.
  Draining(
    /// The first teardown diagnosis, preserved for postponed startup calls.
    reason: String,
    /// The latest stop call, or no recipient after creator death or failure.
    waiter: Option(Reply),
  )
}

/// Data holds transport state that survives phase changes. Keeping it separate
/// prevents a fragmented frame or request counter update from resetting a
/// state timeout or replaying postponed events.
type Data {
  /// The machine owns this transport for its whole lifetime.
  Data(
    /// Direct process ownership closes stdin even after an untrappable kill.
    port: Port,
    /// Native process identity published with startup metadata.
    helper_pid: Int,
    /// A bounded partial response frame; cleared after complete decoding.
    buffer: BitArray,
    /// Next nonzero u32 request identity, wrapping from the maximum to one.
    next_id: Int,
  )
}

/// Load a model in a dedicated helper and wait for its versioned greeting.
/// Supply an absolute helper path and a local model path. The helper is
/// opened without a shell; this function does not check model identity or
/// download weights. The initial runtime uses CPU inference.
///
/// `within` bounds the model-greeting wait in milliseconds. On expiry the
/// call requests shutdown and waits up to five additional seconds for native
/// exit. Machine initialisation has its own one-second Port-opening bound.
///
/// ## Examples
///
/// ```gleam
/// spindle.start("/opt/spindle/spindle-helper", "/models/embeddinggemma.gguf", 60_000)
/// // -> Ok(engine) after the model greeting, or Error(reason).
/// ```
pub fn start(
  helper: String,
  model: String,
  within: Int,
) -> Result(Engine, Error) {
  use _ <- result.try(validate_timeout(within))
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
    |> sm.on_event(handle_event)
    |> sm.on_enter(enter_phase)
    |> sm.unlinked
    |> sm.start
  use started <- result.try(
    started
    |> result.map_error(fn(error) {
      HelperFailed("could not start helper owner: " <> string.inspect(error))
    }),
  )
  let client = Client(started.data, started.pid, owner)
  use answer <- result.try(await_command(client, within, Ready))
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
/// spindle.dimensions(engine)
/// // -> The width advertised by the loaded model.
/// ```
pub fn dimensions(engine: Engine) -> Int {
  engine.dimensions
}

/// Return the loaded model's effective token limit, including special tokens.
///
/// ## Examples
///
/// ```gleam
/// spindle.context_tokens(engine)
/// // -> The token limit advertised by the loaded model.
/// ```
pub fn context_tokens(engine: Engine) -> Int {
  engine.context_tokens
}

/// Return the native PID for diagnostics and lifecycle verification.
///
/// ## Examples
///
/// ```gleam
/// spindle.helper_pid(engine)
/// // -> The PID of the external helper process.
/// ```
pub fn helper_pid(engine: Engine) -> Int {
  engine.helper_pid
}

/// Embed already formatted inputs in their original order.
/// A caller deadline retires this engine. `TimedOut` confirms exit with status
/// zero; `DrainUnconfirmed` does not establish native exit. Start a replacement
/// explicitly after timeout. Input-encoding errors leave the engine usable.
///
/// ## Examples
///
/// ```gleam
/// spindle.embed(engine, [spindle.query("durable agent memory")], 30_000)
/// // -> Ok([vector]) in input order, or Error(reason).
/// ```
pub fn embed(
  engine: Engine,
  texts: List(String),
  within: Int,
) -> Result(List(List(Float)), Error) {
  use answer <- result.try(
    await_command(engine.client, within, fn(reply) {
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
/// Each count includes special tokens. A caller deadline retires the engine,
/// with the same separate drain wait and error meanings as `embed`.
///
/// ## Examples
///
/// ```gleam
/// spindle.count_tokens(engine, [spindle.query("durable agent memory")], 30_000)
/// // -> Ok([count]) including special tokens, or Error(reason).
/// ```
pub fn count_tokens(
  engine: Engine,
  texts: List(String),
  within: Int,
) -> Result(List(Int), Error) {
  use answer <- result.try(
    await_command(engine.client, within, fn(reply) {
      Execute(protocol.CountTokens, texts, reply)
    }),
  )
  case answer {
    Counts(values) -> Ok(values)
    _ -> Error(HelperFailed("invalid token-count answer"))
  }
}

/// Stop an engine and wait for its native process exit.
/// `Ok(Nil)` requires exit status zero. `Unavailable` or `DrainUnconfirmed`
/// makes no claim about native exit. The machine allows five seconds from
/// first entry into draining; the caller's `within` wait is separate.
///
/// ## Examples
///
/// ```gleam
/// spindle.stop(engine, 5000)
/// // -> Ok(Nil) only after native exit with status zero.
/// ```
pub fn stop(engine: Engine, within: Int) -> Result(Nil, Error) {
  await_command(engine.client, within, Stop) |> result.map(fn(_) { Nil })
}

/// Format an EmbeddingGemma retrieval query.
/// Pair this template with an EmbeddingGemma model and record it as part of
/// any stored embedding profile. Formatting does not verify the loaded model.
///
/// ## Examples
///
/// ```gleam
/// assert spindle.query("lease race") == "task: search result | query: lease race"
/// ```
pub fn query(text: String) -> String {
  "task: search result | query: " <> text
}

/// Format an EmbeddingGemma document with its source title.
/// The title and text are passed through verbatim; callers own extraction and
/// chunking, and must keep the template consistent with the stored profile.
///
/// ## Examples
///
/// ```gleam
/// assert spindle.document("decision", "Keep the lease.")
///   == "title: decision | text: Keep the lease."
/// ```
pub fn document(title: String, text: String) -> String {
  "title: " <> title <> " | text: " <> text
}

/// Enforce creator ownership and monitor the machine for each call.
/// Caller timeout requests teardown; it never implies that inference stopped.
fn await_command(
  client: Client,
  within: Int,
  command: fn(Subject(Result(Answer, Error))) -> Command,
) -> Result(Answer, Error) {
  use _ <- result.try(validate_timeout(within))
  case process.self() == client.owner {
    False -> Error(WrongOwner)
    True -> {
      let reply = process.new_subject()
      let monitor = process.monitor(client.pid)
      let selector =
        process.new_selector()
        |> process.select_map(reply, Replied)
        |> process.select_specific_monitor(monitor, fn(_) { Died })

      // The selector and monitor exist before submission, so a fast reply or
      // machine exit remains observable for this call.
      process.send(client.subject, command(reply))
      let outcome = case process.selector_receive(selector, within) {
        Ok(Replied(result)) -> result
        Ok(Died) -> Error(Unavailable)
        Error(Nil) -> retire_timed_out_engine(client, reply)
      }

      // The call has one terminal outcome; release its death notification.
      process.demonitor_process(monitor)
      outcome
    }
  }
}

/// Retire a timed-out engine and wait separately for native exit.
/// The machine settles the original call before the stop acknowledgement, so
/// a confirmed acknowledgement also permits disposal of that queued reply.
fn retire_timed_out_engine(
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

/// Dispatch lifecycle events against the phase in which they are legal. Only Loading postpones Ready; entering either Idle or Draining
/// replays that call, so startup cannot strand a caller after model failure.
fn handle_event(
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
      submit_batch(model, data, operation, texts, reply)
    Loading, Execute(_, _, reply)
    | Running(_, _), Execute(_, _, reply)
    | Draining(_, _), Execute(_, _, reply)
    -> {
      process.send(reply, Error(Unavailable))
      sm.keep(data)
    }

    // Every path into draining settles the operation before transferring
    // custody to the stop waiter. Native stdout can no longer finish work.
    _, Stop(reply) -> begin_drain(phase, data, Some(reply), "engine stopped")
    _, OwnerDied -> begin_drain(phase, data, None, "engine creator exited")
    _, FromPort(native.Exited(status)) -> observe_exit(phase, status)

    // A response racing cancellation has no remaining consumer. Ignoring it
    // preserves the drain waiter and does not restart its deadline.
    Draining(_, _), FromPort(native.Bytes(_))
    | Draining(_, _), FromPort(native.Invalid)
    -> sm.keep(data)
    _, FromPort(native.Invalid) ->
      retire_failed_engine(phase, data, "unexpected port message")
    _, FromPort(native.Bytes(bytes)) -> {
      case protocol.feed(data.buffer, bytes) {
        Error(reason) -> retire_failed_engine(phase, data, reason)
        Ok(#(None, buffer)) -> sm.keep(Data(..data, buffer: buffer))
        Ok(#(Some(response), buffer)) ->
          accept_response(phase, Data(..data, buffer: buffer), response)
      }
    }

    // Closing the Port requests EOF teardown but is not an exit observation.
    // The machine stops too, so an unresponsive helper cannot retain an owner
    // process indefinitely. The caller is told the proof is incomplete.
    Draining(_, waiter), DrainExpired -> {
      acknowledge_stop(waiter, Error(DrainUnconfirmed))
      native.close(data.port)
      sm.stop()
    }
    Loading, DrainExpired
    | Idle(_), DrainExpired
    | Running(_, _), DrainExpired
    -> sm.keep(data)
  }
}

/// Own the shutdown effect and its deadline. A named timeout
/// survives replacement of the stop waiter, which changes the phase value.
/// Neither a repeated stop nor late stdout can extend the drain window.
/// The machine owns the Port directly; machine death also closes its stdin.
fn enter_phase(
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

/// Validate and submit a batch before transferring reply ownership to Running. Encoding errors leave the model usable; a transport failure
/// retires it because native admission can no longer be established.
fn submit_batch(
  model: Model,
  data: Data,
  operation: protocol.Operation,
  texts: List(String),
  reply: Reply,
) -> sm.Next(Phase, Data, Command) {
  // Encoding can reject input before native work is admitted. Only a
  // successful nonblocking send transfers the reply obligation to Running.
  case protocol.request(operation, data.next_id, texts) {
    Error(reason) -> {
      process.send(reply, Error(InvalidInput(reason)))
      sm.keep(data)
    }
    Ok(bytes) -> {
      case native.send(data.port, bytes) {
        Error(reason) -> {
          process.send(reply, Error(HelperFailed(reason)))
          retire_failed_engine(Idle(model), data, reason)
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

/// Accept a greeting only during Loading and operation output only for the
/// current request. A malformed or unsolicited response retires the
/// engine; a correlated native request error leaves the loaded model usable.
///
/// | Response | Accepted phase and condition | Result |
/// |---|---|---|
/// | `Ready` | `Loading`. | Enter `Idle` with validated bounds. |
/// | `Vectors` | `Running(Embed)` with matching ID, dimensions, and count. | Reply vectors; enter `Idle`. |
/// | `TokenCounts` | `Running(CountTokens)` with matching ID and count. | Reply counts; enter `Idle`. |
/// | `Failure` | `Running` with matching request ID. | Fail that request; enter `Idle`. |
/// | `Failure(0, reason)` | No earlier matching request arm. | Begin drain with the native diagnosis. |
/// | Any other pairing | Unsolicited, wrong operation, or wrong identity. | Begin drain. |
///
/// Framing errors drain before reaching this function. Draining discards Port
/// bytes in `handle_event`, so no response can reopen admission after shutdown.
fn accept_response(
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
        False ->
          retire_failed_engine(
            phase,
            data,
            "embedding shape differs from loaded model",
          )
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
        False ->
          retire_failed_engine(phase, data, "token count differs from request")
      }
    }
    Running(model, Request(expected, _, _, reply)), protocol.Failure(id, reason)
      if id == expected
    -> {
      process.send(reply, Error(HelperFailed(reason)))
      sm.transition(Idle(model), data)
    }
    _, protocol.Failure(0, reason) -> retire_failed_engine(phase, data, reason)
    _, _ ->
      retire_failed_engine(
        phase,
        data,
        "unsolicited or mismatched native response",
      )
  }
}

/// Settle the in-flight request before entering terminal teardown.
/// A repeated stop replaces only the acknowledgement recipient; owner death
/// has no recipient. Neither transition can make the engine usable again.
fn begin_drain(
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

/// Report an observed native stop through the only path with exit evidence.
/// Pending work fails before the stop acknowledgement is sent, preserving
/// message ordering for the caller's timed-out reply cleanup.
fn observe_exit(phase: Phase, status: Int) -> sm.Next(Phase, Data, Command) {
  case phase {
    Running(_, request) ->
      process.send(request.reply, Error(HelperFailed("native helper exited")))
    Draining(_, waiter) ->
      acknowledge_stop(waiter, case status {
        0 -> Ok(Stopped)
        _ -> Error(HelperFailed("native shutdown failed"))
      })
    Loading | Idle(_) -> Nil
  }
  sm.stop()
}

/// Send an acknowledgement only when a call requested one.
fn acknowledge_stop(
  waiter: Option(Reply),
  answer: Result(Answer, Error),
) -> Nil {
  case waiter {
    None -> Nil
    Some(reply) -> process.send(reply, answer)
  }
}

/// Retire an engine without replacing an existing drain waiter. Once
/// draining has begun, later protocol failures cannot change its ownership.
fn retire_failed_engine(
  phase: Phase,
  data: Data,
  reason: String,
) -> sm.Next(Phase, Data, Command) {
  case phase {
    Draining(_, _) -> sm.keep(data)
    Loading | Idle(_) | Running(_, _) -> begin_drain(phase, data, None, reason)
  }
}

/// Bound caller waits before a machine or request is created.
fn validate_timeout(within: Int) -> Result(Nil, Error) {
  case within > 0 && within <= 300_000 {
    True -> Ok(Nil)
    False -> Error(InvalidInput("timeout must be 1 to 300000 milliseconds"))
  }
}
