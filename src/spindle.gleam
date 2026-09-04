//// Local embeddings with a model held in an external process.
////
//// Each engine belongs to its creating Gleam process. The synchronous API
//// admits one request at a time; a weft actor owns the Port and observes
//// helper exit before acknowledging an orderly stop. Model execution and
//// the helper's stdin watchdog live outside the BEAM VM.

import gleam/erlang/port.{type Port}
import gleam/erlang/process.{type Pid, type Subject}
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import gleam/string
import spindle/internal/port as native
import spindle/protocol
import weft/actor

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

  /// The owner actor died; this error does not assert native drain.
  Unavailable
}

/// A process-owned engine with immutable model bounds.
pub opaque type Engine {
  Engine(client: Client, dimensions: Int, context_tokens: Int, helper_pid: Int)
}

type Client {
  Client(subject: Subject(Command), pid: Pid, owner: Pid)
}

type Command {
  Ready(reply: Subject(Result(Answer, Error)))
  Execute(
    operation: protocol.Operation,
    texts: List(String),
    reply: Subject(Result(Answer, Error)),
  )
  Stop(reply: Subject(Result(Answer, Error)))
  FromPort(native.Event)
  OwnerDied
}

type Answer {
  ModelReady(dimensions: Int, context_tokens: Int, helper_pid: Int)
  Embeddings(List(List(Float)))
  Counts(List(Int))
  Stopped
}

type Pending {
  Waiting(reply: Subject(Result(Answer, Error)))
  Computing(
    id: Int,
    operation: protocol.Operation,
    count: Int,
    reply: Subject(Result(Answer, Error)),
  )
}

type Closing {
  Open
  Draining(waiter: Option(Subject(Result(Answer, Error))))
}

type State {
  State(
    port: Port,
    helper_pid: Int,
    buffer: BitArray,
    model: Option(#(Int, Int)),
    pending: Option(Pending),
    closing: Closing,
    next_id: Int,
  )
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
    actor.new_with_initialiser(1000, fn(subject) {
      use opened <- result.try(native.open(helper, model))
      let #(port, helper_pid) = opened
      let monitor = process.monitor(owner)
      let selector =
        process.new_selector()
        |> process.select(subject)
        |> process.select_record(port, 1, fn(message) {
          FromPort(native.event(message))
        })
        |> process.select_specific_monitor(monitor, fn(_) { OwnerDied })
      actor.initialised(State(port, helper_pid, <<>>, None, None, Open, 1))
      |> actor.selecting(selector)
      |> actor.returning(subject)
      |> Ok
    })
    |> actor.on_message(handle)
    |> actor.on_shutdown(fn(state, _) { native.close(state.port) })
    |> actor.unlinked
    |> actor.start
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

fn valid_timeout(within: Int) -> Result(Nil, Error) {
  case within > 0 && within <= 300_000 {
    True -> Ok(Nil)
    False -> Error(InvalidInput("timeout must be 1 to 300000 milliseconds"))
  }
}

type Received {
  Replied(Result(Answer, Error))
  Died
}

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

fn cancel_and_drain(
  client: Client,
  original: Subject(Result(Answer, Error)),
) -> Result(Answer, Error) {
  let reply = process.new_subject()
  process.send(client.subject, Stop(reply))
  case process.receive(reply, 5000) {
    Ok(Ok(Stopped)) -> {
      // The actor settles the original request before acknowledging exit.
      let _ = process.receive(original, 0)
      Error(TimedOut)
    }

    // A failed stop acknowledgement still carries an observed native exit.
    Ok(Error(error)) -> {
      let _ = process.receive(original, 0)
      Error(error)
    }
    _ -> Error(DrainUnconfirmed)
  }
}

fn handle(state: State, command: Command) -> actor.Next(State, Command) {
  case command {
    Ready(reply) -> {
      case state.model {
        Some(#(dimensions, context)) -> {
          process.send(
            reply,
            Ok(ModelReady(dimensions, context, state.helper_pid)),
          )
          actor.continue(state)
        }
        None -> actor.continue(State(..state, pending: Some(Waiting(reply))))
      }
    }
    Execute(operation, texts, reply) -> {
      case state.closing {
        Open -> execute(state, operation, texts, reply)
        Draining(_) -> {
          process.send(reply, Error(Unavailable))
          actor.continue(state)
        }
      }
    }
    Stop(reply) -> {
      let _ = native.send(state.port, protocol.shutdown())
      actor.continue(State(..state, closing: Draining(Some(reply))))
    }
    OwnerDied -> {
      let _ = native.send(state.port, protocol.shutdown())
      actor.continue(State(..state, closing: Draining(None)))
    }
    FromPort(native.Exited(status)) -> {
      settle_pending(state.pending, Error(HelperFailed("native helper exited")))
      case state.closing {
        Draining(Some(reply)) ->
          process.send(reply, case status {
            0 -> Ok(Stopped)
            _ -> Error(HelperFailed("native shutdown failed"))
          })
        Open | Draining(None) -> Nil
      }
      actor.stop()
    }
    FromPort(native.Invalid) -> fail(state, "unexpected port message")
    FromPort(native.Bytes(bytes)) -> {
      case protocol.feed(state.buffer, bytes) {
        Error(reason) -> fail(state, reason)
        Ok(#(None, buffer)) -> actor.continue(State(..state, buffer: buffer))
        Ok(#(Some(response), buffer)) ->
          respond(State(..state, buffer: buffer), response)
      }
    }
  }
}

fn execute(
  state: State,
  operation: protocol.Operation,
  texts: List(String),
  reply: Subject(Result(Answer, Error)),
) -> actor.Next(State, Command) {
  let encoded = protocol.request(operation, state.next_id, texts)
  case encoded {
    Error(reason) -> {
      process.send(reply, Error(InvalidInput(reason)))
      actor.continue(state)
    }
    Ok(bytes) -> {
      case native.send(state.port, bytes) {
        Error(reason) -> {
          process.send(reply, Error(HelperFailed(reason)))
          fail(state, reason)
        }
        Ok(Nil) ->
          actor.continue(
            State(
              ..state,
              pending: Some(Computing(
                state.next_id,
                operation,
                list.length(texts),
                reply,
              )),
              next_id: case state.next_id {
                4_294_967_295 -> 1
                n -> n + 1
              },
            ),
          )
      }
    }
  }
}

fn respond(
  state: State,
  response: protocol.Response,
) -> actor.Next(State, Command) {
  case response, state.pending {
    protocol.Ready(dimensions, context), None ->
      actor.continue(State(..state, model: Some(#(dimensions, context))))
    protocol.Ready(dimensions, context), Some(Waiting(reply)) -> {
      process.send(reply, Ok(ModelReady(dimensions, context, state.helper_pid)))
      actor.continue(
        State(..state, model: Some(#(dimensions, context)), pending: None),
      )
    }
    protocol.Vectors(id, dimensions, vectors),
      Some(Computing(expected, protocol.Embed, count, reply))
      if id == expected
    -> {
      case
        case state.model {
          Some(#(expected, _)) -> expected == dimensions
          None -> False
        }
        && list.length(vectors) == count
      {
        True -> {
          process.send(reply, Ok(Embeddings(vectors)))
          actor.continue(State(..state, pending: None))
        }
        False -> fail(state, "embedding shape differs from loaded model")
      }
    }
    protocol.TokenCounts(id, counts),
      Some(Computing(expected, protocol.CountTokens, count, reply))
      if id == expected
    -> {
      case count == list.length(counts) {
        True -> {
          process.send(reply, Ok(Counts(counts)))
          actor.continue(State(..state, pending: None))
        }
        False -> fail(state, "token count differs from request")
      }
    }
    protocol.Failure(id, reason), Some(Computing(expected, _, _, reply))
      if id == expected
    -> {
      process.send(reply, Error(HelperFailed(reason)))
      actor.continue(State(..state, pending: None))
    }
    protocol.Failure(0, reason), _ -> fail(state, reason)
    _, _ -> fail(state, "unsolicited or mismatched native response")
  }
}

fn settle_pending(
  pending: Option(Pending),
  result: Result(Answer, Error),
) -> Nil {
  case pending {
    None -> Nil
    Some(Waiting(reply)) | Some(Computing(_, _, _, reply)) ->
      process.send(reply, result)
  }
}

fn fail(state: State, reason: String) -> actor.Next(State, Command) {
  settle_pending(state.pending, Error(HelperFailed(reason)))
  let _ = native.send(state.port, protocol.shutdown())
  // Protocol failure must preserve a stop waiter until native exit arrives.
  let closing = case state.closing {
    Open -> Draining(None)
    Draining(waiter) -> Draining(waiter)
  }
  actor.continue(State(..state, pending: None, closing: closing))
}
