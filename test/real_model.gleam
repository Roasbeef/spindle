//// Explicit integration executable. Missing model configuration is an error.

import gleam/erlang/process
import gleam/io
import gleam/list
import spindle
import weft/poll

@external(erlang, "spindle_test_ffi", "env")
fn env(name: String) -> Result(String, Nil)

pub fn main() {
  let assert Ok(helper) = env("SPINDLE_HELPER") as "set SPINDLE_HELPER"
  let assert Ok(model) = env("SPINDLE_MODEL") as "set SPINDLE_MODEL"
  let texts = [
    spindle.query("durable agent memory"),
    spindle.document("notes", "The agent stores decisions in durable notes."),
  ]
  let assert Ok(engine) = spindle.start(helper, model, 60_000) as "model loads"
  let assert Ok(counts) = spindle.count_tokens(engine, texts, 30_000)
    as "tokenization succeeds"
  assert list.length(counts) == 2
  let assert Ok(vectors) = spindle.embed(engine, texts, 30_000)
    as "embedding succeeds"
  assert list.length(vectors) == 2
  list.each(vectors, fn(vector) {
    assert list.length(vector) == spindle.dimensions(engine)
  })
  assert spindle.embed(engine, texts, 30_000) == Ok(vectors)
  assert spindle.stop(engine, 5000) == Ok(Nil)
  let assert Ok(restarted) = spindle.start(helper, model, 60_000)
    as "model reloads"
  assert spindle.embed(restarted, texts, 30_000) == Ok(vectors)
  assert spindle.stop(restarted, 5000) == Ok(Nil)
  // The real native reader must handle owner loss without a Gleam shutdown
  // callback. This proves EOF teardown with a loaded model after hard kill.
  let assert Ok(abandoned) = spindle.start(helper, model, 60_000)
    as "model loads for owner-loss test"
  let pid = spindle.helper_pid(abandoned)
  let assert Ok(owner) = helper_owner(pid) as "locate native Port owner"
  process.kill(owner)
  assert poll.until(within: 5000, every: 10, attempt: fn() {
      case alive(pid) {
        True -> poll.Retry
        False -> poll.Done(Nil)
      }
    })
    == poll.Answered(Nil)
  io.println(
    "Real-model embedding, repeatability, stop, restart, and owner kill passed.",
  )
}

@external(erlang, "spindle_test_ffi", "helper_owner")
fn helper_owner(pid: Int) -> Result(process.Pid, Nil)

@external(erlang, "spindle_test_ffi", "alive")
fn alive(pid: Int) -> Bool
