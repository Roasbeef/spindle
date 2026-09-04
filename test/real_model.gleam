//// Explicit integration executable. Missing model configuration is an error.

import gleam/io
import gleam/list
import spindle

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
  io.println("Real-model embedding, repeatability, stop, and restart passed.")
}
