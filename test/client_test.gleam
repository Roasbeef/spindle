import gleam/erlang/process
import gleam/result
import spindle
import weft/poll

@external(erlang, "spindle_test_ffi", "cwd")
fn cwd() -> String

@external(erlang, "spindle_test_ffi", "alive")
fn alive(pid: Int) -> Bool

@external(erlang, "spindle_test_ffi", "mailbox_size")
fn mailbox_size() -> Int

fn helper() -> String {
  cwd() <> "/test/fixtures/helper.py"
}

pub fn owned_engine_roundtrip_and_observed_stop_test() {
  let assert Ok(engine) = spindle.start(helper(), "fake", 5000)
    as "fixture starts"
  assert spindle.dimensions(engine) == 2
  assert spindle.count_tokens(engine, ["one", "two"], 5000) == Ok([3, 3])
  assert spindle.embed(engine, ["one", "two"], 5000)
    == Ok([[1.0, 0.0], [1.0, 0.0]])
  let pid = spindle.helper_pid(engine)
  assert alive(pid)
  assert spindle.stop(engine, 5000) == Ok(Nil)
  assert !alive(pid)
}

pub fn timeout_observes_exit_and_replacement_can_embed_test() {
  let assert Ok(engine) = spindle.start(helper(), "stall", 5000)
    as "fixture starts"
  let pid = spindle.helper_pid(engine)
  let before = mailbox_size()
  let outcome = spindle.embed(engine, ["one"], 20)
  assert mailbox_size() == before
  assert outcome == Error(spindle.TimedOut)
  assert !alive(pid)
  let assert Ok(replacement) = spindle.start(helper(), "fake", 5000)
    as "replacement starts"
  assert spindle.embed(replacement, ["one"], 5000) == Ok([[1.0, 0.0]])
  assert spindle.stop(replacement, 5000) == Ok(Nil)
}

pub fn startup_timeout_can_shutdown_before_greeting_test() {
  assert spindle.start(helper(), "no-greeting", 20) == Error(spindle.TimedOut)
}

pub fn malformed_native_reply_fails_the_request_test() {
  let assert Ok(engine) = spindle.start(helper(), "wrong-id", 5000)
    as "fixture starts"
  assert result.is_error(spindle.embed(engine, ["one"], 5000))
}

pub fn another_process_cannot_share_an_engine_test() {
  let assert Ok(engine) = spindle.start(helper(), "fake", 5000)
    as "fixture starts"
  let reply = process.new_subject()
  let _ =
    process.spawn(fn() {
      process.send(reply, spindle.embed(engine, ["one"], 5000))
    })
  assert process.receive(reply, 5000) == Ok(Error(spindle.WrongOwner))
  assert spindle.stop(engine, 5000) == Ok(Nil)
}

pub fn invalid_input_does_not_break_a_loaded_engine_test() {
  let assert Ok(engine) = spindle.start(helper(), "fake", 5000)
    as "fixture starts"
  assert result.is_error(spindle.embed(engine, [], 5000))
  assert spindle.embed(engine, ["one"], 5000) == Ok([[1.0, 0.0]])
  assert spindle.stop(engine, 5000) == Ok(Nil)
}

pub fn owner_death_terminates_the_helper_test() {
  let reply = process.new_subject()
  let _ =
    process.spawn(fn() {
      let assert Ok(engine) = spindle.start(helper(), "fake", 5000)
        as "fixture starts"
      process.send(reply, spindle.helper_pid(engine))
    })
  let assert Ok(pid) = process.receive(reply, 5000)
    as "owner reports native PID"
  assert poll.until(within: 5000, every: 10, attempt: fn() {
      case alive(pid) {
        True -> poll.Retry
        False -> poll.Done(Nil)
      }
    })
    == poll.Answered(Nil)
}

pub fn timeout_preserves_observed_native_failure_test() {
  let assert Ok(engine) = spindle.start(helper(), "stall-exit-error", 5000)
    as "fixture starts"
  let pid = spindle.helper_pid(engine)
  let before = mailbox_size()
  let outcome = spindle.embed(engine, ["one"], 20)
  assert mailbox_size() == before
  assert outcome == Error(spindle.HelperFailed("native shutdown failed"))
  assert !alive(pid)
}

pub fn startup_failure_preserves_the_stop_waiter_test() {
  assert spindle.start(helper(), "startup-error-on-stop", 20)
    == Error(spindle.HelperFailed("native shutdown failed"))
}
