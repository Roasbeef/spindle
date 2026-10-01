# Spindle architecture

Spindle keeps a local embedding model in a dedicated native process and
exposes it through a process-owned Gleam API. We use this boundary to keep
model memory, native crashes, and inference threads outside the BEAM VM. The
helper stays loaded across requests; callers explicitly create a new engine
after a timeout or terminal failure.

This document describes the current CPU implementation. Model profiles,
artifact verification, GPU execution, and release packaging remain open. The
[implementation plan](next.md) records those limits, and [Loom issue
#226](https://github.com/Roasbeef/loom/issues/226) tracks the consumer's
SQLite indexing and retrieval work.

## Reading the implementation

Start with [`spindle.gleam`](../src/spindle.gleam). Its types precede the
public functions, and its module `Flow` describes startup, request
admission, and retirement using the actual helper names. The command table
covers all four `Phase` constructors and every `Command` constructor. The
response table above `accept_response` covers every `protocol.Response`
constructor.

| Boundary | Entry points | Continue reading | Responsibility |
|---|---|---|---|
| Public calls | `start`, `embed`, `count_tokens`, `stop` in [`spindle.gleam`](../src/spindle.gleam). | `await_command`, `retire_timed_out_engine`. | Enforce creating-process identity; separate the operation wait from the drain wait. |
| Helper lifecycle | `handle_event`, `enter_phase` in [`spindle.gleam`](../src/spindle.gleam). | `submit_batch`, `accept_response`, `begin_drain`, `observe_exit`. | Admit one batch, correlate replies, retire on terminal failure, and observe exit. |
| Pure wire codec | `request`, `shutdown`, `feed`, `decode` in [`protocol.gleam`](../src/spindle/protocol.gleam). | `frame_payload`, `decode_counts_loop`, `decode_vectors_loop`, `decode_vector_loop`. | Bound payloads and fully decode their shape without process operations. |
| BEAM transport | `open`, `send`, `close`, `event` in [`internal/port.gleam`](../src/spindle/internal/port.gleam). | [`spindle_port.erl`](../src/spindle_port.erl). | Open an executable, expose its OS PID, avoid suspended writes, and normalize Port messages. |
| Native model | `main`, `receive` in [`main.cpp`](../native/main.cpp). | `serve`, `tokenize`, `embed`, `send`, `failure`. | Keep inference outside the VM and process owner loss independently of model work. |

The library has three Gleam modules because the boundaries have different
responsibilities. `spindle` performs lifecycle effects; `spindle/protocol`
is pure; `spindle/internal/port` confines the foreign function interface
(FFI). The Erlang shim exposes runtime operations, and the C++ executable
owns llama.cpp. Indexing, chunking, retrieval, and persisted model identity
belong to the consuming application.

## Principles visible in the source

Lifecycle state carries its valid data. `Running(model, request)` has both
loaded bounds and one reply obligation. `Draining(reason, waiter)` carries a
teardown diagnosis and optional acknowledgement recipient after that request
has been settled. Transport fragments live in `Data`, so a buffer update
cannot imitate a phase transition or replay a postponed call.

Calls name the boundary they cross: `protocol.request` encodes,
`native.send` submits, and `sm.transition` transfers lifecycle state to weft.
Private helper names identify their domain operations. `begin_drain` settles
work, `enter_phase` submits shutdown, and `observe_exit` supplies the evidence
required for successful stop. Module `Flow` sections and transition tables
make these paths visible before their implementations.

Teardown closes admission before awaiting native exit. Neither later stdout
nor a subsequent stop can reopen the engine or extend its first drain
window. A replacement is a fresh `start` call with a fresh helper; the
machine has no restart transition. Successful request settlement and
successful helper retirement are separate events.

## Gleam idioms used here

`Engine` is opaque, so callers can inspect startup metadata only through
functions and cannot construct a client with another process's identity.
The private `Phase`, `Command`, and `Answer` custom types are algebraic data
types: each constructor names a possible state or message, and `case`
extracts the data valid for that constructor. The paired match in
`handle_event` expresses phase-by-command handling in one place.

Gleam values are immutable. `Data(..data, buffer: buffer)` creates a new
transport value with one field replaced; it does not modify the previous
value or the Port. Effects such as sending a reply are explicit calls before
returning weft's next transition. That order matters in `begin_drain`, where
the request's terminal reply must precede the stop acknowledgement.

`use value <- result.try(expression)` passes the rest of its block as the
success continuation. An error bypasses that continuation, so request
validation can remain flat without exceptions. `Option(Reply)` represents
whether a stop recipient exists; `Result(Answer, Error)` represents success
or failure. Absence and failure carry different types.

The protocol decoders use bit-array patterns to consume exact wire shapes.
Their recursive helpers prepend values into a reversed accumulator, then
reverse once at the boundary. `decode_vector_loop` also carries the squared
norm, so finite components alone cannot admit a zero or non-unit vector. The
loops reconstruct request order without repeatedly appending to a list.

## Process ownership

```mermaid
flowchart LR
    C[Creating Gleam process] -->|Typed requests and replies| S[weft state machine]
    S -->|Owns and selects| P[BEAM Port]
    P <-->|Framed stdin and stdout| H[Native helper]
    H --> R[Independent stdin reader]
    H --> M[Model loading and inference]
```

The process that calls `spindle.start` owns the resulting engine. The
initialiser opens the Port in the machine process before publishing its
command subject to that creator. Public calls check that process identity
before sending work. A synchronous call admits one batch at a time, so
sharing an engine across processes cannot create an unbounded request queue.
An engine is an opaque handle to its machine and immutable startup metadata.

The machine monitors the creator and owns the Port directly. It selects its
typed command subject, messages from that Port, and the creator's monitor.
The Erlang shim opens an executable directly without a shell, keeps stderr
separate from protocol stdout, and sends without suspending on a busy Port
buffer. FFI exposes these runtime operations; Gleam owns the policy.

A state machine already returns the same `Started` type used by Gleam OTP
actors and communicates through typed subjects. Spindle needs no extra actor
to translate messages or own the helper. The implementation uses weft 0.4.2
without an upstream extension.

## Lifecycle

The phase type carries the information valid at each point in the helper's
lifetime. Transport data holds the Port, native PID, partial frame, and next
request ID separately. Receiving another fragment does not constitute a
lifecycle transition.

```mermaid
stateDiagram-v2
    [*] --> Loading
    Loading --> Idle: Valid model greeting
    Idle --> Running: Batch submitted
    Running --> Idle: Valid result or correlated request error
    Loading --> Draining: Stop, creator death, or protocol failure
    Idle --> Draining: Stop, creator death, or protocol failure
    Running --> Draining: Stop, creator death, or protocol failure
    Draining --> [*]: Native exit or drain deadline
    Loading --> [*]: Native exit
    Idle --> [*]: Native exit
    Running --> [*]: Native exit
```

| Phase | Owned information | Permitted progress |
|---|---|---|
| `Loading` | No model or batch. | Accept the startup greeting; postpone the startup call. |
| `Idle` | Validated model bounds. | Answer startup or admit one batch. |
| `Running` | Model and one request's ID, operation, count, and reply subject. | Validate the matching response, then return to `Idle`. |
| `Draining` | First teardown reason and optional stop acknowledgement recipient. | Observe native exit or report an expired drain wait. |

Startup uses weft's postponed-event support. If `Ready` arrives while the
model loads, the machine postpones that command. A transition to `Idle`
replays it with validated metadata. A transition to `Draining` replays it
with the original teardown reason. A later stop preserves that reason. There
is no separate startup-waiter field to preserve on every failure path.

`Running` always contains a loaded model and a request. The old actor
representation stored optional model data, optional pending work, and a
separate closing flag. The phase type removes combinations such as a
running request without model bounds. A duplicate greeting after startup
is a protocol failure, not a replacement for the model metadata.

Entering `Draining` settles any running request before retaining the stop
recipient. Native stdout received afterward cannot complete work or replace
that recipient. A subsequent `Stop` replaces the recipient, and `OwnerDied`
removes it, while both preserve the first reason and original deadline. Once
this phase begins, no transition admits a new request or returns to `Idle`.

## Deadlines and native exit

The public call waits for its requested deadline, subject to the API's 1 to
300,000 millisecond bound. Model startup also has a separate one-second
machine-initialization bound for opening the Port; model loading happens
afterward. A timed-out call sends `Stop` and allows five additional seconds
for its acknowledgement.

The first entry into `Draining` sends native shutdown and starts a
five-second weft named timeout. A named timeout matters here: replacing the
acknowledgement subject changes the `Draining` value, which weft would treat
as a state change. The named deadline survives that update and is armed only
once. Repeated stops and late stdout cannot extend it.

The native reader starts before model loading. It reads independently of
inference and terminates the entire process on shutdown or stdin EOF.
Shutdown bypasses the one-frame work slot, so it does not wait behind an
embedding batch. The OS reclaims the process's model memory and threads.

The machine reports successful stop only after receiving the Port's native
exit status of zero. A nonzero status is a helper failure. If the drain
deadline fires first, it reports `DrainUnconfirmed`, closes the Port, and
stops. Closing the Port requests EOF teardown; it does not prove that the
native process has exited.

Killing the machine closes its owned Port even though no shutdown callback
runs. The helper's reader then sees EOF. This ownership rule replaces the
actor's explicit `on_shutdown` close callback. A caller that observes
machine death receives `Unavailable`, which makes no claim about native
drain.

| Result | What the caller knows |
|---|---|
| Successful `stop` | Native exit with status zero was observed. |
| `TimedOut` | The operation deadline expired and native exit with status zero was subsequently observed. |
| `HelperFailed` | Startup, request validation, inference, or shutdown failed; the error category alone does not establish drain. |
| `DrainUnconfirmed` | Teardown was requested, but native exit was not confirmed to this call. |
| `Unavailable` | The machine is unavailable; native drain is not established. |

`begin_drain` settles a running request before `observe_exit` can
acknowledge stop. Since both messages come from the same process, a confirmed acknowledgement
lets the caller consume the timed-out request's queued terminal reply.
An unconfirmed teardown can still leave late replies in a long-lived
creator mailbox. Disposable reply delivery remains follow-up work; the
state-machine conversion does not claim to solve it.

## Protocol and model execution

[`protocol.gleam`](../src/spindle/protocol.gleam) contains the pure wire
codec. [`protocol.md`](protocol.md) specifies the exact tags and limits. A
four-byte big-endian length precedes each bounded payload. `protocol.feed`
checks accumulated frame size before concatenating the partial buffer and
incoming Port chunk. The runtime has already allocated that incoming chunk.
Native `receive` checks the header before allocating its body. Frame limits
bound protocol buffers rather than model weights or the model's inference
memory. Diagnostics never enter stdout.

The native reader transfers at most one pending frame through `Inbox` to the
main inference thread. Main removes that frame under the mutex, releases the
lock, then calls `serve`. The reader can therefore handle shutdown during
model work. Native code permits one queued frame while main processes
another; a second queued frame terminates the connection. The public API
does not pipeline requests. Native code validates all inputs before
inference and builds the full response before sending it, so a later input
error cannot publish a partial embedding batch.

`serve` tokenizes every input before evaluating the first embedding. It
publishes a response only after the full operation succeeds; a caught request
error produces `Failure(id, reason)` and keeps the loaded model available.
Startup failure produces ID zero and exits. The Gleam owner treats a matching
request failure as recoverable, while a framing, identity, or shape failure
retires the engine because the helper's output can no longer be correlated.

Token counting and embedding share tokenizer settings. Inputs exceeding the
2,048-token context fail explicitly. The current helper loads one model with
CPU execution, four threads, and model-selected mean, CLS, or last-token
pooling. It rejects nonfinite and zero-norm output before normalization.
Gleam independently validates vector shape, finite float encoding, and unit
norm against the in-flight request and loaded model.

The observed dimensions identify a shape, not an embedding space. The
current `query` and `document` helpers format EmbeddingGemma inputs; they do
not verify that the supplied model is EmbeddingGemma. A future profile must
pin the artifact, tokenizer, templates, pooling, and output conventions
before Loom can safely persist compatible vectors.

## Verification and remaining boundaries

The deterministic suite covers protocol framing, shape validation, creator
ownership, postponed startup, timeout acknowledgement races, late malformed
stdout during drain, machine hard kill, and a peer that ignores shutdown,
including repeated stops against the original deadline. The real-model
executable verifies embeddings, repeatability, stop, reload, and EOF cleanup
after killing the machine with a loaded model. Native tests cover malformed
requests, input limits, and shutdown delivery around startup and batch
submission.

[`client_test.gleam`](../test/client_test.gleam) covers machine lifecycle,
[`protocol_test.gleam`](../test/protocol_test.gleam) covers codec rejection,
[`real_model.gleam`](../test/real_model.gleam) exercises the model through
the public API, and [`test_native.py`](../scripts/test_native.py) exercises
the native wire directly. The [README](../README.md#real-model-verification)
records the required paths, model digest, and commands. `gleam docs build`
renders the public module and function comments into the package's API docs.

These tests do not establish interruption at a controlled point inside model
execution or cleanup while the kernel itself blocks native work. Linux and
macOS CI run the deterministic suite and compile the native helper;
real-model execution remains an explicit local command using separately
supplied weights. No result here establishes retrieval quality. Loom owns
source extraction, chunking, index transactions, erasure, and ranking,
independently of Spindle's helper lifecycle.
