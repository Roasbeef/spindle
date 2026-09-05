# Native protocol, version 1

The Gleam owner starts `spindle-helper MODEL.gguf` directly with a BEAM
Port. Protocol data uses stdin/stdout; diagnostics use stderr. Each frame
starts with a four-byte unsigned big-endian payload length, followed by
that many bytes. Payloads must contain 1 to 1,048,576 bytes. The receiver
rejects the length before allocating the body.

Integers below are unsigned big-endian. Strings are a u32 byte count and
UTF-8 bytes. Embeddings use IEEE float32 in big-endian byte order. The
owner sends one request at a time and never pipelines batches. Native
responses use request order, with a count that must exactly match the
request. The owner assigns nonzero u32 request IDs and validates replies.

| Tag | Direction | Payload after tag |
|---|---|---|
| 0 | Helper to owner | u16 version (1), u32 dimensions, u32 effective context limit. |
| 1 | Owner to helper | u32 request ID, u32 count, counted strings to embed. |
| 2 | Owner to helper | u32 request ID, u32 count, counted strings to tokenize. |
| 3 | Owner to helper | u32 zero; terminate the helper immediately. |
| 129 | Helper to owner | u32 ID, u32 count, u32 dimensions, count times dimensions float32 values. |
| 130 | Helper to owner | u32 ID, u32 count, count u32 token counts. |
| 255 | Helper to owner | u32 ID (zero for startup), counted error string, at most 4,096 bytes. |

A batch has 1 to 16 inputs, each at most 65,536 UTF-8 bytes, subject to the
aggregate frame limit. The current helper uses a 2,048-token context,
including special tokens. It tokenizes with `add_special=true` and
`parse_special=false` for both operations. A text that exceeds the token
limit fails instead of being truncated. All texts are validated before
batch inference begins. Computation is sequential within a batch.

The helper accepts model-selected mean, CLS, or last-token pooling,
rejects ranking and unpooled models, and returns finite L2-normalized
vectors of 1 to 4,096 dimensions. The Gleam decoder independently checks
wire lengths, dimensions, finite float encodings, and squared norms
between 0.999 and 1.001. A native request error has no partial success.

## Ownership and shutdown

The native reader thread starts before model loading. It bounds and reads
frames independently of model computation, with one pending request slot.
Shutdown bypasses that slot and exits the entire process. EOF means the
owner is gone and also exits the process. This is necessary during model
load and GPU work, where an inference abort callback alone is insufficient.
Native exit reclaims the model and inference threads through the OS.

The Gleam API restricts an engine to its creating process. A weft state machine
owns its Port and monitors that creator. Creator death requests native
shutdown; machine death closes stdin, which the native reader treats as
owner loss. A normal `stop` waits for the Port's OS exit-status message.

Request timeout triggers engine shutdown and allows up to five additional
seconds to observe native exit. `TimedOut` means exit was observed;
`DrainUnconfirmed` means it was not. `Unavailable` means the machine died
and does not assert native drain. An engine cannot be reused after a
timeout. The machine also bounds draining to five seconds, then reports
unconfirmed teardown, closes the Port, and stops. Late stdout and repeated
stop requests cannot extend that deadline. Callers create a replacement explicitly. Confirmed teardown
consumes the original request's terminal reply before returning. After
`DrainUnconfirmed`, a late reply can remain in the creator's mailbox until
that process exits; bounded late-reply disposal is a follow-up before
long-lived indexing workers rely on this failure mode.

The reader exits immediately on EOF, including an interrupted frame. A
peer that supplies malformed complete requests receives an error;
zero-length or oversized frames terminate the connection. Frame limits
bound protocol allocation, not model weight or inference memory.

## Scope of the first version

The helper uses CPU inference with four threads. The CMake Metal option
can compile that backend but does not enable offload in this initial API.
The public `query` and `document` formatting functions are specifically
for EmbeddingGemma; they do not infer a model profile from a filename.
Model checksum enforcement and an explicit profile object are follow-up
work. The model's observed dimensions are not its identity.

The package does not download models, create indexes, retry failed
requests, or transparently restart helpers. These policies stay explicit
while lifecycle and model-profile contracts are being established.
