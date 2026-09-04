# Initial implementation

Spindle now has a typed process-owned Gleam API and a weft actor owning a
native Port. The C++ helper uses pinned llama.cpp, loads a model once,
counts tokens, returns normalized batch embeddings, and accepts shutdown
independently of model work. The current path is CPU-only with four
threads and a 2,048-token input limit.

Local verification includes deterministic Gleam protocol/lifecycle tests
and real EmbeddingGemma round trips, repeated output, stop, and reload.
The native suite exercises malformed requests, input overflow, oversized
frame headers, EOF shutdown, and shutdown delivery around startup and a
large batch. These timing tests do not prove interruption at a specific
point inside model loading or inference. See the
README for the exact model digest and commands. Platform CI compiles the
native helper and runs the model-free suite; it does not download weights.

Before a stable release, add an explicit model profile with artifact
verification, query/document templates, pooling, tokenizer settings, and
dimensions. Add a measured GPU path and matching release artifacts.
Dispose of late replies after unconfirmed teardown without retaining
them in a long-lived caller mailbox. Extend native lifecycle tests to fault injection that proves hard owner
kill and blocked OS paths rather than only cooperative fixture behavior.

Loom owns the SQLite/vector integration, indexed-source revisions,
chunking, erasure, and retrieval evaluation. Track that work in
https://github.com/Roasbeef/loom/issues/226. Shared vector caches, query
expansion, reranking, and transparent helper restart are outside this
initial implementation.
