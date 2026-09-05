# Spindle

Local embeddings for Gleam, with model inference in an owned native
process. A small C++ helper links against pinned llama.cpp and keeps its
model loaded across requests. The Gleam package communicates over a BEAM
Port, with a weft state machine owning the helper's lifecycle.

Spindle owns embedding requests and helper lifetime. Applications own text
extraction, indexing, and retrieval policy. The initial consumer is
[Loom](https://github.com/Roasbeef/loom); its
[tracking issue](https://github.com/Roasbeef/loom/issues/226) records the
larger vector-retrieval project.

**Status: initial implementation, not yet published to Hex.** The current
runtime is CPU-only. It has been exercised locally with EmbeddingGemma
300M Q8_0; model profiles and release packaging are still in progress.

## Build

Requires Gleam 1.18+, Erlang/OTP 29, CMake 3.24+, a C++17 compiler, and
Python 3 for lifecycle fixtures. CMake downloads a checksummed llama.cpp
source archive pinned to `427291b5b34cd914a31b3fd3b61a68f6184f4b9f`.

```sh
gleam deps download
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --target spindle-helper --parallel 4
gleam build --warnings-as-errors
gleam test
```

Model weights are supplied separately. Use an absolute helper path and a
verified local GGUF model. The example uses EmbeddingGemma's query format:

```gleam
import gleam/result
import spindle

pub fn example(helper: String, model: String) {
  use engine <- result.try(spindle.start(helper, model, 60_000))
  let result = spindle.embed(
    engine,
    [spindle.query("why did the agent preserve the writer lease?")],
    30_000,
  )
  let stopped = spindle.stop(engine, 5000)
  use _ <- result.try(stopped)
  result
}
```

An engine belongs to the process that created it. Moving it to another
process returns `WrongOwner`. One request is in flight; vectors preserve
input order. A timeout shuts down the engine and reports whether its
native exit was observed. Start a replacement explicitly after timeout.
`stop` can report `Unavailable` if a prior failure already ended the machine.

## Real-model verification

The ordinary suite uses a deterministic native-protocol fixture. The
explicit integration commands require a model; missing configuration is
an error rather than a skipped test.

```sh
export SPINDLE_HELPER="$PWD/native/build/spindle-helper"
export SPINDLE_MODEL=/absolute/path/to/embeddinggemma-300M-Q8_0.gguf
gleam run -m real_model
python3 scripts/test_native.py
```

The model used for the initial local proof had SHA-256
`f470220f84b6235197541352d22f10bf00098a8242c18eaacea9c8a4add557bc`.
The integration checks embedding dimensions, normalized output,
repeatability, stop, and restart. Native tests exercise input limits,
framing, EOF shutdown, and cancellation around startup and a large batch.
These checks establish an execution path, not retrieval quality.

See [the architecture](docs/architecture.md) for ownership and lifecycle
transitions, [the protocol](docs/protocol.md) for bounds and ownership details, and
[the implementation plan](docs/next.md) for remaining work.
