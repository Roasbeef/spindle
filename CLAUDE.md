# Spindle

Spindle is a Gleam library with an external C++ embedding helper. Keep
inference outside the BEAM VM. Use weft for actor and lifecycle machinery.
Keep FFI confined to src/spindle/internal and its small Erlang shim.

Use total decoders, explicit errors, bounded wire frames, and typed domain
states. No panics in library code. Document public APIs and subtle ownership
transitions with complete sentences. Preserve user changes. Commits use
Olaoluwa Osuntokun <laolu32@gmail.com> without attribution footers.

Run Gleam format, warning-free build, tests, native tests, and applicable
real-model tests before calling the implementation ready. Distinguish
request settlement from observed helper exit. Never claim cancellation
has drained a helper merely because a waiting caller timed out.
