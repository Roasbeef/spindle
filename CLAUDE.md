# Spindle

Spindle is a Gleam library with an external C++ embedding helper. Keep
inference outside the BEAM VM. Use weft for actor and lifecycle machinery.
Keep FFI confined to src/spindle/internal and its small Erlang shim.

Use total decoders, explicit errors, bounded wire frames, and typed domain
states. No panics in library code. Document public and private functions and types in the literate style:
explain ownership, ordering, invariants, and failure behavior, rather than
narrating syntax. Separate logical steps with explanatory comments and
blank lines. Use complete sentences. Preserve user changes. Commits use
Olaoluwa Osuntokun <laolu32@gmail.com> without attribution footers.

Run Gleam format, warning-free build, tests, native tests, and applicable
real-model tests before calling the implementation ready. Distinguish
request settlement from observed helper exit. Never claim cancellation
has drained a helper merely because a waiting caller timed out.

## Literate source layout

The audience is an experienced programmer learning Gleam. Make ownership,
ordering, invariants, and failure behavior readable while skimming the source.
The [architecture reader guide](docs/architecture.md#reading-the-implementation)
connects the entrypoints to the implementation. Preserve these conventions:

1. Large modules start with a `//// ## Flow` narrative naming their actual
   entrypoints and local helpers. Explain the path across ownership boundaries.
2. Put entrypoints before their callees where that helps the reader follow the
   operation. Keep small validation and recursive decoding helpers after the
   public encode/decode path; avoid extra indirection merely to arrange a file.
3. Qualify domain calls, such as `protocol.request`, `native.send`, and
   `sm.transition`, so the call site exposes the boundary being crossed.
4. Name private helpers for domain operations, such as `begin_drain` and
   `observe_exit`, rather than generic transformations or implementation steps.
5. Put state, message, effect, and action types before implementation where
   present. Document each constructor's valid data and each field's custody.
6. Major state machines carry compact transition tables covering every state
   and event constructor. Describe guards, failure paths, and ignored events;
   use actual handler transitions rather than illustrative unreachable paths.

Keep source Flow sections and tables aligned with their handlers whenever the
lifecycle changes. Do not infer successful native drain from Port closure or
request settlement: only an observed native exit supplies that evidence.
