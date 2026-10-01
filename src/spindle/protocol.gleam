//// Bounded wire codec for the external embedding helper.
////
//// A single owner admits one request at a time. Frames contain big-endian
//// integers and IEEE float32 vectors; no native pointer enters the VM.
//// This module performs no I/O. The lifecycle owner decides whether a decoded
//// response belongs to its loaded model and current request.
////
//// ## Flow
////
//// `request` validates request identity, batch count, each UTF-8 byte length,
//// and the aggregate payload before `frame_payload` adds the header.
//// `shutdown` uses the same framing with an operation-independent tag.
////
//// `feed` receives the owner's partial buffer and a Port chunk. It bounds
//// their combined size before concatenation, then returns either an incomplete
//// frame or one `decode` result. Additional trailing bytes are unsolicited.
////
//// `decode` matches a complete payload by tag and exact byte shape.
//// `decode_counts_loop` checks token bounds; `decode_vectors_loop` divides a
//// batch into rows, and `decode_vector_loop` checks each row's squared norm.
//// Each recursive decoder prepends into an accumulator and reverses once,
//// preserving wire order without repeated list appends.

import gleam/bit_array
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result

/// Maximum payload bytes in either direction, excluding the four-byte header.
pub const max_frame = 1_048_576

/// The native operations over already formatted text.
pub type Operation {
  /// Produce one normalized vector per input.
  Embed

  /// Count the tokens used by the embedding operation.
  CountTokens
}

/// A fully decoded native response.
pub type Response {
  /// Protocol version one and the loaded model's effective bounds.
  Ready(
    /// Output width, between 1 and 4,096 components.
    dimensions: Int,
    /// Effective input limit, between 1 and 2,048 tokens.
    context_tokens: Int,
  )

  /// One normalized vector per input, in request order.
  Vectors(
    /// Wire identity; the lifecycle owner must match its current request.
    id: Int,
    /// Validated row width, still checked against the loaded model.
    dimensions: Int,
    /// Finite rows whose squared norms lie strictly between 0.999 and 1.001.
    values: List(List(Float)),
  )

  /// One token count per input, with embedding special-token handling.
  TokenCounts(
    /// Wire identity; the lifecycle owner must match its current request.
    id: Int,
    /// One to sixteen counts, each between 1 and 2,048 tokens.
    counts: List(Int),
  )

  /// A native request or startup failure.
  Failure(
    /// Zero denotes startup failure; other IDs are correlated by the owner.
    id: Int,
    /// A UTF-8 diagnostic of at most 4,096 bytes, with no partial success.
    reason: String,
  )
}

/// Encode a bounded batch without trusting native input checks alone.
/// Request IDs are nonzero u32 values; batches contain one to sixteen texts.
/// Each text is at most 65,536 UTF-8 bytes and the combined payload is bounded
/// by `max_frame`. This function does not tokenize or apply model templates.
///
/// ## Examples
///
/// ```gleam
/// protocol.request(protocol.Embed, 1, ["hello"])
/// // -> Ok(frame) with a length header and one counted UTF-8 input.
/// ```
pub fn request(
  operation: Operation,
  id: Int,
  texts: List(String),
) -> Result(BitArray, String) {
  use _ <- result.try(validate_wire_invariant(
    id > 0 && id <= 4_294_967_295,
    "invalid request ID",
  ))
  let count = list.length(texts)
  use _ <- result.try(validate_wire_invariant(
    count > 0 && count <= 16,
    "batch must contain 1 to 16 texts",
  ))
  let tag = case operation {
    Embed -> 1
    CountTokens -> 2
  }
  use body <- result.try(
    list.try_fold(texts, <<tag:8, id:32, count:32>>, fn(acc, text) {
      let bytes = bit_array.from_string(text)
      let size = bit_array.byte_size(bytes)
      use _ <- result.try(validate_wire_invariant(
        size <= 65_536,
        "text exceeds 65536 bytes",
      ))
      use _ <- result.try(validate_wire_invariant(
        bit_array.byte_size(acc) + size + 4 <= max_frame,
        "batch exceeds frame limit",
      ))
      Ok(<<acc:bits, size:32, bytes:bits>>)
    }),
  )
  Ok(frame_payload(body))
}

/// Encode shutdown, which the native reader handles during inference.
///
/// ## Examples
///
/// ```gleam
/// assert protocol.shutdown() == <<5:32, 3:8, 0:32>>
/// ```
pub fn shutdown() -> BitArray {
  frame_payload(<<3:8, 0:32>>)
}

/// Accumulate one response, bounding its header and combined buffer size.
/// An extra frame is unsolicited under the one-in-flight protocol.
/// `Ok(#(None, buffer))` retains an incomplete header or body for the next
/// chunk. `Ok(#(Some(response), <<>>))` consumes exactly one complete frame.
/// Bounds apply before this module concatenates chunks; they cannot prevent
/// the BEAM runtime from allocating the incoming Port message itself.
///
/// ## Examples
///
/// ```gleam
/// assert protocol.feed(<<>>, <<0, 0>>) == Ok(#(None, <<0, 0>>))
/// ```
pub fn feed(
  buffer: BitArray,
  bytes: BitArray,
) -> Result(#(Option(Response), BitArray), String) {
  use _ <- result.try(validate_wire_invariant(
    bit_array.byte_size(buffer) + bit_array.byte_size(bytes) <= max_frame + 4,
    "response exceeds frame limit",
  ))

  // Check the combined bound before concatenation. A fragmented body must
  // not turn several individually bounded chunks into an unbounded buffer.
  let combined = <<buffer:bits, bytes:bits>>
  case combined {
    <<size:32, body:bits>> -> {
      use _ <- result.try(validate_wire_invariant(
        size > 0 && size <= max_frame,
        "invalid response frame length",
      ))
      case bit_array.byte_size(body) {
        n if n < size -> Ok(#(None, combined))
        n if n == size ->
          decode(body) |> result.map(fn(value) { #(Some(value), <<>>) })
        _ -> Error("unsolicited trailing response bytes")
      }
    }
    _ -> Ok(#(None, combined))
  }
}

/// Decode a complete response with bounded dimensions and counts.
/// Every tag requires its full payload shape; trailing bytes and malformed
/// float encodings fail explicitly. Request identity and agreement with the
/// loaded model remain the lifecycle owner's responsibility.
///
/// ## Examples
///
/// ```gleam
/// assert protocol.decode(<<0:8, 1:16, 768:32, 2048:32>>)
///   == Ok(protocol.Ready(768, 2048))
/// ```
pub fn decode(body: BitArray) -> Result(Response, String) {
  case body {
    <<0:8, 1:16, dimensions:32, context:32>>
      if dimensions > 0 && dimensions <= 4096 && context > 0 && context <= 2048
    -> Ok(Ready(dimensions, context))
    <<129:8, id:32, count:32, dimensions:32, rest:bits>>
      if count > 0 && count <= 16 && dimensions > 0 && dimensions <= 4096
    -> {
      use _ <- result.try(validate_wire_invariant(
        bit_array.byte_size(rest) == count * dimensions * 4,
        "vector payload length mismatch",
      ))

      // Exact byte length makes the recursive decode consume the whole
      // batch; trailing or missing vector components cannot be ignored.
      use values <- result.try(decode_vectors_loop(rest, count, dimensions, []))
      Ok(Vectors(id, dimensions, values))
    }
    <<130:8, id:32, count:32, rest:bits>> if count > 0 && count <= 16 -> {
      use _ <- result.try(validate_wire_invariant(
        bit_array.byte_size(rest) == count * 4,
        "token payload length mismatch",
      ))
      use values <- result.try(decode_counts_loop(rest, []))
      Ok(TokenCounts(id, values))
    }
    <<255:8, id:32, size:32, text:size(size)-bytes>> if size <= 4096 -> {
      use reason <- result.try(
        bit_array.to_string(text) |> result.replace_error("invalid error text"),
      )
      Ok(Failure(id, reason))
    }
    _ -> Error("invalid native response")
  }
}

/// Add the wire header after the caller has bounded the payload.
fn frame_payload(body: BitArray) -> BitArray {
  <<bit_array.byte_size(body):32, body:bits>>
}

/// Convert a checked wire invariant into an explicit protocol error.
fn validate_wire_invariant(
  condition: Bool,
  reason: String,
) -> Result(Nil, String) {
  case condition {
    True -> Ok(Nil)
    False -> Error(reason)
  }
}

/// Walk an already size-checked payload. Each count must fit the
/// same context limit enforced by native tokenization.
fn decode_counts_loop(
  bytes: BitArray,
  acc: List(Int),
) -> Result(List(Int), String) {
  case bytes {
    <<>> -> Ok(list.reverse(acc))
    <<count:32, rest:bits>> if count > 0 && count <= 2048 ->
      decode_counts_loop(rest, [count, ..acc])
    _ -> Error("invalid token count")
  }
}

/// Consume exactly the declared batch shape. The reversed
/// accumulator preserves request order without repeated list appends.
fn decode_vectors_loop(
  bytes: BitArray,
  remaining: Int,
  dimensions: Int,
  acc: List(List(Float)),
) -> Result(List(List(Float)), String) {
  case remaining {
    0 -> Ok(list.reverse(acc))
    _ -> {
      use pair <- result.try(decode_vector_loop(bytes, dimensions, [], 0.0))
      let #(vector, rest) = pair
      decode_vectors_loop(rest, remaining - 1, dimensions, [vector, ..acc])
    }
  }
}

/// Decode one vector while accumulating its squared norm. Invalid
/// IEEE encodings fail the float pattern; a finite but zero or non-unit vector
/// fails the terminal norm check before any vector is returned.
fn decode_vector_loop(
  bytes: BitArray,
  remaining: Int,
  acc: List(Float),
  squared: Float,
) -> Result(#(List(Float), BitArray), String) {
  case remaining, bytes {
    0, _ -> {
      use _ <- result.try(validate_wire_invariant(
        squared >. 0.999 && squared <. 1.001,
        "embedding is not normalized",
      ))
      Ok(#(list.reverse(acc), bytes))
    }
    _, <<value:float-size(32), rest:bits>> ->
      decode_vector_loop(
        rest,
        remaining - 1,
        [value, ..acc],
        squared +. value *. value,
      )
    _, _ -> Error("invalid embedding component")
  }
}
