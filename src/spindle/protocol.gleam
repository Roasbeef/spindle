//// Bounded wire codec for the external embedding helper.
////
//// A single owner admits one request at a time. Frames contain big-endian
//// integers and IEEE float32 vectors; no native pointer enters the VM.

import gleam/bit_array
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result

/// Maximum payload bytes in either direction.
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
  Ready(dimensions: Int, context_tokens: Int)

  /// One normalized vector per input, in request order.
  Vectors(id: Int, dimensions: Int, values: List(List(Float)))

  /// One token count per input, with embedding special-token handling.
  TokenCounts(id: Int, counts: List(Int))

  /// A native request or startup failure.
  Failure(id: Int, reason: String)
}

/// Encode a bounded batch without trusting native input checks alone.
///
/// ## Examples
///
/// ```gleam
/// request(Embed, 1, ["hello"])
/// ```
pub fn request(
  operation: Operation,
  id: Int,
  texts: List(String),
) -> Result(BitArray, String) {
  use _ <- result.try(valid(id > 0 && id <= 4_294_967_295, "invalid request ID"))
  let count = list.length(texts)
  use _ <- result.try(valid(
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
      use _ <- result.try(valid(size <= 65_536, "text exceeds 65536 bytes"))
      use _ <- result.try(valid(
        bit_array.byte_size(acc) + size + 4 <= max_frame,
        "batch exceeds frame limit",
      ))
      Ok(<<acc:bits, size:32, bytes:bits>>)
    }),
  )
  Ok(frame(body))
}

/// Encode shutdown, which the native reader handles during inference.
///
/// ## Examples
///
/// ```gleam
/// shutdown()
/// ```
pub fn shutdown() -> BitArray {
  frame(<<3:8, 0:32>>)
}

/// frame adds the wire header after the caller has bounded the payload.
fn frame(body: BitArray) -> BitArray {
  <<bit_array.byte_size(body):32, body:bits>>
}

/// valid converts a checked wire invariant into the decoder error chain.
fn valid(condition: Bool, reason: String) -> Result(Nil, String) {
  case condition {
    True -> Ok(Nil)
    False -> Error(reason)
  }
}

/// Accumulate one response, rejecting its length before body allocation.
/// An extra frame is unsolicited under the one-in-flight protocol.
///
/// ## Examples
///
/// ```gleam
/// feed(<<>>, <<0, 0>>)
/// ```
pub fn feed(
  buffer: BitArray,
  bytes: BitArray,
) -> Result(#(Option(Response), BitArray), String) {
  use _ <- result.try(valid(
    bit_array.byte_size(buffer) + bit_array.byte_size(bytes) <= max_frame + 4,
    "response exceeds frame limit",
  ))
  // Check the combined bound before concatenation. A fragmented body must
  // not turn several individually bounded chunks into an unbounded buffer.
  let combined = <<buffer:bits, bytes:bits>>
  case combined {
    <<size:32, body:bits>> -> {
      use _ <- result.try(valid(
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
///
/// ## Examples
///
/// ```gleam
/// decode(<<0:8, 1:16, 768:32, 2048:32>>)
/// ```
pub fn decode(body: BitArray) -> Result(Response, String) {
  case body {
    <<0:8, 1:16, dimensions:32, context:32>>
      if dimensions > 0 && dimensions <= 4096 && context > 0 && context <= 2048
    -> Ok(Ready(dimensions, context))
    <<129:8, id:32, count:32, dimensions:32, rest:bits>>
      if count > 0 && count <= 16 && dimensions > 0 && dimensions <= 4096
    -> {
      use _ <- result.try(valid(
        bit_array.byte_size(rest) == count * dimensions * 4,
        "vector payload length mismatch",
      ))
      // Exact byte length makes the recursive decode consume the whole
      // batch; trailing or missing vector components cannot be ignored.
      use values <- result.try(vectors(rest, count, dimensions, []))
      Ok(Vectors(id, dimensions, values))
    }
    <<130:8, id:32, count:32, rest:bits>> if count > 0 && count <= 16 -> {
      use _ <- result.try(valid(
        bit_array.byte_size(rest) == count * 4,
        "token payload length mismatch",
      ))
      use values <- result.try(counts(rest, []))
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

/// counts walks an already size-checked payload. Each count must fit the
/// same context limit enforced by native tokenization.
fn counts(bytes: BitArray, acc: List(Int)) -> Result(List(Int), String) {
  case bytes {
    <<>> -> Ok(list.reverse(acc))
    <<count:32, rest:bits>> if count > 0 && count <= 2048 ->
      counts(rest, [count, ..acc])
    _ -> Error("invalid token count")
  }
}

/// vectors consumes exactly the declared batch shape. The reversed
/// accumulator preserves request order without repeated list appends.
fn vectors(
  bytes: BitArray,
  remaining: Int,
  dimensions: Int,
  acc: List(List(Float)),
) -> Result(List(List(Float)), String) {
  case remaining {
    0 -> Ok(list.reverse(acc))
    _ -> {
      use pair <- result.try(floats(bytes, dimensions, [], 0.0))
      let #(vector, rest) = pair
      vectors(rest, remaining - 1, dimensions, [vector, ..acc])
    }
  }
}

/// floats decodes one vector while accumulating its squared norm. Invalid
/// IEEE encodings fail the float pattern; a finite but zero or non-unit vector
/// fails the terminal norm check before any vector is returned.
fn floats(
  bytes: BitArray,
  remaining: Int,
  acc: List(Float),
  squared: Float,
) -> Result(#(List(Float), BitArray), String) {
  case remaining, bytes {
    0, _ -> {
      use _ <- result.try(valid(
        squared >. 0.999 && squared <. 1.001,
        "embedding is not normalized",
      ))
      Ok(#(list.reverse(acc), bytes))
    }
    _, <<value:float-size(32), rest:bits>> ->
      floats(rest, remaining - 1, [value, ..acc], squared +. value *. value)
    _, _ -> Error("invalid embedding component")
  }
}
