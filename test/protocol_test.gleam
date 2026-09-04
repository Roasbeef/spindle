import gleam/bit_array
import gleam/list
import gleam/option.{None, Some}
import gleam/result
import gleam/string
import spindle/protocol

pub fn fragmented_greeting_is_reassembled_test() {
  let greeting = <<11:32, 0:8, 1:16, 768:32, 2048:32>>
  list.each([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], fn(offset) {
    let assert Ok(head) = bit_array.slice(greeting, 0, offset)
      as "prefix exists"
    let assert Ok(tail) = bit_array.slice(greeting, offset, 15 - offset)
      as "suffix exists"
    let assert Ok(#(None, buffer)) = protocol.feed(<<>>, head)
      as "partial frame waits"
    assert protocol.feed(buffer, tail)
      == Ok(#(Some(protocol.Ready(768, 2048)), <<>>))
  })
}

pub fn frame_limits_are_checked_from_the_header_test() {
  assert result.is_error(protocol.feed(<<>>, <<1_048_577:32>>))
  assert result.is_error(protocol.feed(<<>>, <<0:32>>))
  assert result.is_error(
    protocol.feed(<<>>, <<11:32, 0:8, 2:16, 768:32, 2048:32>>),
  )
}

pub fn malformed_vector_shapes_and_values_are_refused_test() {
  assert result.is_error(protocol.decode(<<129:8, 1:32, 17:32, 1:32>>))
  assert result.is_error(protocol.decode(<<129:8, 1:32, 1:32, 4097:32>>))
  assert result.is_error(
    protocol.decode(<<129:8, 1:32, 1:32, 2:32, 1.0:float-size(32)>>),
  )
  assert result.is_error(
    protocol.decode(<<129:8, 1:32, 1:32, 1:32, 0.0:float-size(32)>>),
  )
  assert result.is_error(
    protocol.decode(<<129:8, 1:32, 1:32, 1:32, 0x7FC00000:32>>),
  )
  assert result.is_error(
    protocol.decode(<<129:8, 1:32, 1:32, 1:32, 0x7F800000:32>>),
  )
  assert protocol.decode(<<
      129:8,
      1:32,
      1:32,
      2:32,
      1.0:float-size(32),
      0.0:float-size(32),
    >>)
    == Ok(protocol.Vectors(1, 2, [[1.0, 0.0]]))
}

pub fn request_bounds_apply_before_native_work_test() {
  assert result.is_error(protocol.request(protocol.Embed, 1, []))
  assert result.is_error(protocol.request(
    protocol.Embed,
    1,
    list.repeat("x", 17),
  ))
  assert result.is_error(protocol.request(protocol.Embed, 0, ["x"]))
  assert result.is_error(
    protocol.request(protocol.Embed, 1, [string.repeat("x", 65_537)]),
  )
}

pub fn token_counts_and_error_text_are_total_test() {
  assert result.is_error(protocol.decode(<<130:8, 1:32, 1:32, 2049:32>>))
  assert result.is_error(protocol.decode(<<255:8, 1:32, 1:32, 255:8>>))
  assert protocol.decode(<<255:8, 1:32, 4:32, "oops":utf8>>)
    == Ok(protocol.Failure(1, "oops"))
}
