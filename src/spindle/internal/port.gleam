//// The confined binding to BEAM Port primitives. Policy remains in Gleam.
////
//// Gleam's Port API does not expose this combination of executable arguments,
//// OS PID discovery, nonblocking writes, and total runtime-message decoding.
//// The small `spindle_port` shim supplies those operations; the parent machine
//// owns admission, protocol validation, deadlines, and shutdown evidence.
////
//// ## Flow
////
//// `open` runs inside the machine initialiser, so the Port belongs directly to
//// that process. `event` decodes messages after its selector matches the owned
//// Port. `send` submits bytes without suspending the lifecycle owner; failure
//// retires the engine. `close` requests EOF teardown after an expired drain
//// wait, while successful stop still requires an `Exited(0)` event.

import gleam/dynamic.{type Dynamic}
import gleam/erlang/port.{type Port}

/// Raw transport events selected only for the owned Port.
/// The selector, rather than this decoder, establishes Port identity.
pub type Event {
  /// Protocol stdout bytes, which can contain a partial header or body.
  Bytes(BitArray)

  /// Observed OS process exit status; zero alone confirms orderly teardown.
  Exited(Int)

  /// A selected message with neither supported runtime shape.
  Invalid
}

/// Open the helper directly, without a shell, and return its Port and OS PID.
/// `erlang:open_port` keeps stderr separate from protocol stdout. The calling
/// process owns the Port, so its death closes the helper's stdin even when no
/// Gleam shutdown callback runs. Paths are passed as executable arguments.
///
/// ## Examples
///
/// ```gleam
/// port.open("/opt/spindle/spindle-helper", "/models/embeddinggemma.gguf")
/// // -> Ok(#(owned_port, native_pid)) or Error(reason).
/// ```
@external(erlang, "spindle_port", "open")
pub fn open(executable: String, model: String) -> Result(#(Port, Int), String)

/// Write with `erlang:port_command`'s `nosuspend` option.
/// A full buffer or closed Port returns an error, so inference submission
/// cannot suspend the machine and delay its stop or creator-death handling.
/// Successful submission does not acknowledge model work or native exit.
///
/// ## Examples
///
/// ```gleam
/// port.send(owned_port, protocol.shutdown())
/// // -> Ok(Nil) when submitted, or Error(reason).
/// ```
@external(erlang, "spindle_port", "send")
pub fn send(port: Port, bytes: BitArray) -> Result(Nil, String)

/// Close the Port with `erlang:port_close`, tolerating an already closed Port.
/// This requests stdin EOF, which the native reader treats as owner loss.
/// Return does not establish that the OS process has exited.
///
/// ## Examples
///
/// ```gleam
/// port.close(owned_port)
/// // -> Nil after requesting EOF teardown.
/// ```
@external(erlang, "spindle_port", "close")
pub fn close(port: Port) -> Nil

/// Convert a selected runtime message to a total typed event.
/// The shim checks binary stdout and integer exit-status shapes; unsupported
/// terms become `Invalid`. Match the owned Port in the selector before calling.
///
/// ## Examples
///
/// ```gleam
/// port.event(selected_message)
/// // -> Bytes(chunk), Exited(status), or Invalid.
/// ```
@external(erlang, "spindle_port", "event")
pub fn event(message: Dynamic) -> Event
