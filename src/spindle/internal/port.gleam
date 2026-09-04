//// The confined binding to BEAM Port primitives. Policy remains in Gleam.

import gleam/dynamic.{type Dynamic}
import gleam/erlang/port.{type Port}

/// Raw transport events selected only for the owned port.
pub type Event {
  /// Protocol stdout bytes.
  Bytes(BitArray)
  /// Observed OS process exit status.
  Exited(Int)
  /// An unexpected port message.
  Invalid
}

/// Open the packaged helper directly, without a shell.
@external(erlang, "spindle_port", "open")
pub fn open(executable: String, model: String) -> Result(#(Port, Int), String)

/// Write without blocking the actor on a full port buffer.
@external(erlang, "spindle_port", "send")
pub fn send(port: Port, bytes: BitArray) -> Result(Nil, String)

/// Close stdin. Native EOF handling terminates the external process.
@external(erlang, "spindle_port", "close")
pub fn close(port: Port) -> Nil

/// Convert runtime messages to a total typed event.
@external(erlang, "spindle_port", "event")
pub fn event(message: Dynamic) -> Event
