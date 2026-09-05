%% This module exposes Port primitives only. The Gleam state machine owns
%% admission, response validation, deadlines, and acknowledgement ordering.
-module(spindle_port).
-export([open/2, send/2, close/1, event/1]).

%% Keep stdout exclusively for protocol bytes. Diagnostics retain stderr.
open(Executable, Model) ->
    try
        Port = erlang:open_port(
            {spawn_executable, unicode:characters_to_list(Executable)},
            [{args, [unicode:characters_to_list(Model)]}, binary, stream,
             exit_status, hide]
        ),
        {os_pid, Pid} = erlang:port_info(Port, os_pid),
        {ok, {Port, Pid}}
    catch error:Reason -> {error, unicode:characters_to_binary(io_lib:format("~p", [Reason]))}
    end.

%% Backpressure is an error, never a suspended lifecycle owner.
send(Port, Bytes) ->
    try erlang:port_command(Port, Bytes, [nosuspend]) of
        true -> {ok, nil};
        false -> {error, <<"native input buffer is full">>}
    catch error:badarg -> {error, <<"native port is closed">>}
    end.

%% close requests EOF teardown. It cannot establish that the OS process has
%% exited; only a selected exit_status event supplies that evidence.
close(Port) ->
    try erlang:port_close(Port) catch error:badarg -> ok end,
    nil.

%% event is a total decoder after the selector has matched the owned Port.
event({_Port, {data, Bytes}}) when is_binary(Bytes) -> {bytes, Bytes};
event({_Port, {exit_status, Status}}) when is_integer(Status) -> {exited, Status};
event(_) -> invalid.
