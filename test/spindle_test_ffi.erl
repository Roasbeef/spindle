-module(spindle_test_ffi).
-export([cwd/0, env/1, alive/1, mailbox_size/0, helper_owner/1, monotonic_ms/0]).
cwd() -> {ok, Dir} = file:get_cwd(), unicode:characters_to_binary(Dir).
env(Name) ->
    case os:getenv(binary_to_list(Name)) of
        false -> {error, nil};
        Value -> {ok, unicode:characters_to_binary(Value)}
    end.
alive(Pid) when is_integer(Pid), Pid > 1 ->
    os:cmd("kill -0 " ++ integer_to_list(Pid) ++ " 2>/dev/null && printf live") == "live".

mailbox_size() ->
    {message_queue_len, Size} = process_info(self(), message_queue_len),
    Size.


%% Resolve the actual Port owner so tests can kill the machine itself, rather
%% than only its creator. This is test introspection, not a library API.
helper_owner(Pid) ->
    Owners = [Owner || Port <- erlang:ports(),
                      erlang:port_info(Port, os_pid) == {os_pid, Pid},
                      {connected, Owner} <- [erlang:port_info(Port, connected)]],
    case Owners of
        [Owner] -> {ok, Owner};
        _ -> {error, nil}
    end.

%% Measure API elapsed time independently of wall-clock adjustments.
monotonic_ms() -> erlang:monotonic_time(millisecond).
