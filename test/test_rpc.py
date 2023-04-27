import moolib

def foo(str):
    print(str)
    return 42

def test_rpc():
    host = moolib.Rpc()
    host.set_name("host")
    host.define("bar", foo)
    host.listen("127.0.0.1:1234")

    client = moolib.Rpc()
    client.connect("127.0.0.1:1234")

    future = client.async_("host", "bar", "hello world")
    assert future.result() == 42
