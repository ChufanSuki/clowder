# RPC

Actors, Learners, Environments may distributed across processes and machines. We use RPC to communicate between these processes. We try to make this deep learning framework agnostic. But right now, I uses Moolib(a library for distributed ML training with PyTorch) to define and test a set of APIs. The backend may change in the future.

An `RPC` object calls remote function or accepts remote function from other `RPC` object. In convention, we call the caller `client` and the callee `server`. Because the `server` `RPC` object provides a remote function to be called, it also has the name `service`.

![ Remote Procedure Call Flow](https://www.ibm.com/docs/en/ssw_aix_71/commprogramming/figures/A12C0bb01.jpg)

```python
def foo(str):
    print(str)
    return 42

server = RPC()
server.set_name("server") # Set the unique name of the peer.
server.define("bar", foo) # Define a function to be available for peers to call.
client = RPC()

server.listen("127.0.0.1:1234")
client.connect("127.0.0.1:1234")

future = client.async_("server", "bar", "hello world")
print(future.result())
```

## Launchable

`Launchable` is a interface that defines `init_launcher` and `init_execution` methods.

## Remotable

`RemotableMeta` is a metaclass that automatically mark methods in a class as remote functions if they have the `__remote__` attribute set to True, and collects all remote method names into a list in the `__remote_methods__` attribute of the class. It creates a set called `remote_methods` and populates it with any methods defined in the `__remote_methods__` attribute in the class and any of its base classes. Any method has the `__remote__` attribute set to `True` also be added to the set.

