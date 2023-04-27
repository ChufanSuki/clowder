import abc
import functools

from typing import Any, Callable, List, Optional

import moolib

class RemotableMeta(abc.ABCMeta):
    def __new__(mcs, name, bases, attrs):
        remote_methods = set(attrs.get("__remote_methods__", []))
        for base in bases:
            remote_methods.update(getattr(base, "__remote_methods__", []))
        for method in attrs.values():
            if getattr(method, "__remote__", False):
                remote_methods.add(method.__name__)
        attrs["__remote_methods__"] = list(remote_methods)
        return super().__new__(mcs, name, bases, attrs)