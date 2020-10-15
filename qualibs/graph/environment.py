import functools
from typing import Dict, Union, Any
from collections.abc import Callable
from inspect import signature


def env_dependency(envmodule={}):
    def decorator_env_dependency(func):
        envmodule[func.__name__] = func

        @functools.wraps(func)
        def wrapper_x(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper_x

    return decorator_env_dependency


def env_resolve(fn, envmodule={}, cache={}):
    sig = signature(fn)
    args = []
    for pname, pobj in sig.parameters.items():
        if pname not in cache:
            cache[pname] = env_resolve(envmodule[pname], envmodule, cache)
        args.append(cache[pname])
    return lambda: fn(*args)


class Dependency:
    """
    A class to represent any piece of metadata relating to an experiment
    :param val: Handle to a resource or a concrete value
    :type val: any type convertible to string.
    :param getter: function to retrieve the dependency value
    """

    def __init__(self, name: str, val=None, getter: Callable = None):
        self.name = name
        self._val = val
        self._getter = getter

    def getter(self):
        if self._getter:
            return self._getter()
        else:
            return self._val


class ExecutionEnvironment:
    """
    class to save the experiment metadata
    """

    def __init__(self, dependency_dict: Dict[str, Dependency] = None):
        if dependency_dict is None:
            self.dependencies = {}
        else:
            self.dependencies = dependency_dict

    def __setattr__(self, name: str, value: Union[Dependency, Dict[str, Dependency]]) -> None:
        if isinstance(value, Dependency):
            self.dependencies[name] = value
        elif isinstance(value, Dict) & all([isinstance(val, Dependency) for val in value.values()]):
            self.dependencies = {**self.dependencies, **value}
        else:
            raise TypeError

    def __delattr__(self, name: str) -> None:
        del self.dependencies[name]

    def __getattribute__(self, name: str) -> Any:
        return self.dependencies[name]

    def get_state(self):
        return {key: self.dependencies[key].getter() for key in self.dependencies.keys()}

