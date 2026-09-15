"Utility functions for working with optimization config files."
from argparse import Namespace
from collections.abc import MutableMapping
from typing import Type, Dict, Any
import importlib
import copy
import inspect


def flatten_namespace(ns, parent_key="", separator="."):
    items = {}
    for k, v in ns.__dict__.items():
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, Namespace):
            items.update(flatten_namespace(v, new_key, separator=separator))
        elif isinstance(v, dict):
            items.update(
                flatten_namespace(Namespace(**v), new_key, separator=separator)
            )
        else:
            items[new_key] = v
    return items


def flatten(d, parent_key="", sep="."):
    """
    Recursively flatten a nested dictionary into a flat dictionary with dotted keys.
    
    Args:
        d: The dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: String to use for separating nested keys
    
    Returns:
        A flat dictionary where nested keys are joined with the separator
    
    Example:
        >>> d = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
        >>> flatten(d)
        {'a': 1, 'b.c': 2, 'b.d.e': 3}
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def convert_to_nested_dict(flat_dict):
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(".")
        d = nested_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
        if value == "True" or value == "False":
            d[keys[-1]] = value == "True"
    return nested_dict


def instantiate_class(input_dict: Dict[str, Any], *args, **kwargs):
    """
    Dynamically instantiate a class from a configuration dictionary.
    
    The input dictionary should contain:
    - 'class_path': A string with the full module path and class name (e.g., 'torch.nn.Linear')
    - 'init_args': Optional dictionary of arguments to pass to the class constructor
    
    Supports recursive instantiation: if any init_args values are themselves
    dictionaries with 'class_path', they will be instantiated first.
    
    Args:
        input_dict: Configuration dictionary with 'class_path' and optional 'init_args'
        *args: Positional arguments to pass to the class constructor
        **kwargs: Additional keyword arguments to pass to the class constructor
                 (merged with and override init_args)
    
    Returns:
        An instance of the specified class

    Example:
        >>> config = {
        ...     'class_path': 'collections.Counter',
        ...     'init_args': {'iterable': [1, 2, 2, 3]}
        ... }
        >>> instance = instantiate_class(config)
        >>> isinstance(instance, collections.Counter)
        True
    """
    class_path = input_dict["class_path"]
    init_args = copy.deepcopy(input_dict.get("init_args", {}))

    # Convert init_args to dictionary if it is a Namespace
    if isinstance(init_args, Namespace):
        init_args = vars(init_args)

    init_args.update(kwargs)  # merge extra arguments into init_args

    # Iterate over init_args, checking if any values are themselves classes to be instantiated
    for arg_name, arg_value in init_args.items():
        if isinstance(arg_value, dict) and "class_path" in arg_value:
            init_args[arg_name] = instantiate_class(arg_value)

    module_name, class_name = class_path.rsplit(".", 1)
    MyClass = getattr(importlib.import_module(module_name), class_name)
    instance = MyClass(*args, **init_args)  # passing extra arguments to the class

    return instance


def ctor_supports(param_name: str, cls: Type) -> bool:
    """    
    This function attempts to inspect the constructor signature to determine
    if the given parameter name is accepted.

    Args:
        param_name: The parameter name to check for
        cls: The class to inspect
    
    Returns:
        True if the parameter is supported, False otherwise or if inspection fails
    Example:
        >>> class MyClass:
        ...     def __init__(self, x, y=10):
        ...         pass
        >>> ctor_supports('x', MyClass)
        True
        >>> ctor_supports('z', MyClass)
        False
    """
    try:
        sig = inspect.signature(cls.__init__)
        return param_name in sig.parameters
    except Exception:
        try:
            return param_name in tuple(cls.__init__.__code__.co_varnames)
        except Exception:
            return False
