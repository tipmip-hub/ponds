"""
Utility functions and constants for PONDS.
"""

import inspect
from typing import Optional, get_origin, get_args, Union, Any


def get_first_annotation_type(
        cls: object,
        arg_name: str,
        type_dict: Optional[dict[str, Any]] = None,
    ):
    """
    Function to extract the first non-NoneType annotation from a possibly
    Union or Optional type annotation of a class __init__ argument.

    Args:
        cls: The class whose __init__ method is to be inspected.
        arg_name: The name of the argument whose annotation is to be extracted.
    Returns:
        The first non-NoneType annotation if the argument is annotated with a Union
        or Optional type; otherwise, the original annotation.

    **Example Usage**:
        dtype = get_first_annotation_type(SIGMOID, "central_time")
        >> <int>
    """
    if type_dict is not None:
        from ponds import type_dict
    if type(cls) is str:
        cls = string_to_generator(cls, type_dict)
    sig = inspect.signature(cls.__init__)
    annotation = sig.parameters[arg_name].annotation

    # Case: no union involved → return directly
    origin = get_origin(annotation)
    args = []

    # Case 1: Optional/Union style
    if origin is Union:
        args = list(get_args(annotation))

    # Case 2: Python 3.10+ pipe unions → annotation is types.UnionType
    elif hasattr(annotation, "__args__"):
        args = list(annotation.__args__)

    # No union → return annotation as is
    if not args:
        return annotation

    # Remove NoneType if present
    args = [a for a in args if a is not type(None)]

    # If something remains, return first item
    if args:
        return args[0]

    # Only NoneType existed
    return type(None)

def generator_to_string(
    generator: object,
) -> str:
    """
    Converts a generator instance to a string representation of its class name.
    """
    return str(generator)

def string_to_generator(
    generator_str: str,
    type_dict: Optional[dict[str, Any]] = None,
) -> object:
    """
    Converts a string representation of a generator class name back to the class itself.
    """
    if type_dict is None:
        from ponds import type_dict

    return type_dict[generator_str.split(".")[-1]]