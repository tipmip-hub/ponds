"""
Utility functions and constants for PONDS.
"""

import inspect
from typing import get_origin, get_args, Union


def get_first_annotation_type(cls, arg_name: str):
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
