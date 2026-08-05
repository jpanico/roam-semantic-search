"""Type-narrowing predicates for JSON-decoded values.

A JSON-decoded value arrives as ``object``; these :data:`~typing.TypeIs` predicates
narrow it structurally (JSON object keys are strings by construction), so strictly
typed code can walk decoded structures without casts.

Public symbols:

- :func:`is_json_object` — narrow to ``dict[str, object]``.
- :func:`is_json_list` — narrow to ``list[object]``.
"""

from typing import TypeIs

# These are typing constructs in the mold of `isinstance` itself, called on hot paths;
# they carry no @validate_call (pydantic cannot build a validator for a TypeIs form).


def is_json_object(value: object) -> TypeIs[dict[str, object]]:
    """Whether *value* is a JSON object (a dict; JSON keys are strings).

    Args:
        value: The JSON-decoded value to test.

    Returns:
        ``True`` when *value* is a dict, narrowing it to ``dict[str, object]``.
    """
    return isinstance(value, dict)


def is_json_list(value: object) -> TypeIs[list[object]]:
    """Whether *value* is a JSON array (a list).

    Args:
        value: The JSON-decoded value to test.

    Returns:
        ``True`` when *value* is a list, narrowing it to ``list[object]``.
    """
    return isinstance(value, list)
