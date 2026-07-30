"""Private QUA protobuf helpers (betterproto and classic google.protobuf)."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Dict, Iterator, Optional, Tuple

from betterproto import Message


def _is_proto_message(message: Any) -> bool:
    return isinstance(message, Message) or (
        hasattr(message, "DESCRIPTOR") and hasattr(message, "ListFields")
    )


def _get_attribute(message: Any, *names: str, default: Any = ...):
    for name in names:
        if hasattr(message, name):
            return getattr(message, name)
    if default is not ...:
        return default
    raise AttributeError(f"{type(message).__name__} has none of the attributes {names!r}.")


def _set_attribute(message: Any, value: Any, *names: str) -> None:
    for name in names:
        if hasattr(message, name):
            setattr(message, name, value)
            return
    raise AttributeError(f"{type(message).__name__} has none of the attributes {names!r}.")


def _active_oneof_fields(message: Message) -> Dict[str, str]:
    return dict(getattr(message, "_group_current", {}))


def _field_is_set(message: Message, field_name: str) -> bool:
    oneof_group = message._betterproto.oneof_group_by_field.get(field_name)
    if oneof_group is None:
        return True
    return _active_oneof_fields(message).get(oneof_group) == field_name


def _message_field_values(message: Any) -> Iterator[Tuple[str, Any]]:
    if isinstance(message, Message):
        for field in dataclasses.fields(message):
            field_name = field.name
            if field_name.startswith("_"):
                continue
            if not _field_is_set(message, field_name):
                continue
            try:
                value = getattr(message, field_name)
            except AttributeError:
                continue
            if value is None:
                continue
            yield field_name, value
        return

    if hasattr(message, "ListFields"):
        for field_descriptor, value in message.ListFields():
            if value is None:
                continue
            yield field_descriptor.name, value


def _is_repeated_field(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return True
    return (
        hasattr(value, "__len__")
        and hasattr(value, "__getitem__")
        and not isinstance(value, (str, bytes, dict))
        and not _is_proto_message(value)
    )


def _walk_messages(message: Any, path: str = "program") -> Iterator[Tuple[str, Any]]:
    """Depth-first traversal through every populated QUA protobuf message."""
    yield path, message
    if not _is_proto_message(message):
        return

    for field_name, value in _message_field_values(message):
        if _is_repeated_field(value):
            for index, item in enumerate(value):
                yield from _walk_messages(item, f"{path}.{field_name}[{index}]")
        elif _is_proto_message(value):
            yield from _walk_messages(value, f"{path}.{field_name}")


def _statement_type(message: Any) -> Optional[str]:
    if isinstance(message, Message):
        active_oneofs = _active_oneof_fields(message)
        for oneof_name in ("statement", "statement_oneof"):
            if oneof_name in active_oneofs:
                return active_oneofs[oneof_name]
        return None

    if hasattr(message, "WhichOneof"):
        for oneof_name in ("statement_oneof", "statement"):
            try:
                active = message.WhichOneof(oneof_name)
            except ValueError:
                continue
            if active:
                return active
    return None


def _all_strings(message: Any) -> Iterator[str]:
    if isinstance(message, str):
        yield message
        return
    if _is_repeated_field(message):
        for item in message:
            yield from _all_strings(item)
        return
    if not _is_proto_message(message):
        return

    for field_name, value in _message_field_values(message):
        if isinstance(value, str):
            yield value
        else:
            yield from _all_strings(value)


def _clone_message(message: Any) -> Any:
    return copy.deepcopy(message)
