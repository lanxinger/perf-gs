# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""The trainer's own GPU allocations, and fitting them to a VRAM budget."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import slangpy as spy

from trainer.backend.platform.dxgi import query_video_memory

GIB = 1024 ** 3


def _resource_bytes(resource: Any) -> int:
    resource = getattr(resource, "storage", resource)    # NDBuffer views report their buffer
    if isinstance(resource, (spy.Buffer, spy.Texture)):
        return int(resource.memory_usage.device)
    return 0


def allocated_bytes(resources: Any) -> dict[str, int]:
    """Device bytes slangpy reports per field of a dataclass tree of GPU resources.

    Portable (any backend, any OS); a dict field is summed into one entry.
    """
    sizes: dict[str, int] = {}

    def walk(node: Any, name: str) -> None:
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for field in dataclasses.fields(node):
                walk(getattr(node, field.name), f"{name}.{field.name}" if name else field.name)
            return
        size = (sum(map(_resource_bytes, node.values())) if isinstance(node, dict)
                else _resource_bytes(node))
        if size:
            sizes[name] = size

    walk(resources, "")
    return sizes


def available_bytes(budget_gib: float, adapter_name: str) -> int | None:
    """Bytes this run may still allocate, or None when no budget is known.

    ``budget_gib`` > 0 overrides the measured (Windows DXGI / macOS Metal) budget; a reserve
    is held back for driver overhead and allocations the estimate does not count.
    """
    measured = query_video_memory(adapter_name)
    if budget_gib > 0:
        budget = int(budget_gib * GIB)
    elif measured is not None:
        budget = measured[0]
    else:
        return None
    usage = measured[1] if measured is not None else 0
    # Shortcut: a fixed 10% reserve; re-measure if runs page or it proves wasteful.
    return budget - usage - max(GIB // 2, budget // 10)


def fit_capacity(requested: int, available: int, cost: Callable[[int], int]) -> int:
    """Largest capacity <= ``requested`` with ``cost(capacity) <= available``.

    ``cost`` must not decrease with capacity. Returns 0 when nothing fits.
    """
    low, high = 0, requested
    while low < high:
        middle = (low + high + 1) // 2
        if cost(middle) <= available:
            low = middle
        else:
            high = middle - 1
    return low
