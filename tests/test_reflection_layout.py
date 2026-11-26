# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Reflection/layout freeze: the Python <-> shader struct contract (slice 5).

buffers.py sizes every persistent NDBuffer from slangpy reflection of the shared
shader structs, so their byte layout is a hard Python<->shader contract. This
freezes the reflected size/stride/alignment of each struct AND proves the
neutral shared module (``common/types.slang`` — now the reflection source used
by Pipeline) agrees byte-for-byte with the optimizer module that previously
provided them (``optimizers/adam.slang``, which merely ``#include``s
types.slang). That makes the reflection repoint provably layout-preserving.
"""

from __future__ import annotations

import pytest
import slangpy as spy

from trainer.backend.memory.buffers import _reflect_types
from trainer.backend.compute.device import SHADERS_DIR, create_device

BACKENDS = ["vulkan", "d3d12", "metal"]

# Frozen byte size per struct — scalar layout on Vulkan/D3D12, MSL 16-byte float3 alignment on Metal.
FROZEN_SIZES = {
    "Gaussian3D": 56,
    "Projected2DGaussian": 28,
    "BlendData": 40,
    "AdamState": 112,
}
FROZEN_SIZES_METAL = {
    "Gaussian3D": 80,
    "Projected2DGaussian": 28,
    "BlendData": 40,
    "AdamState": 112,
}


def _device(backend: str) -> spy.Device:
    try:
        return create_device(backend=backend)
    except Exception as exc:  # backend unavailable on host
        pytest.skip(f"backend {backend!r} unavailable: {exc}")


def _types_module(device: spy.Device) -> spy.Module:
    return spy.Module.load_from_file(device, str(SHADERS_DIR / "common" / "types.slang"))


@pytest.mark.parametrize("backend", BACKENDS)
def test_types_module_reflects_all_layout_structs(backend: str) -> None:
    """common/types.slang is a valid neutral reflection source for all 4 structs
    buffers.py needs (this is exactly what Pipeline now loads)."""
    t = _reflect_types(_types_module(_device(backend)))  # raises if any missing
    assert all(x is not None for x in (t.gaussian, t.projected, t.blend, t.adam))


@pytest.mark.parametrize("backend", BACKENDS)
def test_frozen_struct_sizes(backend: str) -> None:
    m = _types_module(_device(backend))
    expected = FROZEN_SIZES_METAL if backend == "metal" else FROZEN_SIZES
    for name, size in expected.items():
        bl = m.layout.find_type_by_name(name).buffer_layout
        assert bl.size == size, f"{name} size {bl.size} != frozen {size}"
        assert bl.stride == size, f"{name} stride {bl.stride} != frozen {size}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_types_matches_adam_reflection(backend: str) -> None:
    """The neutral module (types.slang) and the optimizer module (adam.slang,
    which #includes it) reflect byte-identical layouts, so repointing the
    reflection source from adam.slang to types.slang cannot change any buffer."""
    d = _device(backend)
    m_types = _types_module(d)
    m_adam = spy.Module.load_from_file(d, str(SHADERS_DIR / "optimizers" / "adam.slang"))
    for name in FROZEN_SIZES:
        a = m_adam.layout.find_type_by_name(name).buffer_layout
        t = m_types.layout.find_type_by_name(name).buffer_layout
        assert (t.size, t.stride, t.alignment) == (a.size, a.stride, a.alignment), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_gaussian_layout_dtype_matches_reflection(backend: str) -> None:
    layout = _reflect_types(_types_module(_device(backend))).gaussian_layout
    assert layout.float_count == (20 if backend == "metal" else 14)
    assert layout.dtype.names == ("mean", "log_scale", "rot_quat", "opacity", "sh0")
    fields = layout.dtype.fields
    expected_offsets = (
        [(0, (3,)), (16, (3,)), (32, (4,)), (48, ()), (64, (3,))]
        if backend == "metal"
        else [(0, (3,)), (12, (3,)), (24, (4,)), (40, ()), (44, (3,))]
    )
    assert [(fields[name][1], fields[name][0].shape) for name in layout.dtype.names] == expected_offsets
