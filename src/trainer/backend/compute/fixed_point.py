# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Device-driven fixed-point accumulator formats shared by Python and Slang setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import slangpy as spy

ETA_FIXED_SCALE = 65536.0
INT32_DEFINE = "FIXED_POINT_INT32"


@dataclass(frozen=True)
class FixedPointFormat:
    dtype: type[np.int32] | type[np.int64]
    buffer_dtype: str
    scale: float


FIXED_POINT_INT64 = FixedPointFormat(dtype=np.int64, buffer_dtype="int64_t", scale=65536.0)
FIXED_POINT_INT32 = FixedPointFormat(dtype=np.int32, buffer_dtype="int", scale=4096.0)


def supports_int64_atomics(device: spy.Device) -> bool:
    """Whether ``device`` supports 64-bit integer buffer atomics in compute shaders."""
    if device.has_feature(spy.Feature.atomic_int64):
        return True
    # SlangPy's D3D12 backend omits Feature.atomic_int64 even when SM 6.6 int64 atomics work.
    return (
        device.info.api_name == "D3D12"
        and device.has_feature(spy.Feature.sm_6_6)
        and device.has_feature(spy.Feature.int64)
    )


def _session_defines(source: Any) -> dict[str, str] | None:
    session = getattr(source, "session", None)
    if session is None:
        session = getattr(source, "slang_session", source)
    desc = getattr(session, "desc", None)
    options = getattr(desc, "compiler_options", None)
    return getattr(options, "defines", None)


def _is_metal_source(source: Any, defines: dict[str, str] | None) -> bool:
    if defines is not None and "TRAINER_METAL_BACKEND" in defines:
        return True
    device = source if hasattr(source, "info") else getattr(source, "device", None)
    info = getattr(device, "info", None)
    return getattr(info, "api_name", None) == "Metal"


def fixed_point_format(source: Any) -> FixedPointFormat:
    """Resolve the fixed-point format for a SlangPy device, session, or module."""
    if source is None:
        return FIXED_POINT_INT64
    defines = _session_defines(source)
    if defines is not None and INT32_DEFINE in defines:
        return FIXED_POINT_INT32
    # Native Metal emulates exact 64-bit atomics for blend_weight/removal_cost and uses fp32 gradients.
    if _is_metal_source(source, defines):
        return FIXED_POINT_INT64
    device = source if hasattr(source, "has_feature") else getattr(source, "device", None)
    if device is not None and not supports_int64_atomics(device):
        return FIXED_POINT_INT32
    return FIXED_POINT_INT64


def fixed_point_shader_defines(device: spy.Device) -> dict[str, str]:
    """Return shader macros matching ``device``'s fixed-point accumulator format."""
    if fixed_point_format(device) is FIXED_POINT_INT32:
        return {INT32_DEFINE: "1"}
    return {}
