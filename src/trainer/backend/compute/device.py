# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Device creation: a SlangPy device (Vulkan primary, D3D12 cross-check) with the
packaged or source ``shaders/`` directory on the Slang include path."""

from __future__ import annotations

import sys
from importlib.resources import files
from pathlib import Path
from typing import Literal

import slangpy as spy

from trainer.backend.compute.fixed_point import fixed_point_shader_defines

Backend = Literal["vulkan", "d3d12", "metal"]

_REPO_ROOT = Path(__file__).resolve().parents[4]
SHADERS_DIR = Path(str(files("trainer").joinpath("shaders")))
if not SHADERS_DIR.is_dir():
    SHADERS_DIR = _REPO_ROOT / "shaders"  # editable/source checkout

_BACKEND_MAP: dict[Backend, spy.DeviceType] = {
    "vulkan": spy.DeviceType.vulkan,
    "d3d12": spy.DeviceType.d3d12,
    "metal": spy.DeviceType.metal,
}

# Backend-level macros always preserved across run-specific Slang sessions on the same device.
BACKEND_DEFINES: frozenset[str] = frozenset({"TRAINER_METAL_BACKEND"})


def default_backend() -> Backend:
    return "metal" if sys.platform == "darwin" else "vulkan"


def _open_device(
    backend: Backend,
    enable_debug: bool,
    shader_debug_info: bool,
    defines: dict[str, str] | None,
) -> spy.Device:
    if not shader_debug_info and not defines:
        return spy.create_device(
            type=_BACKEND_MAP[backend],
            enable_debug_layers=enable_debug,
            include_paths=[SHADERS_DIR],
        )

    # spy.create_device() hides compiler_options; replicate its include paths and hot reload.
    compiler_options: dict = {
        "include_paths": [spy.SHADER_PATH, str(SHADERS_DIR)],
    }
    if shader_debug_info:
        compiler_options["debug_info"] = spy.SlangDebugInfoLevel.standard
    if defines:
        compiler_options["defines"] = dict(defines)
    return spy.Device(
        type=_BACKEND_MAP[backend],
        enable_debug_layers=enable_debug,
        enable_hot_reload=True,
        compiler_options=compiler_options,
    )


def create_device(
    backend: Backend | None = None,
    enable_debug: bool = False,
    shader_debug_info: bool = False,
    defines: dict[str, str] | None = None,
) -> spy.Device:
    """Create a SlangPy device with ``shaders/`` on the include path.

    ``shader_debug_info`` adds source-line mapping for Nsight without lowering optimization;
    ``defines`` are preprocessor macros injected into every shader compile. Non-Metal devices
    without 64-bit integer buffer atomics automatically fall back to ``FIXED_POINT_INT32=1``.
    """
    if backend is None:
        backend = default_backend()
    if backend not in _BACKEND_MAP:
        raise ValueError(f"unknown backend {backend!r}; expected one of {list(_BACKEND_MAP)}")

    resolved_defines: dict[str, str] | None = dict(defines) if defines else None
    if backend == "metal":
        resolved_defines = {"TRAINER_METAL_BACKEND": "1", **(resolved_defines or {})}

    device = _open_device(backend, enable_debug, shader_debug_info, resolved_defines)
    device_defines = fixed_point_shader_defines(device)
    if device_defines and not all(k in (resolved_defines or {}) for k in device_defines):
        device.close()
        device = _open_device(
            backend, enable_debug, shader_debug_info,
            {**device_defines, **(resolved_defines or {})},
        )
    return device


def create_slang_session(device: spy.Device, defines: dict[str, str] | None) -> spy.SlangSession:
    """A session compiling like the device's own, preserving backend and fixed-point defines."""
    # Copied field by field: desc.compiler_options is the device's live object, not a copy.
    # Preserve only backend-level and fixed-point macros, never run-specific specializations
    # from the device's initial session.
    own = device.slang_session.desc.compiler_options
    backend_defines = {
        k: v for k, v in (own.defines or {}).items() if k in BACKEND_DEFINES
    }
    merged = {**backend_defines, **fixed_point_shader_defines(device), **(defines or {})}
    if dict(own.defines or {}) == merged:
        return device.slang_session
    return device.create_slang_session(compiler_options=spy.SlangCompilerOptions({
        "include_paths": own.include_paths,
        "debug_info": own.debug_info,
        "defines": merged,
    }))
