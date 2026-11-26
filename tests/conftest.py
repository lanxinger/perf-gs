# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Shared pytest setup for the GPU math tests and standalone scripts."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import slangpy as spy

_RUNNING_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts" / "running")
if _RUNNING_SCRIPTS not in sys.path:
    sys.path.append(_RUNNING_SCRIPTS)


@pytest.fixture(autouse=True)
def close_test_devices():
    """Release per-test devices held by SlangPy's caches. Shared fixtures own their devices."""
    existing = set(spy.Device.get_created_devices())
    yield
    for device in spy.Device.get_created_devices():
        if device not in existing:
            device.close()


@pytest.fixture(scope="module", params=["vulkan", "d3d12", "metal"])
def gpu_device(request: pytest.FixtureRequest):
    from trainer.backend.compute.device import create_device

    try:
        device = create_device(backend=request.param)
    except RuntimeError as exc:
        pytest.skip(f"backend {request.param!r} unavailable: {exc}")
    try:
        yield device
    finally:
        device.close()


@pytest.fixture(scope="module")
def math_module(gpu_device):
    return spy.Module.load_from_file(gpu_device, str(Path(__file__).with_name("shaders") / "math_tests.slang"))
