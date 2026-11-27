# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Platform smoke test.

For each supported backend (Vulkan primary, D3D12 cross-check), verify that:

1. A ``spy.Device`` can be created.
2. A trivial ``.slang`` shader compiles against it.
3. A compute kernel dispatches and mutates a buffer as expected.

If a backend is not available on the host (no driver / no runtime), the
parametrization is skipped rather than failing — the primary job of this
test is to catch regressions on the backends the host *does* support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

from trainer.backend.compute.device import create_device

BACKENDS = ["vulkan", "d3d12", "metal"]


def _try_create_device(backend: str) -> spy.Device:
    try:
        return create_device(backend)
    except Exception as exc:  # backend unavailable on host
        pytest.skip(f"backend {backend!r} unavailable: {exc}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_device_creates(backend: str) -> None:
    device = _try_create_device(backend)
    assert device is not None


@pytest.mark.parametrize("backend", BACKENDS)
def test_shader_compiles_and_dispatches(backend: str) -> None:
    device = _try_create_device(backend)
    module = spy.Module.load_from_file(device, str(Path(__file__).with_name("shaders") / "smoke.slang"))

    initial = np.arange(256, dtype=np.float32)
    buffer = spy.Tensor.from_numpy(device, initial)

    result_tensor = module.add_one(buffer)

    result = result_tensor.to_numpy()
    np.testing.assert_allclose(result, initial + 1.0)
