# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Opacity reset/cap behavior shared by simple and structure densification."""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.backend.compute.device import create_device


@pytest.mark.parametrize("backend", ["vulkan", "d3d12", "metal"])
def test_opacity_cap_clamps_only_values_above_cap(backend: str) -> None:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    pipeline = TestPipeline(device=device, width=32, height=32, max_splats=32)
    probabilities = np.array([0.9, 0.5, 0.01], dtype=np.float32)
    pipeline.set_primitive_from_numpy(
        means=np.zeros((3, 3), dtype=np.float32),
        log_scales=np.log(np.full((3, 3), 0.01, dtype=np.float32)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (3, 1)),
        opacities=probabilities,
        color_logits=np.zeros((3, 3), dtype=np.float32),
    )

    cap = 0.05
    pipeline.cap_opacity(cap)
    pipeline.runtime.wait_for_idle()
    raw = pipeline.buffers.primitive.values.to_numpy()
    gaussians = np.frombuffer(raw.tobytes(), dtype=pipeline.buffers.primitive.layout.dtype)[:3]

    def logit(values):
        clipped = np.clip(values, 1e-5, 1.0 - 1e-5)
        return np.log(clipped / (1.0 - clipped))

    expected = np.minimum(logit(probabilities), logit(cap)).astype(np.float32)
    np.testing.assert_allclose(gaussians["opacity"], expected, atol=1e-5)
