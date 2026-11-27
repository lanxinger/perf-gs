# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Check the renderer's colour gradient against the visible opacity of each splat."""

from __future__ import annotations

import numpy as np
import pytest
from _pipeline_testkit import TestPipeline

from trainer.backend.compute.device import create_device
from trainer.data.dataset import simple_camera

SIZE = 32


def _splats(x: list[float], z: list[float], opacity: list[float]) -> dict:
    count = len(x)
    return {
        "means": np.stack([x, np.zeros(count), z], axis=1).astype(np.float32),
        "log_scales": np.log(np.full((count, 3), 0.3, np.float32)),
        "rot_quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (count, 1)),
        "opacities": np.asarray(opacity, np.float32),
        "color_logits": np.zeros((count, 3), np.float32),
    }


def _alpha(pipeline: TestPipeline, camera, splats: dict) -> np.ndarray:
    pipeline.set_primitive_from_numpy(**splats)
    pipeline.render(camera)
    return pipeline.read_output_image()[..., 3].astype(np.float64)


@pytest.mark.parametrize("backend", ["vulkan", "d3d12", "metal"])
def test_red_sh_gradient_matches_visible_blend_weight(backend: str) -> None:
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # pragma: no cover - backend availability
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    pipeline = TestPipeline(device=device, width=SIZE, height=SIZE, max_splats=16)
    camera = simple_camera(width=SIZE, height=SIZE, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)
    front = _alpha(pipeline, camera, _splats([0.0], [2.0], [0.6]))
    back = _alpha(pipeline, camera, _splats([0.1], [3.0], [0.8]))

    pipeline.set_primitive_from_numpy(**_splats([0.0, 0.1, 50.0], [2.0, 3.0, 2.0],
                                                [0.6, 0.8, 0.9]))
    pipeline.optimizer.zero_grad()
    image_gradient = np.zeros((SIZE, SIZE, 4), dtype=np.float32)
    image_gradient[..., 0] = 1.0
    pipeline.buffers.loss.image_gradient.copy_from_numpy(image_gradient)
    with pipeline.commands.recording("occlusion_gradient"):
        pipeline.rasterize(camera)
        pipeline.run_backward()
    pipeline.runtime.wait_for_idle()

    visible_opacity = np.array([front.sum(), (back * (1.0 - front)).sum(), 0.0])
    sh_dc_basis = 1.0 / np.sqrt(4.0 * np.pi)
    np.testing.assert_allclose(pipeline.read_gradients()["color_r"] / sh_dc_basis,
                               visible_opacity, rtol=1e-3, atol=1e-3)
