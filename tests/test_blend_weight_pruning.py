# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Blend-weight pruning (``--prune-blend-weight``).

The backward sums each splat's blend weight alpha * T into the densify window;
checked against the forward alone. A densify tick then prunes the splats whose
weight per visible view is under the threshold, on both densify paths, and keeps
the ones the window never showed.
"""

from __future__ import annotations

import numpy as np
import pytest
from _pipeline_testkit import TestPipeline

from trainer.backend.compute.device import create_device
from trainer.backend.densify.removal import accumulate_removal_cost, prune_by_removal_cost
from trainer.config.config import DensifyConfig, StructureConfig
from trainer.config.config import DensificationProperties, IterationContextualVariables
from trainer.data.dataset import simple_camera

BACKENDS = ["vulkan", "d3d12", "metal"]
SIZE = 32
FIXED_SCALE = 65536.0


def _pipeline(backend: str) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # pragma: no cover - backend availability
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(device=device, width=SIZE, height=SIZE, max_splats=16)


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


def _blend_weight(pipeline: TestPipeline, count: int) -> np.ndarray:
    raw = pipeline.buffers.densification.blend_weight.to_numpy().tobytes()
    return np.frombuffer(raw, dtype=np.int64)[:count] / FIXED_SCALE


def _backward(pipeline: TestPipeline, camera, track: bool) -> None:
    contextual = IterationContextualVariables(
        hardness_value=1.0, sh_degree=0,
        densification_variables=DensificationProperties(track_blend_weight=track))
    with pipeline.commands.recording("blend_weight"):
        pipeline.rasterize(camera, contextual)
        pipeline.record_backward(contextual)
    pipeline.runtime.wait_for_idle()


@pytest.mark.parametrize("backend", BACKENDS)
def test_backward_tracks_the_blend_weight(backend: str) -> None:
    pipeline = _pipeline(backend)
    camera = simple_camera(width=SIZE, height=SIZE, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)
    front = _alpha(pipeline, camera, _splats([0.0], [2.0], [0.6]))
    back = _alpha(pipeline, camera, _splats([0.1], [3.0], [0.8]))
    pipeline.set_primitive_from_numpy(**_splats([0.0, 0.1, 50.0], [2.0, 3.0, 2.0],
                                                [0.6, 0.8, 0.9]))
    pipeline.densify_stats.clear()

    _backward(pipeline, camera, track=True)
    _backward(pipeline, camera, track=True)
    _backward(pipeline, camera, track=False)

    expected = 2.0 * np.array([front.sum(), (back * (1.0 - front)).sum(), 0.0])
    np.testing.assert_allclose(_blend_weight(pipeline, 3), expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("structure", [False, True], ids=["simple", "structure"])
@pytest.mark.parametrize(("min_weight", "survivors"),
                         [(0.0, [-3.0, -1.0, 1.0, 3.0]), (1.0, [-1.0, 1.0, 3.0])],
                         ids=["off", "on"])
def test_densify_prunes_underweight_splats(backend: str, structure: bool,
                                           min_weight: float,
                                           survivors: list[float]) -> None:
    pipeline = _pipeline(backend)
    x = [-3.0, -1.0, 1.0, 3.0]
    pipeline.set_primitive_from_numpy(**_splats(x, [5.0] * 4, [0.9] * 4))
    # Per visible view: 0.5 px (pruned), 2 px (kept), never shown (kept), 50 px.
    views = np.zeros(pipeline.max_splats, np.uint32)
    views[:4] = [4, 4, 0, 2]
    weight = np.zeros(pipeline.max_splats, np.int64)
    weight[:4] = np.array([2.0, 8.0, 0.0, 100.0]) * FIXED_SCALE
    pipeline.buffers.densification.visibility_count.copy_from_numpy(views)
    pipeline.buffers.densification.blend_weight.copy_from_numpy(weight)

    if structure:
        pipeline.structure_densifier.densify(StructureConfig(opacity_cap=0.0), min_weight)
    else:
        pipeline.simple_densifier.run(DensifyConfig(), allow_scale_prune=False,
                                min_blend_weight=min_weight)

    pipeline.runtime.wait_for_idle()
    count = int(pipeline.buffers.primitive.active_count.to_numpy().view(np.uint32)[0])
    records = np.frombuffer(pipeline.buffers.primitive.values.to_numpy().tobytes(),
                            dtype=pipeline.buffers.primitive.layout.dtype)[:count]
    means = records["mean"][:, 0]
    np.testing.assert_allclose(np.sort(means), survivors, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_removal_cost_matches_rendering_without_each_splat(backend: str) -> None:
    pipeline = _pipeline(backend)
    camera = simple_camera(width=SIZE, height=SIZE, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)
    scene = _splats([0.0, 0.1, 50.0], [2.0, 3.0, 2.0], [0.65, 0.4, 0.9])
    scene["color_logits"][:2] = [[1, -1, 0], [0, 1, -1]]

    def render(keep):
        pipeline.set_primitive_from_numpy(**{k: v[keep] for k, v in scene.items()})
        pipeline.render(camera)
        return pipeline.read_output_image()[..., :3].astype(np.float64)

    omissions = [render([1, 2]), render([0, 2])]
    image = render([0, 1, 2])
    expected = [np.sum((image - omitted) ** 2) for omitted in omissions]
    pipeline.densify_stats.clear()
    for phase in range(16):
        accumulate_removal_cost(pipeline, phase, 1.0)
    scores = pipeline.buffers.densification.blend_weight.to_numpy().view(np.int64).reshape(-1)[:3]
    np.testing.assert_allclose(scores / 2**30, [*expected, 0], atol=1e-5, rtol=2e-4)

    stats = prune_by_removal_cost(pipeline, 2 / 3)
    assert stats == {"pre_count": 3, "target_count": 2, "final_count": 2, "unobserved_count": 1}
    keep = [int(np.argmax(expected)), 2]
    np.testing.assert_allclose(pipeline.primitives.read()["mean"], scene["means"][keep])
