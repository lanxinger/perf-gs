# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Cross-validate the transposed per-splat backward against the proven
per-pixel backward.

Both kernels must produce the same per-gaussian gradients (within
fixed-point + atomic-order noise) for the same scene. This is a stronger,
more direct correctness gate than the finite-difference test: it compares
the two analytic kernels head-to-head over a scene deep enough to exercise
multiple checkpoint buckets (>32 overlapping splats in a tile) and the
early-break mask.
"""

from __future__ import annotations

import numpy as np
import pytest
from _pipeline_testkit import TestPipeline

from trainer.backend.compute.device import create_device, create_slang_session
from trainer.config.config import LossConfig
from trainer.data.dataset import simple_camera

BACKENDS = ["vulkan", "d3d12", "metal"]

GRAD_NAMES = [
    "mean_x", "mean_y", "mean_z",
    "log_scale_x", "log_scale_y", "log_scale_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
    "opacity", "color_r", "color_g", "color_b",
]


def _deep_scene(n: int, seed: int):
    rng = np.random.default_rng(seed)
    means = np.zeros((n, 3), dtype=np.float32)
    means[:, 0] = rng.uniform(-0.2, 0.2, n)
    means[:, 1] = rng.uniform(-0.2, 0.2, n)
    means[:, 2] = rng.uniform(2.0, 4.0, n)
    log_scales = np.log(np.full((n, 3), 0.35, dtype=np.float32))
    rot = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1))
    opac = rng.uniform(0.3, 0.8, n).astype(np.float32)
    colors = rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)
    return {
        "means": means, "log_scales": log_scales, "rot_quats": rot,
        "opacities": opac, "color_logits": colors,
    }


def _target(h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy /= max(h - 1, 1)
    xx /= max(w - 1, 1)
    return np.stack([xx, yy, (1.0 - xx) * yy], axis=-1).astype(np.float32)


@pytest.mark.parametrize("backend", BACKENDS)
def test_large_live_bucket_dispatch(backend: str) -> None:
    """Native-resolution scenes can use more than 65,535 pairs of live buckets."""
    import slangpy as spy

    from trainer.backend.compute.device import SHADERS_DIR

    try:
        device = create_device(backend=backend)
    except RuntimeError as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    try:
        program = device.load_program(str(SHADERS_DIR / "rasterize/tile_binned/live_buckets.slang"),
                                      ["prepareBucketDispatchArgs"])
        kernel = device.create_compute_kernel(program)
        total = spy.NDBuffer(device=device, dtype="uint", shape=(1,))
        args = spy.NDBuffer(device=device, dtype="uint", shape=(3,))
        for count, expected in ((0, [1, 1, 1]), (131070, [65535, 1, 1]), (131071, [65535, 2, 1])):
            total.copy_from_numpy(np.array([count], dtype=np.uint32))
            kernel.dispatch(thread_count=[1, 1, 1], live_bucket_total=total.storage,
                            dispatch_args=args.storage, max_buckets=count)
            np.testing.assert_array_equal(args.to_numpy(), expected)
    finally:
        device.close()


def _grads_with(pipeline: TestPipeline, scene, camera):
    pipeline.set_primitive_from_numpy(**scene)
    pipeline.forward_and_loss(camera)
    pipeline.optimizer.zero_grad()
    pipeline.run_backward()
    pipeline.device.wait_for_idle()
    return pipeline.read_gradients()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(("per_bucket", "scan_mode"), [
    (False, 0), (True, 0), (True, 1),
], ids=["per-tile", "flat-buckets", "flat-separate-scan"])
def test_bucket_gradients_match_perpixel(backend: str, per_bucket: bool, scan_mode: int) -> None:
    try:
        device = create_device(backend=backend)
    except RuntimeError as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    # Both pipelines receive identical inputs. The kernel variant is selected at construction.
    size = 64
    scene = _deep_scene(100, seed=3)
    camera = simple_camera(width=size, height=size, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)
    def gradients(per_splat, per_bucket, scan_mode):
        session = (create_slang_session(device, {"BACKWARD_BLEND_SCAN_MODE": str(scan_mode)})
                   if scan_mode else device.slang_session)
        pipeline = TestPipeline(device=device, width=size, height=size, max_splats=128,
                                use_persplat_backward=per_splat, use_flat_bucket_backward=per_bucket,
                                slang_session=session, loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))
        pipeline.targets.set_image(_target(size, size))
        return _grads_with(pipeline, scene, camera)

    ref = gradients(False, False, 0)
    actual = gradients(True, per_bucket, scan_mode)
    assert np.any(ref["color_r"] != 0.0)
    for name in GRAD_NAMES:
        # fp16 checkpoints, atomic order and the order of floating-point summation differ.
        tolerance = max(1e-4, 0.02 * np.max(np.abs(ref[name])))
        np.testing.assert_allclose(actual[name], ref[name], atol=tolerance, rtol=0.0, err_msg=name)


@pytest.mark.parametrize("backend", BACKENDS)
def test_projection_backward_resets_the_seeds(backend: str) -> None:
    """projection_backward consumes the 2D seeds and leaves them zero, so a
    second backward without any clear yields the same geometry gradients."""
    try:
        device = create_device(backend=backend)
    except RuntimeError as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W = H = 64
    pipeline = TestPipeline(device=device, width=W, height=H, max_splats=128,
                            loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0))
    pipeline.targets.set_image(_target(H, W))
    camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)

    first = _grads_with(pipeline, _deep_scene(100, seed=3), camera)
    for name, buffer in pipeline.buffers.gradients.seeds.items():
        seeds = np.frombuffer(buffer.to_numpy().tobytes(), dtype=np.int64)
        assert not seeds.any(), f"seed {name} not reset by projection_backward"

    pipeline.forward_and_loss(camera)
    pipeline.run_backward()
    pipeline.device.wait_for_idle()
    second = pipeline.read_gradients()
    assert np.any(first["mean_x"] != 0.0) and np.any(first["log_scale_x"] != 0.0)
    for name in GRAD_NAMES[:10]:   # geometry grads: stored, not accumulated
        np.testing.assert_array_equal(second[name], first[name], err_msg=name)
