# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Mip-Splatting 3D smoothing filter: sane sizes, indirect dispatch, and gradients.

Unseen/empty slots get 0; a visible gaussian gets min_depth * sqrt(0.2) / max_focal.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from _pipeline_testkit import TestPipeline
from trainer.backend.compute.device import SHADERS_DIR, create_device
from trainer.backend.compute.gpu import CommandRecorder, GpuRuntime
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import MipResources, PrimitiveStorage
from trainer.backend.memory.layout import StructLayout
from trainer.backend.render.mip_filter import MipFilter
from trainer.data.dataset import simple_camera

BACKENDS = ["vulkan", "d3d12", "metal"]


def _setup(backend: str, max_splats: int = 64) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(device=device, width=64, height=64, max_splats=max_splats)


@pytest.mark.parametrize("backend", BACKENDS)
def test_compute_3d_filter_basic(backend: str) -> None:
    p = _setup(backend)
    # Two gaussians in front of a camera at z=0 looking toward +z.
    means = np.array([[0.0, 0.0, 3.0], [0.5, -0.3, 4.0]], dtype=np.float32)
    p.set_primitive_from_numpy(
        means=means,
        log_scales=np.log(np.full((2, 3), 0.1, dtype=np.float32)),
        rot_quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (2, 1)),
        opacities=np.array([0.9, 0.9], dtype=np.float32),
        color_logits=np.zeros((2, 3), dtype=np.float32),
    )
    cam = simple_camera(width=64, height=64, eye=(0.0, 0.0, 0.0),
                        target=(0.0, 0.0, 3.0))

    p.mip_filter.set_cameras([cam])
    p.mip_filter.compute()
    p.device.wait_for_idle()

    f = np.frombuffer(p.buffers.primitive.filter_3d.to_numpy().tobytes(), dtype=np.float32)
    live = f[:2]
    dead = f[2:]

    assert np.all(np.isfinite(f)), "filter has non-finite values"
    assert np.all(live > 0.0), f"seen gaussians must get a positive filter: {live}"
    assert np.all(live < 1.0), f"filter magnitude unreasonably large: {live}"
    # Farther gaussian (z=4) should get a larger low-pass than the nearer (z=3).
    assert live[1] > live[0], f"expected filter to grow with depth: {live}"
    # Empty slots stay zero.
    assert np.all(dead == 0.0), "empty slots must remain zero"


@pytest.mark.parametrize("backend", BACKENDS)
def test_filter_indirect_live_count_and_dead_tail(backend: str) -> None:
    """Cross the 65,535-group boundary, then shrink to an empty population.

    Only filter resources are needed here; avoid allocating a complete second
    training pipeline just to exercise a large dispatch and its tail clear.
    """
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    capacity, live = 100_000, 65_539
    module = spy.Module.load_from_file(device, str(SHADERS_DIR / "common/types.slang"))
    gaussian = module.layout.find_type_by_name("Gaussian3D")
    layout = StructLayout.of(gaussian)
    primitive = PrimitiveStorage(
        spy.NDBuffer(device=device, dtype=gaussian, shape=(capacity,)),
        spy.NDBuffer(device=device, dtype="uint", shape=(1,)),
        spy.NDBuffer(device=device, dtype="float", shape=(capacity,)), layout)
    resources = MipResources(
        dispatch_args=spy.NDBuffer(device=device, dtype="uint", shape=(6,), usage=(
            spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
            | spy.BufferUsage.indirect_argument)),
        previous_count=spy.NDBuffer(device=device, dtype="uint", shape=(2,)))
    resources.previous_count.copy_from_numpy(np.array([capacity, capacity], np.uint32))
    data = np.zeros((capacity, layout.float_count), np.float32)
    data[:live, :3] = np.random.default_rng(42).uniform(-0.5, 0.5, (live, 3))
    data[:live, 2] += 3.0
    primitive.values.storage.copy_from_numpy(data)
    primitive.active_count.copy_from_numpy(np.array([live], np.uint32))
    commands = CommandRecorder(GpuRuntime(device))
    mip_filter = MipFilter(device, commands, KernelLibrary(device), primitive=primitive,
                           resources=resources, capacity=capacity)

    mip_filter.set_cameras([simple_camera(width=64, height=64, eye=(0, 0, 0), target=(0, 0, 3))])
    primitive.filter_3d.copy_from_numpy(np.full(capacity, -123.0, np.float32))
    mip_filter.compute()
    commands.runtime.wait_for_submit(commands.last_submit_id)
    f = primitive.filter_3d.to_numpy()
    assert np.all(f[:live] > 0), "second dispatch row was not computed"
    assert np.all(f[live:] == 0), "dead slots were not cleared"
    np.testing.assert_array_equal(resources.dispatch_args.to_numpy()[:3],
                                  np.array([65535, 2, 1], np.uint32))

    primitive.active_count.copy_from_numpy(np.zeros(1, np.uint32))
    mip_filter.compute()
    commands.runtime.wait_for_submit(commands.last_submit_id)
    assert np.all(primitive.filter_3d.to_numpy() == 0), "stale filters survived shrink"
    device.close()


@pytest.mark.parametrize("backend", BACKENDS)
def test_mip_on_gradients_finite_difference(backend: str) -> None:
    """Finite-difference validate the backward with Mip-Splatting ON, including
    the DIFFERENTIABLE opacity compensation (coef3d/coef2d -> scale/cov/mean).

    filter_3D is no_diff (a geometric constant), so it is computed ONCE at the
    base params and held FIXED across every FD perturbation — exactly how the
    analytic backward treats it. If the FD instead recomputed filter_3D per
    perturbation it would include a d(filter)/d(mean) path the analytic (per
    mip-splatting) intentionally omits, and the test would spuriously fail.
    """
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    from test_gradients import _scene, _target, PARAM_TABLE
    from trainer.config.config import LossConfig, RasterConfig

    W = H = 16
    pipeline = TestPipeline(
        device=device, width=W, height=H, max_splats=16,
        loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
        raster_config=RasterConfig(mip_enabled=True, kernel_size=0.1),
    )
    camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)
    pipeline.targets.set_image(_target(H, W))

    scene = _scene()
    n = scene["means"].shape[0]

    # Upload base params, then compute the 3D filter ONCE and freeze it.
    pipeline.set_primitive_from_numpy(**scene)
    pipeline.mip_filter.set_cameras([camera])
    pipeline.mip_filter.compute()
    f = np.frombuffer(pipeline.buffers.primitive.filter_3d.to_numpy().tobytes(), dtype=np.float32)[:n]
    assert np.any(f > 0.0), f"mip filter should be non-zero for the test gaussians: {f}"

    # Analytic gradients (mip on, fixed filter, default per-splat backward).
    pipeline.forward_and_loss(camera)
    pipeline.optimizer.zero_grad()
    pipeline.run_backward()
    analytic = pipeline.read_gradients()

    def _loss(scn: dict) -> float:
        # set_gaussians uploads params but does NOT touch filter_3D (no
        # recompute), so the FD holds the filter fixed at its base value.
        pipeline.set_primitive_from_numpy(**scn)
        pipeline.forward_and_loss(camera)
        return pipeline.read_loss()

    # Smaller eps than the vanilla gradient test: the Mip 2D compensation
    # sqrt(det0/det1) adds curvature to the loss w.r.t. the projected position,
    # so the O(eps^2) central-difference error on the largest mean gradients is
    # bigger. 5e-4 keeps FD accurate without hitting float cancellation.
    eps = 5e-4
    mismatches: list[str] = []
    for grad_name, field_name, idx in PARAM_TABLE:
        col = analytic[grad_name]
        for gi in range(n):
            sp = {k: v.copy() for k, v in scene.items()}
            sm = {k: v.copy() for k, v in scene.items()}
            if grad_name == "opacity":
                pr = min(max(float(scene[field_name][gi]), 1e-5), 1.0 - 1e-5)
                bl = np.log(pr / (1.0 - pr))
                sp[field_name][gi] = 1.0 / (1.0 + np.exp(-(bl + eps)))
                sm[field_name][gi] = 1.0 / (1.0 + np.exp(-(bl - eps)))
            elif idx is None:
                sp[field_name][gi] += eps
                sm[field_name][gi] -= eps
            else:
                sp[field_name][gi, idx] += eps
                sm[field_name][gi, idx] -= eps

            num = (_loss(sp) - _loss(sm)) / (2 * eps)
            a = float(col[gi])
            abs_diff = abs(num - a)
            denom = max(abs(num), abs(a), 1e-2)
            # Looser than test_gradients (15% rel / 3e-2 abs): the Mip
            # sqrt(det0/det1) compensation adds curvature, so central-difference
            # noise near the tolerance boundary is larger. Still strict enough to
            # catch real bugs (sign flips ~200%, wrong-scale grads >50%).
            if abs_diff / denom > 0.15 and abs_diff > 3e-2:
                mismatches.append(
                    f"g{gi} {grad_name:14s}: analytic={a:+.4e} "
                    f"numerical={num:+.4e}  rel_err={abs_diff / denom:.3f}")

    assert not mismatches, ("mip-on gradient FD mismatches:\n" + "\n".join(mismatches))

