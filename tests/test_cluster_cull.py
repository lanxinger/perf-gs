# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU cluster-frustum cull.

Verifies that the buildClusterAABB + cullClustersFrustum + bin_count
pipeline correctly identifies clusters fully outside the camera frustum
and zeroes their tile counts.

Scene layout: one cluster of gaussians INSIDE the frustum (in front of
the camera) and one cluster OUTSIDE (far behind the camera). After a
forward pass with cluster_cull enabled:
- cluster_visible[in_cluster]  == 1
- cluster_visible[out_cluster] == 0
- tile_count_per_gaussian == 0 for every splat in the out cluster
- tile_count_per_gaussian > 0 for at least one splat in the in cluster
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.backend.memory.buffers import CLUSTER_SIZE
from trainer.config.config import RasterConfig
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device


BACKENDS = ["vulkan", "d3d12", "metal"]


def _setup(backend: str, max_splats: int, cluster_cull: bool) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(
        device=device, width=128, height=128, max_splats=max_splats,
        raster_config=RasterConfig(cluster_cull=cluster_cull),
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_cluster_cull_kills_offscreen_cluster(backend: str) -> None:
    # 2 clusters * 128 splats = 256 gaussians. Slots [0, 128) live at
    # z=+5 in front of the camera; slots [128, 256) live at z=-5 behind.
    n = 2 * CLUSTER_SIZE
    max_splats = 512  # round up so num_clusters = 4 (only first 2 are live)
    pipeline = _setup(backend, max_splats=max_splats, cluster_cull=True)

    means = np.zeros((n, 3), dtype=np.float32)
    means[:CLUSTER_SIZE, 2] = 5.0    # in-frustum cluster
    means[CLUSTER_SIZE:, 2] = -5.0   # behind-camera cluster
    # Small jitter so each splat has a distinct projection.
    means[:, 0] = np.linspace(-0.05, 0.05, n)
    means[:, 1] = np.linspace(-0.05, 0.05, n)

    log_scales = np.log(np.full((n, 3), 0.05, dtype=np.float32))
    rot_quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    opacities = np.full(n, 0.5, dtype=np.float32)
    color_logits = np.zeros((n, 3), dtype=np.float32)

    pipeline.set_primitive_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot_quats,
        opacities=opacities, color_logits=color_logits,
    )

    # Camera at origin looking down +z. In-cluster (z=5) is in front,
    # out-cluster (z=-5) is behind — a strict frustum cull.
    cam = simple_camera(width=128, height=128,
                        eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0))
    pipeline.rasterize(cam)
    pipeline.device.wait_for_idle()

    # Read cluster_visible.
    cv = pipeline.buffers.raster.cluster_visible.to_numpy().view(np.uint32)
    assert cv[0] == 1, f"in-frustum cluster wrongly culled: cv[0]={cv[0]}"
    assert cv[1] == 0, f"behind-camera cluster not culled: cv[1]={cv[1]}"

    # Read tile_count_per_gaussian.
    tc = pipeline.buffers.raster.tile_count_per_gaussian.to_numpy().view(np.uint32)
    # Behind-camera cluster: all 128 splats must have tile_count == 0.
    behind_counts = tc[CLUSTER_SIZE:2 * CLUSTER_SIZE]
    assert np.all(behind_counts == 0), (
        f"behind-camera splats emit tiles: max={behind_counts.max()}")
    # In-frustum cluster: at least one splat contributes tiles.
    in_counts = tc[:CLUSTER_SIZE]
    assert int(in_counts.sum()) > 0, (
        "in-frustum cluster produced zero tiles — cluster_cull may have "
        "over-culled or the scene is degenerate")


@pytest.mark.parametrize("backend", BACKENDS)
def test_cluster_cull_off_is_noop(backend: str) -> None:
    """With cluster_cull=False, cluster_visible stays all-ones (seeded at
    allocate) and bin_count treats every splat normally — a splat behind
    the camera still gets tile_count=0 via the near-plane cull, not the
    cluster gate. This test guards against accidental activation."""
    n = CLUSTER_SIZE
    pipeline = _setup(backend, max_splats=256, cluster_cull=False)

    means = np.zeros((n, 3), dtype=np.float32)
    means[:, 2] = 5.0
    means[:, 0] = np.linspace(-0.05, 0.05, n)
    log_scales = np.log(np.full((n, 3), 0.05, dtype=np.float32))
    rot_quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    opacities = np.full(n, 0.5, dtype=np.float32)
    color_logits = np.zeros((n, 3), dtype=np.float32)

    pipeline.set_primitive_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot_quats,
        opacities=opacities, color_logits=color_logits,
    )
    cam = simple_camera(width=128, height=128,
                        eye=(0.0, 0.0, 0.0), target=(0.0, 0.0, 1.0))
    pipeline.rasterize(cam)
    pipeline.device.wait_for_idle()

    # cluster_visible must remain all-ones (never dispatched cull kernel).
    cv = pipeline.buffers.raster.cluster_visible.to_numpy().view(np.uint32)
    assert np.all(cv == 1), (
        f"cluster_cull=False but cluster_visible changed: unique={np.unique(cv)}")
