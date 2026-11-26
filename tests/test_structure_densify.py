# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""End-to-end test of the structure-aware densification path (SAD-GS).

Two levels:
  1. Decision + anisotropic split: inject eta accumulators directly (bypassing
     the scoring kernel) so the multiview decision and grid-split child counts
     are checked against a hand-computed oracle.
  2. Scoring kernel: a large Gaussian viewed against a high-frequency (large
     lambda1) structure field must produce eta > 1 and vote to split; a tiny
     Gaussian against a low-frequency field must not.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config.config import DensifyConfig, StructureConfig
from trainer.data.dataset import simple_camera


def _setup(gpu_device, max_splats: int = 256, sh_degree: int = 0) -> TestPipeline:
    from trainer.config.config import AppearanceConfig

    return TestPipeline(device=gpu_device, width=64, height=64, max_splats=max_splats,
                        max_sort_pairs=max(1024, max_splats * 16),
                        appearance_config=AppearanceConfig(sh_degree=sh_degree))


def _count(p: TestPipeline) -> int:
    return int(p.buffers.primitive.active_count.to_numpy().view(np.uint32)[0])


def _count_actions(p: TestPipeline, pre_count: int) -> dict[str, int]:
    """Read the action buffer post-densify and count each action label.

    ``densify_structure_aware`` is wait-free now (returns None counts), so tests
    that check the decision's action distribution must read the buffer from
    the GPU themselves. ``pre_count`` is the count BEFORE the densify so we
    only look at the pre-densify slots (post-densify slots include appended
    children, whose action tag is stale KEEP)."""
    raw = p.buffers.densification.actions.to_numpy()
    a = np.frombuffer(raw.tobytes(), dtype=np.uint32)[:pre_count]
    return {
        "n_keep":  int((a == 0).sum()),
        "n_split": int((a == 1).sum()),
        "n_clone": int((a == 2).sum()),
        "n_prune": int((a == 3).sum()),
    }


def _set_eta(p: TestPipeline, idx: int, *, n_total: int,
             n_high: int = 0, n_low: int = 0, eta_max=(0.0, 0.0, 0.0),
             grad_high: bool = True) -> None:
    """Write eta accumulators for one gaussian (others stay zero). Matches the
    SADGS model: n_high/n_low are SCALAR view counts (max-over-axes eta), and
    the split gate also requires is_grad_high, so we inject the avg-grad signal
    (grad_mean_2d_mag_accum / visibility_count) above the grad_threshold."""
    b = p.buffers
    N = p.max_splats

    def w(buf, val, dtype=np.uint32):
        a = np.zeros(N, dtype=dtype)
        a[idx] = val
        buf.copy_from_numpy(a)

    w(b.eta.n_total, n_total)
    w(b.eta.components["n_high"], n_high)
    w(b.eta.components["n_low"], n_low)
    for ax, key in enumerate(("max_x", "max_y", "max_z")):
        w(b.eta.components[key], int(eta_max[ax] * 65536.0))

    # is_grad_high signal: avg_grad = (mag/65536)/vis, compared against the
    # convention-scaled floor grad_threshold*W*H*3 (~0.12 at 64x64). Set vis=1
    # and a mag comfortably above it (or at zero, below).
    from trainer.backend.memory.buffers import encode_grad_array
    vis = 1
    w(b.densification.visibility_count, vis)
    a = np.zeros(N, dtype=np.float64)
    a[idx] = 1.0 if grad_high else 0.0
    b.densification.grad_mean_2d_mag.copy_from_numpy(
        encode_grad_array(b.densification.grad_mean_2d_mag, a))


def _one_gaussian(p: TestPipeline, scale=0.1, opacity=0.9) -> None:
    p.set_primitive_from_numpy(
        means=np.array([[0.0, 0.0, 3.0]], dtype=np.float32),
        log_scales=np.log(np.full((1, 3), scale, dtype=np.float32)),
        rot_quats=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        opacities=np.array([opacity], dtype=np.float32),
        color_logits=np.zeros((1, 3), dtype=np.float32),
    )


def test_anisotropic_split_child_count(gpu_device) -> None:
    """A gaussian that violates on x (eta_max=4) and y (eta_max=2) only should
    grid-split into ceil(sqrt(4)) * ceil(sqrt(2)) * 1 = 2 * 2 * 1 = 4 children."""
    p = _setup(gpu_device)
    _one_gaussian(p)
    # high_ratio = 10/10 = 1.0 > tau_split, gradient high. Per-axis eta_max
    # drives the factor: x=4, y=2, z clean.
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0,
             eta_max=(4.0, 2.0, 0.05), grad_high=True)

    # dense_scale=0 so the (scale 0.1) gaussian is "large" -> SPLIT path.
    cfg = StructureConfig(tau_split=0.8, tau_prune=0.8, split_power=0.5,
                          max_split_factor=4, dense_scale=0.0)
    pre_count = _count(p)
    p.structure_densifier.densify(cfg)

    # n_x = ceil(sqrt(4)) = 2, n_y = ceil(sqrt(2)) = 2, n_z = 1 -> 4 children.
    actions = _count_actions(p, pre_count)
    assert actions["n_split"] == 1, f"expected 1 split, got {actions}"
    assert _count(p) == 4, f"expected 4 grid children, got {_count(p)}"


def test_small_gaussian_clones_not_splits(gpu_device) -> None:
    """A triggered SMALL gaussian (max scale <= dense_scale) should CLONE
    (exact duplicate, +1) rather than anisotropically split — SADGS routes the
    same eta trigger to clone-if-small / split-if-large."""
    p = _setup(gpu_device)
    _one_gaussian(p, scale=0.1)  # world-space max scale = 0.1
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0,
             eta_max=(4.0, 2.0, 0.05), grad_high=True)

    # dense_scale=0.2 > 0.1, so the gaussian is "small" -> CLONE path.
    cfg = StructureConfig(tau_split=0.8, tau_prune=0.8, split_power=0.5,
                          max_split_factor=4, dense_scale=0.2)
    pre_count = _count(p)
    p.structure_densifier.densify(cfg)

    actions = _count_actions(p, pre_count)
    assert actions["n_clone"] == 1, f"expected 1 clone, got {actions}"
    assert actions["n_split"] == 0
    assert _count(p) == 2, f"clone should append exactly 1 duplicate, got {_count(p)}"

    # The clone is an exact duplicate of the parent (same 14 semantic floats).
    g = _read_gaussians(p, 2)
    assert np.allclose(g[0], g[1]), "clone is not an exact duplicate of the parent"


def test_no_split_when_ratio_below_threshold(gpu_device) -> None:
    """high_ratio = 3/10 = 0.3 < tau_split=0.8 -> no split, count unchanged
    (even though eta_max=9 and gradient is high)."""
    p = _setup(gpu_device)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=3, n_low=0, eta_max=(9.0, 0.0, 0.0), grad_high=True)

    cfg = StructureConfig(tau_split=0.8)
    pre_count = _count(p)
    p.structure_densifier.densify(cfg)
    actions = _count_actions(p, pre_count)
    assert actions["n_split"] == 0, f"expected 0 splits, got {actions}"
    assert _count(p) == 1


def test_no_split_when_gradient_low(gpu_device) -> None:
    """high_ratio passes (10/10) but is_grad_high is False (well-placed large
    gaussian) -> no split. This is the SADGS gradient AND-gate."""
    p = _setup(gpu_device)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0, eta_max=(4.0, 0.0, 0.0), grad_high=False)

    cfg = StructureConfig(tau_split=0.8)
    pre_count = _count(p)
    p.structure_densifier.densify(cfg)
    actions = _count_actions(p, pre_count)
    assert actions["n_split"] == 0, f"expected 0 splits, got {actions}"
    assert _count(p) == 1


def test_prune_low_ratio(gpu_device) -> None:
    """low_ratio = 10/10 = 1.0 > tau_prune -> prune (over-resolved)."""
    p = _setup(gpu_device)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=0, n_low=10, eta_max=(0.05, 0.05, 0.05))

    cfg = StructureConfig(tau_prune=0.8)
    pre_count = _count(p)
    p.structure_densifier.densify(cfg)
    actions = _count_actions(p, pre_count)
    assert actions["n_prune"] == 1, f"expected 1 prune, got {actions}"
    assert _count(p) == 0


def test_split_factor_capped(gpu_device) -> None:
    """eta_max=100 with cap 3 -> n_k clamped to 3, not ceil(sqrt(100))=10."""
    p = _setup(gpu_device)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0,
             eta_max=(100.0, 0.0, 0.0), grad_high=True)

    cfg = StructureConfig(tau_split=0.8, split_power=0.5, max_split_factor=3)
    p.structure_densifier.densify(cfg)
    # x: min(ceil(sqrt(100))=10, cap 3) = 3; y,z = 1 -> 3 children.
    assert _count(p) == 3


@pytest.mark.parametrize("sh_degree", [0, 3])
def test_full_grid_parallel_children_and_fresh_moments(gpu_device, sh_degree) -> None:
    """512 children span several cooperative-lane iterations; all inherit SH."""
    from trainer.common.sh import unpack_features_rest

    p = _setup(gpu_device, max_splats=600, sh_degree=sh_degree)
    _one_gaussian(p, scale=0.3, opacity=0.7)
    _set_eta(p, 0, n_total=10, n_high=10, eta_max=(64, 64, 64))
    if sh_degree:
        rest = np.arange(p.buffers.sh.rest_floats, dtype=np.float32)[None, :] * 0.01
        p.primitives.write_features_rest(rest)
        p.buffers.sh.adam_rest.copy_from_numpy(
            np.full(p.max_splats * p.buffers.sh.rest_floats * 2, 0x3F800000, np.uint32))
    cfg = StructureConfig(ks_scale_power=1.2, opacity_cap=0, max_split_factor=8)
    stats = p.structure_densifier.densify(cfg)
    assert _count(p) == 512 and not stats["overbudget"]
    g = _read_gaussians(p, 512)
    expected_scale = np.log(0.3) - 1.2 * np.log(8)
    np.testing.assert_allclose(g[:, 3:6], expected_scale, atol=1e-6)
    assert np.unique(g[:, :3], axis=0).shape[0] == 512
    np.testing.assert_allclose(g[:, :3].mean(axis=0), [0, 0, 3], atol=1e-6)
    adam = p.buffers.adam.state.to_numpy().view(np.float32).reshape(p.max_splats, -1)
    assert np.all(adam[:512] == 0)
    if sh_degree:
        packed = p.buffers.sh.features_rest.to_numpy().view(np.uint32)
        features = unpack_features_rest(
            packed[:512 * p.buffers.sh.feat_uints], 512, p.buffers.sh.rest_floats)
        np.testing.assert_array_equal(features, np.repeat(features[:1], 512, axis=0))
        moments = p.buffers.sh.adam_rest.to_numpy().view(np.uint32).reshape(-1, p.max_splats)
        assert np.all(moments[:, :512] == 0)


@pytest.mark.parametrize("sh_degree", [0, 3])
@pytest.mark.parametrize("ranked", [False, True])
def test_overbudget_structure_split(
    gpu_device, sh_degree, ranked, monkeypatch, capsys,
) -> None:
    p = _setup(gpu_device, max_splats=7, sh_degree=sh_degree)
    _one_gaussian(p, opacity=0.7)
    _set_eta(p, 0, n_total=10, n_high=10, eta_max=(64, 64, 64))
    monkeypatch.setattr(np, "percentile", lambda *a, **k: pytest.fail(
        "production overflow must not run CPU percentile diagnostics"))
    stats = p.structure_densifier.densify(StructureConfig(rank_split_families=ranked))
    assert stats["overbudget"]
    if ranked:
        assert _count(p) == 4  # A complete 2x2x1 grid fits. A 2x2x2 grid needs eight slots.
        positions = _read_gaussians(p, 4)[:, :3]
        np.testing.assert_allclose(positions.mean(axis=0), [0, 0, 3], atol=1e-6)
        assert tuple(len(np.unique(axis)) for axis in positions.T) == (2, 2, 1)
    else:
        assert _count(p) == 7
        assert np.all(np.isfinite(_read_gaussians(p, 7)))
        assert "overflow children DROPPED" in capsys.readouterr().out


@pytest.mark.parametrize("sh_degree", [0, 3])
@pytest.mark.parametrize("mode", ["structure", "simple"])
def test_full_budget_preserves_unsplittable_parent(gpu_device, sh_degree, mode) -> None:
    p = _setup(gpu_device, max_splats=1, sh_degree=sh_degree)
    _one_gaussian(p, scale=0.3, opacity=0.7)
    _set_eta(p, 0, n_total=10, n_high=10, eta_max=(64, 64, 64))
    before = _read_gaussians(p, 1).copy()
    if mode == "structure":
        stats = p.structure_densifier.densify(StructureConfig(dense_scale=0, opacity_cap=0))
        assert stats["overbudget"]
    else:
        p.simple_densifier.run(DensifyConfig(), grad_thresh=0, scale_thresh=0,
                               allow_scale_prune=False, carry_eta=True)
        assert _read_eta(p, 1)["n_total"][0] == 10
    assert _count(p) == 1
    np.testing.assert_array_equal(_read_gaussians(p, 1), before)


def test_clones_count_toward_requested_tail_growth(gpu_device, capsys) -> None:
    p = _setup(gpu_device, max_splats=1)
    _one_gaussian(p)
    _set_eta(p, 0, n_total=10, n_high=10, eta_max=(4, 2, 0.05))
    stats = p.structure_densifier.densify(StructureConfig(dense_scale=0.2))
    assert _count(p) == 1 and stats["overbudget"]
    assert int(p.buffers.structure.children_sum.to_numpy()[0]) == 1
    assert "requested 1 new children" in capsys.readouterr().out


def test_candidate_compaction_partial_wave_and_reuse(gpu_device) -> None:
    """Repeated ticks consume a 257-parent list, including a partial final wave."""
    n = 257
    p = _setup(gpu_device, max_splats=600)
    p.set_primitive_from_numpy(
        means=np.column_stack([np.arange(n) * 0.01, np.zeros(n), np.full(n, 3)]).astype(np.float32),
        log_scales=np.log(np.full((n, 3), 0.1, np.float32)),
        rot_quats=np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        opacities=np.full(n, 0.7, np.float32), color_logits=np.zeros((n, 3), np.float32))
    b = p.buffers
    total = np.zeros(p.max_splats, np.uint32)
    total[:n] = 10
    b.eta.n_total.copy_from_numpy(total)
    b.eta.components["n_high"].copy_from_numpy(total)
    b.eta.components["n_low"].copy_from_numpy(np.zeros_like(total))
    for axis in ("max_x", "max_y", "max_z"):
        b.eta.components[axis].copy_from_numpy(total * 65536)
    from trainer.backend.memory.buffers import encode_grad_array
    b.densification.visibility_count.copy_from_numpy((total > 0).astype(np.uint32))
    b.densification.grad_mean_2d_mag.copy_from_numpy(
        encode_grad_array(b.densification.grad_mean_2d_mag, (total > 0).astype(np.float64)))
    p.structure_densifier.densify(StructureConfig(dense_scale=0.2))
    assert _count(p) == 514
    assert int(b.structure.children_sum.to_numpy()[0]) == n
    p.structure_densifier.clear_scores()
    p.structure_densifier.densify(StructureConfig(dense_scale=0.2))
    assert _count(p) == 514
    assert int(b.structure.children_sum.to_numpy()[0]) == 0


def test_full_controller_tick_includes_mip_timing_and_queue(gpu_device) -> None:
    """Exercise scratch reuse -> Morton -> Mip on a real production tick."""
    from types import SimpleNamespace

    from trainer.config.config import IterationEvents
    from trainer.config.config_types import DensificationMode
    from trainer.frontend.densification import DensificationController
    from trainer.frontend.profiling.mutation_timing import MutationTimingRecorder
    from trainer.frontend.queue_limiter import QueueLimiter
    from trainer.frontend.schedule import ExtinctionHardnessSchedule

    p = _setup(gpu_device, max_splats=600, sh_degree=3)
    _one_gaussian(p, scale=0.3, opacity=0.7)
    _set_eta(p, 0, n_total=10, n_high=10, eta_max=(64, 64, 64))
    camera = simple_camera(width=64, height=64, eye=(0, 0, 0), target=(0, 0, 3))
    p.mip_filter.set_cameras([camera] * 3)
    queue = QueueLimiter(p.runtime, max_depth=5)
    controller = DensificationController(
        pipeline=p, args=SimpleNamespace(reset_every=1200),
        densify_mode=DensificationMode.STRUCTURE_AWARE, densify_cfg=DensifyConfig(),
        structure_cfg=StructureConfig(opacity_cap=0, max_split_factor=8), extent=1,
        image_width=64, image_height=64,
        morton_world_min=np.array([-2, -2, 0], np.float32),
        morton_world_max=np.array([2, 2, 6], np.float32),
        extinction_hardness_schedule=ExtinctionHardnessSchedule(1, 0), queue_limiter=queue)
    events = IterationEvents(
        iteration=500, densify=True, structure=True, opacity_reset=False,
        preview=False, render_all=False, log=False, eval=False, psnr_curve=False,
        morton=False, mip_after_densify=True, mip_periodic=False)
    timing = MutationTimingRecorder(enabled=True)
    with timing.measure(500, p, "structure"):
        outcome = controller.run(500, events, 1)
    assert outcome.kind == "structure" and _count(p) == 512
    assert queue.depth == 3  # densification, Morton, and the expensive Mip submit.
    filters = p.buffers.primitive.filter_3d.to_numpy()
    assert np.all(filters[:512] > 0) and np.all(filters[512:] == 0)
    markers = timing.samples[0]["markers_ms"]
    assert {"densify_structure", "densify_tracking", "morton_sort", "mip_filter"} <= markers.keys()
    assert p.commands.timer is None


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Replicate shaders/common/math.slang::matrixFromQuaternion EXACTLY
    (q = (w, x, y, z), normalized, row-major). Must match bit-for-bit in
    convention so the Python oracle and the GPU split agree on R."""
    q = q / np.linalg.norm(q)
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),       2.0 * (x * z + w * y)],
        [2.0 * (x * y + w * z),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
        [2.0 * (x * z - w * y),       2.0 * (y * z + w * x),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def test_anisotropic_split_preserves_covariance(gpu_device) -> None:
    """SADGS split geometry invariant: for ks_scale_power == 1, the equal-weight
    mixture of the grid children must reproduce the PARENT's 3D covariance
    exactly (SADGS gaussian_model.py Eq. 8-9; the sqrt(12) child spacing is
    precisely the factor that makes this hold).

    This is the regression guard for the sqrt(12) fix: the previous code spaced
    children at coord/n with NO sqrt(12), which contracts the mixture to ~31% of
    the parent variance at n=2 (and worse for larger n) — that version fails
    this test dramatically; the sqrt(12) version passes to fp32 precision.

    Uses a non-trivial rotation, anisotropic scale, and UNEQUAL per-axis split
    factors n = (3, 2, 1) so the test exercises multi-factor grids and the
    n_k == 1 no-op axis, not just the isotropic k=2 case."""
    p = _setup(gpu_device)

    # Anisotropic parent scale + a real (non-identity) rotation.
    scale = np.array([0.30, 0.20, 0.12], dtype=np.float64)
    quat = np.array([0.9, 0.1, 0.2, 0.3], dtype=np.float32)   # (w, x, y, z), unnormalized
    p.set_primitive_from_numpy(
        means=np.array([[0.1, -0.2, 3.0]], dtype=np.float32),
        log_scales=np.log(scale.astype(np.float32))[None, :],
        rot_quats=quat[None, :],
        opacities=np.array([0.9], dtype=np.float32),
        color_logits=np.zeros((1, 3), dtype=np.float32),
    )

    # eta_max=(9,4,0.05), split_power=0.5 -> n=(ceil(sqrt 9), ceil(sqrt 4),
    # ceil(sqrt 0.05)) = (3, 2, 1) -> 6 children. high_ratio=1.0, grad high.
    _set_eta(p, 0, n_total=10, n_high=10, n_low=0,
             eta_max=(9.0, 4.0, 0.05), grad_high=True)

    # dense_scale=0 so the (large) gaussian takes the SPLIT path; ks_scale_power
    # = 1.0 is the variance-preserving choice paired with the sqrt(12) spacing.
    cfg = StructureConfig(tau_split=0.8, tau_prune=0.8, split_power=0.5,
                          max_split_factor=8, dense_scale=0.0, ks_scale_power=1.0)
    p.structure_densifier.densify(cfg)

    m = _count(p)
    assert m == 6, f"expected 3*2*1 = 6 grid children, got {m}"

    # Read back the 6 children: 14 semantic floats each (mean[0:3], log_scale[3:6],
    # rot_quat[6:10], opacity[10], sh0[11:14]).
    g = _read_gaussians(p, m).astype(np.float64)
    child_means = g[:, 0:3]
    child_scales = np.exp(g[:, 3:6])

    R = _quat_to_matrix(quat)
    # Parent covariance Sigma = R diag(scale^2) R^T.
    parent_cov = R @ np.diag(scale ** 2) @ R.T

    # Equal-weight Gaussian-mixture covariance:
    #   Sigma_mix = mean_i(Sigma_i) + mean_i((mu_i - mu_bar)(mu_i - mu_bar)^T)
    # where Sigma_i = R diag(child_scale_i^2) R^T (children share the parent R).
    within = np.zeros((3, 3))
    for s_i in child_scales:
        within += R @ np.diag(s_i ** 2) @ R.T
    within /= m

    mu_bar = child_means.mean(axis=0)
    d = child_means - mu_bar
    between = (d[:, :, None] * d[:, None, :]).mean(axis=0)

    mixture_cov = within + between

    # fp32 GPU math on cov entries ~1e-2..1e-1: atol 2e-4 is comfortable.
    assert np.allclose(mixture_cov, parent_cov, atol=2e-4, rtol=1e-3), (
        f"grid-child mixture covariance does not preserve the parent's.\n"
        f"parent=\n{parent_cov}\nmixture=\n{mixture_cov}\n"
        f"diag ratio (mixture/parent) = "
        f"{np.diag(mixture_cov) / np.diag(parent_cov)}")

    # The mixture mean must also match the parent mean (grid is centered).
    assert np.allclose(mu_bar, np.array([0.1, -0.2, 3.0]), atol=2e-4), (
        f"grid is not centered on the parent: mean-of-means = {mu_bar}")


def test_scoring_kernel_flags_large_gaussian(gpu_device) -> None:
    """The scoring kernel itself: a LARGE gaussian against a high-frequency
    structure field (large lambda1) yields eta > 1 (splits), while a TINY
    gaussian against the same field does not."""
    import slangpy as spy

    cam = simple_camera(width=64, height=64, eye=(0.0, 0.0, 0.0),
                        target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                        fovy_degrees=60.0)
    cfg = StructureConfig(n_scoring_views=1, tau_split=0.5)

    # Structure field with lambda1 = 0.25 everywhere -> wavelength ~ 2px.
    # Chosen so the LARGE gaussian (extent ~tens of px) has eta >> 1, while the
    # TINY gaussian stays below eta=1 EVEN after the renderer's dilation floors
    # its footprint to ~sqrt(0.3)=0.55px (0.55 * sqrt(0.25) = 0.27 < 1).
    # Packed (Sxx, Sxy, Syy, 0); lambda1 = Sxx here (Sxy=Syy=0).
    lam = 0.25
    field = np.zeros((64, 64, 4), dtype=np.float32)
    field[..., 0] = lam

    def _make_tex(p):
        tex = p.device.create_texture(
            type=spy.TextureType.texture_2d, format=spy.Format.rgba32_float,
            width=64, height=64,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label="sbar")
        tex.copy_from_numpy(np.ascontiguousarray(field))
        return tex

    def _score(p):
        p.structure_densifier.clear_scores()
        p.rasterize(cam)
        p.structure_densifier.accumulate_scores(
            _make_tex(p), p.rasterizer.camera_buffer, pass_idx=0)

    def _set_grad_high(p):
        # The split gate ANDs in is_grad_high; inject an above-threshold avg
        # grad so the test isolates the eta/frequency behaviour.
        from trainer.backend.memory.buffers import encode_grad_array
        N = p.max_splats
        p.buffers.densification.visibility_count.copy_from_numpy(np.ones(N, dtype=np.uint32))
        mag = encode_grad_array(p.buffers.densification.grad_mean_2d_mag, np.ones(N, dtype=np.float64))
        p.buffers.densification.grad_mean_2d_mag.copy_from_numpy(mag)

    # Large gaussian -> big projected extent -> eta > 1 -> should split.
    p_big = _setup(gpu_device)
    _one_gaussian(p_big, scale=0.5)
    _score(p_big)
    _set_grad_high(p_big)
    pre_big = _count(p_big)
    p_big.structure_densifier.densify(cfg)
    actions_big = _count_actions(p_big, pre_big)

    # Tiny gaussian -> small extent -> eta < 1 -> no split.
    p_small = _setup(gpu_device)
    _one_gaussian(p_small, scale=0.001)
    _score(p_small)
    _set_grad_high(p_small)
    pre_small = _count(p_small)
    p_small.structure_densifier.densify(cfg)
    actions_small = _count_actions(p_small, pre_small)

    assert actions_big["n_split"] >= 1, "large gaussian should violate frequency and split"
    assert actions_small["n_split"] == 0, "tiny gaussian should not split"


def _read_gaussians(p: TestPipeline, n: int) -> np.ndarray:
    """Read back the first n gaussians as an (n, 14) float array (mean[0:3],
    log_scale[3:6], rot_quat[6:10], opacity[10], sh0[11:14])."""
    rec = np.frombuffer(p.buffers.primitive.values.to_numpy().tobytes(),
                        dtype=p.buffers.primitive.layout.dtype)[:n]
    return np.column_stack([
        rec["mean"], rec["log_scale"], rec["rot_quat"],
        rec["opacity"][:, None], rec["sh0"],
    ])


def _read_eta(p: TestPipeline, n: int) -> dict[str, np.ndarray]:
    """Read back the first n slots of every eta accumulator."""
    def r(buf):
        return np.frombuffer(buf.to_numpy().tobytes(), dtype=np.uint32)[:n]
    return {
        "n_total": r(p.buffers.eta.n_total),
        "n_high":  r(p.buffers.eta.components["n_high"]),
        "n_low":   r(p.buffers.eta.components["n_low"]),
        "max_x":   r(p.buffers.eta.components["max_x"]),
        "max_y":   r(p.buffers.eta.components["max_y"]),
        "max_z":   r(p.buffers.eta.components["max_z"]),
    }


def test_eta_carried_through_filler_densify(gpu_device) -> None:
    """SAD-GS eta persistence (#2): a FILLER (AbsGS) densify with carry_eta=True
    must permute the slot-indexed eta accumulators through the same
    valid_mask/scan as the gaussians, carrying the multiview window of surviving
    gaussians and zeroing it for brand-new primitives.

    Carry rule under test (matches SADGS):
      - KEEP survivor        -> eta carried
      - CLONE parent (kept)  -> eta carried; the appended clone -> zero
      - SPLIT parent slot    -> overwritten with a NEW child-1 -> eta zeroed;
                                the appended child-2 -> zero
      - PRUNE                -> dropped

    We drive the four actions deterministically via decide_actions_simple's
    gates (opacity<t_opacity -> PRUNE; grad>thresh & scale>scale_thresh ->
    SPLIT; grad>thresh & small -> CLONE; grad<=thresh -> KEEP), tag each parent
    with a UNIQUE eta signature, and after the densify match each surviving
    gaussian back to its parent by POSITION (the gaussian scatter is co-indexed
    with the eta scatter, so a correct implementation keeps them aligned)."""
    p = _setup(gpu_device, max_splats=64)

    # 4 parents at distinct x-positions. Scale: 3 small (0.01) so KEEP/CLONE/
    # PRUNE stay in the clone-size band; 1 large (0.30) so the SPLIT candidate
    # takes the split path. Opacity: parent 3 near-zero so it PRUNES.
    #   idx 0: KEEP   (grad low)                          -> carry
    #   idx 1: CLONE  (grad high, small scale)            -> carry (+appended 0)
    #   idx 2: SPLIT  (grad high, large scale)            -> zero  (+appended 0)
    #   idx 3: PRUNE  (opacity below floor)               -> dropped
    xs = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    scales = np.array([0.01, 0.01, 0.30, 0.01], dtype=np.float32)
    opac = np.array([0.9, 0.9, 0.9, 1e-4], dtype=np.float32)
    means = np.stack([xs, np.zeros(4, np.float32), np.full(4, 5.0, np.float32)], axis=1)
    p.set_primitive_from_numpy(
        means=means,
        log_scales=np.log(np.repeat(scales[:, None], 3, axis=1)),
        rot_quats=np.tile(np.array([1.0, 0, 0, 0], np.float32), (4, 1)),
        opacities=opac,
        color_logits=np.zeros((4, 3), dtype=np.float32),
    )
    p.buffers.primitive.active_count.copy_from_numpy(np.array([4], dtype=np.uint32))

    # Unique eta signature per parent so we can identify who landed where.
    # (values are arbitrary but distinct; max_* stored as fixed-point uint.)
    N = p.max_splats
    def wbuf(buf, vals, dtype=np.uint32):
        a = np.zeros(N, dtype=dtype)
        a[:4] = vals
        buf.copy_from_numpy(a)
    wbuf(p.buffers.eta.n_total, [11, 22, 33, 44])
    wbuf(p.buffers.eta.components["n_high"], [1, 2, 3, 4])
    wbuf(p.buffers.eta.components["n_low"], [5, 6, 7, 8])
    wbuf(p.buffers.eta.components["max_x"], [100, 200, 300, 400])
    wbuf(p.buffers.eta.components["max_y"], [110, 210, 310, 410])
    wbuf(p.buffers.eta.components["max_z"], [120, 220, 320, 420])

    # Grad/visibility signal driving the KEEP/CLONE/SPLIT decision. avg_grad =
    # (mag/65536)/vis; the internal floor is grad_thresh * W*H*3 = 0.0002 *
    # 64*64*3 = ~2.46. Give idx 1,2 avg_grad = 5.0 (>> floor) and idx 0 zero
    # grad (-> KEEP). vis=1 for all live slots.
    from trainer.backend.memory.buffers import encode_grad_array
    vis = np.zeros(N, dtype=np.uint32); vis[:4] = 1
    p.buffers.densification.visibility_count.copy_from_numpy(vis)
    mag = np.zeros(N, dtype=np.float64)
    mag[1] = mag[2] = 5.0                   # avg_grad = 5.0 >> ~2.46 floor
    mag[0] = mag[3] = 0.0                   # KEEP / (PRUNE decided by opacity)
    p.buffers.densification.grad_mean_2d_mag.copy_from_numpy(
        encode_grad_array(p.buffers.densification.grad_mean_2d_mag, mag))

    cfg = DensifyConfig(t_opacity=0.005, t_big_scale=1e9)
    # scale_thresh between the small (0.01) and large (0.30) scales so idx 2
    # splits and idx 1 clones. grad_thresh at gsplat's 0.0002. Disable big-scale
    # prune (t_big_scale huge + allow_scale_prune False) so only opacity prunes.
    p.simple_densifier.run(cfg, grad_thresh=0.0002, scale_thresh=0.1,
                     prune_big_scale=1e9, allow_scale_prune=False,
                     carry_eta=True)

    n = _count(p)
    # 4 parents - 1 prune (idx3) + 1 clone-append (idx1) + 1 split-append (idx2)
    # = 5 live gaussians.
    assert n == 5, f"expected 5 gaussians after densify, got {n}"

    g = _read_gaussians(p, n)
    eta = _read_eta(p, n)
    xpos = g[:, 0]

    # Locate the KEEP survivor (x=-3) and the CLONE parent (x=-1) by position.
    keep_slots = np.where(np.isclose(xpos, -3.0, atol=1e-3))[0]
    clone_parent_slots = np.where(np.isclose(xpos, -1.0, atol=1e-3))[0]
    assert len(keep_slots) == 1, f"KEEP survivor not found uniquely: x={xpos}"
    # CLONE produces two gaussians at x=-1 (parent + exact duplicate).
    assert len(clone_parent_slots) == 2, f"expected 2 gaussians at clone x=-1: x={xpos}"

    ks = int(keep_slots[0])
    # KEEP survivor keeps its full eta signature (parent 0).
    assert eta["n_total"][ks] == 11, f"KEEP eta_n_total not carried: {eta['n_total'][ks]}"
    assert eta["n_high"][ks] == 1 and eta["n_low"][ks] == 5
    assert eta["max_x"][ks] == 100 and eta["max_y"][ks] == 110 and eta["max_z"][ks] == 120

    # Exactly ONE of the two x=-1 gaussians carries parent-1's eta (the surviving
    # parent slot); the other (the appended clone) is zeroed.
    c_ntotal = eta["n_total"][clone_parent_slots]
    assert sorted(c_ntotal.tolist()) == [0, 22], (
        f"clone eta: exactly one slot should carry n_total=22, other 0; got {c_ntotal}")
    carrier = clone_parent_slots[np.argmax(c_ntotal)]
    assert eta["n_high"][carrier] == 2 and eta["max_x"][carrier] == 200

    # The SPLIT children must ALL be zeroed — a split replaces the parent with
    # brand-new children. split_and_clone jitters child positions within the
    # parent covariance (parent scale 0.30), so they land NEAR x=+1 but not
    # exactly; identify them as the slots that are neither the KEEP survivor
    # nor either clone-x=-1 gaussian (and there must be exactly 2).
    identified = set(keep_slots.tolist()) | set(clone_parent_slots.tolist())
    split_slots = [s for s in range(n) if s not in identified]
    assert len(split_slots) == 2, f"expected 2 split children, got {split_slots}: x={xpos}"
    # Sanity: they really are the split children (positioned near the split
    # parent's x=+1, within a few sigma of the 0.30-scale jitter).
    for s in split_slots:
        assert abs(xpos[s] - 1.0) < 2.0, f"unexpected split-child position x={xpos[s]}"
        assert eta["n_total"][s] == 0, f"SPLIT child eta not zeroed: slot {s} = {eta['n_total'][s]}"
        assert eta["max_x"][s] == 0 and eta["n_high"][s] == 0

    # Parent 3 (x=+3) was PRUNED — it must not appear at all (no gaussian near x=+3).
    assert not np.any(np.abs(xpos - 3.0) < 0.5), "pruned gaussian survived"

    # Global sanity: total carried n_total across all slots = 11 (keep) + 22
    # (clone parent) = 33; everything else zero. This catches any stray carry.
    assert int(eta["n_total"].sum()) == 33, (
        f"total carried eta_n_total should be 11+22=33, got {int(eta['n_total'].sum())}")


def test_eta_carried_through_morton_sort(gpu_device) -> None:
    """mortonScatterEta: a Morton re-sort with carry_eta=True permutes the eta
    accumulators by the SAME sorted_values as the gaussians (pure permutation,
    no zeroing). After the sort, each gaussian must still be paired with its
    original eta signature — verified by matching on position."""
    p = _setup(gpu_device, max_splats=64)

    # Place 5 gaussians deliberately OUT of Morton order (descending x) so the
    # sort actually permutes them; tag each with a position-keyed eta.
    k = 5
    xs = np.array([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32)
    means = np.stack([xs, np.zeros(k, np.float32), np.zeros(k, np.float32)], axis=1)
    p.set_primitive_from_numpy(
        means=means,
        log_scales=np.log(np.full((k, 3), 0.05, dtype=np.float32)),
        rot_quats=np.tile(np.array([1.0, 0, 0, 0], np.float32), (k, 1)),
        opacities=np.full(k, 0.9, dtype=np.float32),
        color_logits=np.zeros((k, 3), dtype=np.float32),
    )
    p.buffers.primitive.active_count.copy_from_numpy(np.array([k], dtype=np.uint32))

    # eta signature = round(10 * (x + 1)) so it's uniquely tied to position.
    N = p.max_splats
    sig = np.zeros(N, dtype=np.uint32)
    sig[:k] = (10 * (xs + 1)).astype(np.uint32)   # [50,40,30,20,10]
    p.buffers.eta.n_total.copy_from_numpy(sig)
    for key in ("n_high", "n_low", "max_x", "max_y", "max_z"):
        p.buffers.eta.components[key].copy_from_numpy(sig)

    p.morton.reorder(np.array([-1, -1, -1], np.float32),
                  np.array([5, 5, 5], np.float32), carry_eta=True)

    g = _read_gaussians(p, k)
    eta = _read_eta(p, k)
    # For every surviving gaussian, its eta must equal its position signature.
    for i in range(k):
        expected = int(round(10 * (g[i, 0] + 1)))
        assert eta["n_total"][i] == expected, (
            f"slot {i} (x={g[i,0]:.2f}) expected eta {expected}, got {eta['n_total'][i]}")
        for key in ("n_high", "n_low", "max_x", "max_y", "max_z"):
            assert eta[key][i] == expected, f"slot {i} {key} misaligned after sort"
