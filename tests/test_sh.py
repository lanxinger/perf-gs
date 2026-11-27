# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU tests for the optional view-dependent spherical-harmonics color path.

The default (SH-off) trainer never dispatches these kernels; enabling SH via
``AppearanceConfig(sh_degree>0)`` allocates the separate features_rest buffer
and turns on the per-gaussian SH forward/backward. These tests exercise the
forward (view dependence + the zero-rest invariant) and, once wired, the
gradient path. GPU-backed: skip if no backend is available.
"""

from __future__ import annotations

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.config.config import AdamLRs, AppearanceConfig, LossConfig
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device
from trainer.common.sh import pack_features_rest, unpack_features_rest

BACKENDS = ["vulkan", "d3d12", "metal"]


def _read_features_rest(p: TestPipeline, n: int) -> np.ndarray:
    """Read + unpack the fp16-packed features_rest for the first n gaussians."""
    sh = p.buffers.sh
    packed = np.frombuffer(p.buffers.sh.features_rest.to_numpy().tobytes(),
                           dtype=np.uint32)[: n * sh.feat_uints]
    return unpack_features_rest(packed, n, sh.rest_floats)


def _read_features_grad(p: TestPipeline, n: int) -> np.ndarray:
    sh = p.buffers.sh
    packed = np.frombuffer(p.buffers.sh.features_rest_grad.to_numpy().tobytes(),
                           dtype=np.uint32)[: n * sh.feat_uints]
    return unpack_features_rest(packed, n, sh.rest_floats)


def _pack_adam_mv(m: np.ndarray, v: np.ndarray, max_splats: int) -> np.ndarray:
    """(N, rest_floats) m/v float -> flat uint32 in adam_rest's SoA float32 layout:
    word k of gaussian g lives at k*max_splats + g, k in [0, 2*rest_floats)
    (m at 2s, v at 2s+1). Returns the full max_splats * 2*rest_floats flat array."""
    N, rf = m.shape
    flat = np.zeros(max_splats * rf * 2, dtype=np.uint32)
    m32 = m.astype(np.float32).view(np.uint32)
    v32 = v.astype(np.float32).view(np.uint32)
    idx = np.arange(N)
    for s in range(rf):
        flat[(2 * s) * max_splats + idx] = m32[:, s]
        flat[(2 * s + 1) * max_splats + idx] = v32[:, s]
    return flat


def _read_adam_m(p: TestPipeline, n: int) -> np.ndarray:
    rf = p.buffers.sh.rest_floats
    ms = p.max_splats
    raw = np.frombuffer(p.buffers.sh.adam_rest.to_numpy().tobytes(), dtype=np.float32)
    out = np.empty((n, rf), dtype=np.float32)
    idx = np.arange(n)
    for s in range(rf):
        out[:, s] = raw[(2 * s) * ms + idx]     # m word (SoA)
    return out


def _make_pipeline(backend: str, sh_degree: int, width: int = 48,
                   height: int = 48) -> TestPipeline:
    try:
        device = create_device(backend=backend)
    except Exception as exc:  # backend unavailable on host
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(
        device=device, width=width, height=height, max_splats=4,
        loss_config=LossConfig(ssim_weight=0.0, alpha_weight=0.0),
        appearance_config=AppearanceConfig(sh_degree=sh_degree),
    )


def _one_gaussian(p: TestPipeline, sh0_rgb: np.ndarray) -> None:
    means = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    log_scales = np.log(np.full((1, 3), 0.6, dtype=np.float32))
    rot_quats = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    opacities = np.array([0.99], dtype=np.float32)
    p.set_primitive_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot_quats,
        opacities=opacities, color_logits=sh0_rgb.reshape(1, 3).astype(np.float32))


def _center_rgb(p: TestPipeline, camera) -> np.ndarray:
    p.render(camera)
    img = p.read_output_image()
    h, w = img.shape[:2]
    return img[h // 2, w // 2, :3].astype(np.float64)


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_forward_is_view_dependent(backend: str) -> None:
    """A non-zero band-1 coefficient makes the rendered color depend on the
    camera direction; two orbit views of the same gaussian must differ."""
    p = _make_pipeline(backend, sh_degree=1)
    _one_gaussian(p, np.zeros(3, np.float32))  # sh0 = 0 -> base color 0.5

    # Band-1 z-term on the R channel: rest coeff index 1, channel 0.
    rest = np.zeros(p.buffers.sh.rest_floats, np.float32)
    rest[1 * 3 + 0] = 5.0
    p.primitives.write_features_rest(rest.reshape(1, -1))
    p.set_active_sh_degree(1)

    # Both cameras look at the gaussian (centered), from +z-behind vs +x-side,
    # so the view direction (mean - camera) differs: (0,0,1) vs (-1,0,0).
    cam_front = simple_camera(48, 48, eye=(0.0, 0.0, -4.0), target=(0.0, 0.0, 0.0))
    cam_side = simple_camera(48, 48, eye=(4.0, 0.0, 0.0), target=(0.0, 0.0, 0.0))

    front = _center_rgb(p, cam_front)
    side = _center_rgb(p, cam_side)

    # Front view sees the +z lobe (bright R); side view sees z≈0 (base). The R
    # channel must differ clearly.
    assert front[0] - side[0] > 0.3, f"expected view dependence, got {front} vs {side}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_zero_rest_matches_sh0(backend: str) -> None:
    """With all rest coefficients zero, the SH path reduces to shDC(sh0): the
    render must match the SH0 (view-independent) baseline for positive colors."""
    cam = simple_camera(48, 48, eye=(0.0, 0.0, -4.0), target=(0.0, 0.0, 0.0))
    sh0 = np.array([0.3, -0.2, 0.6], np.float32)

    sh_off = _make_pipeline(backend, sh_degree=0)
    _one_gaussian(sh_off, sh0)
    base = _center_rgb(sh_off, cam)

    sh_on = _make_pipeline(backend, sh_degree=3)
    _one_gaussian(sh_on, sh0)
    # features_rest is already zero (set_gaussians clears it).
    sh_on.set_active_sh_degree(3)
    got = _center_rgb(sh_on, cam)

    np.testing.assert_allclose(got, base, rtol=1e-3, atol=1e-3)


def _upload_rest(p: TestPipeline, gid0_rest: np.ndarray) -> None:
    """Upload gaussian 0's rest coefficients (rest is zero for other slots)."""
    p.primitives.write_features_rest(gid0_rest.reshape(1, -1))


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_rest_gradient_finite_difference(backend: str) -> None:
    """Analytic features_rest gradient (autodiff sh_backward) matches a central
    finite difference of the summed L1 loss, across all three SH bands."""
    W = H = 24
    p = _make_pipeline(backend, sh_degree=3, width=W, height=H)

    # Off-axis view so every SH basis function is non-zero (dir has x,y,z != 0),
    # and a bright DC base + small coeffs keep every channel positive (no clamp
    # kinks under the finite-difference perturbation).
    cam = simple_camera(W, H, eye=(-1.0, -1.0, 0.0), target=(0.0, 0.0, 3.0),
                        fovy_degrees=60.0)
    sh0 = np.full(3, 0.71, np.float32)          # base color ~0.7

    def _place() -> None:
        means = np.array([[0.0, 0.0, 3.0]], np.float32)
        log_scales = np.log(np.full((1, 3), 0.5, np.float32))
        rot = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
        opac = np.array([0.99], np.float32)
        p.set_primitive_from_numpy(means=means, log_scales=log_scales,
                                   rot_quats=rot, opacities=opac,
                                   color_logits=sh0.reshape(1, 3))

    rng = np.random.default_rng(0)
    rest0 = (rng.standard_normal(p.buffers.sh.rest_floats) * 0.05).astype(np.float32)

    target = np.zeros((H, W, 3), np.float32)
    target[..., 0] = 0.4
    target[..., 1] = 0.6
    target[..., 2] = 0.5
    p.targets.set_image(target)

    def loss_with(rest_vec: np.ndarray) -> float:
        _place()
        _upload_rest(p, rest_vec)
        p.set_active_sh_degree(3)
        p.forward_and_loss(cam)
        return p.read_loss()

    # Analytic gradient from the backward pass.
    _place()
    _upload_rest(p, rest0)
    p.set_active_sh_degree(3)
    p.forward_and_loss(cam)
    p.optimizer.zero_grad()
    p.run_backward()
    analytic = _read_features_grad(p, 1)[0].astype(np.float64)

    eps = 2e-3
    # Flat indices spanning band 1 (0..8), band 2 (9..23), band 3 (24..44),
    # across R/G/B channels.
    probe = [0, 4, 8, 9, 16, 23, 24, 33, 44]
    for c in probe:
        rp = rest0.copy(); rp[c] += eps
        rm = rest0.copy(); rm[c] -= eps
        numerical = (loss_with(rp) - loss_with(rm)) / (2 * eps)
        a = float(analytic[c])
        assert abs(numerical - a) <= 0.05 * abs(numerical) + 3e-2, (
            f"coeff {c}: analytic {a:.5f} vs numerical {numerical:.5f}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_mean_gradient_includes_view_term(backend: str) -> None:
    """With SH on, the mean gradient is the geometry term (projection_backward)
    PLUS the view-direction term (sh_backward). A central finite difference of
    the loss w.r.t. the mean must match the combined analytic grad_mean."""
    W = H = 24
    p = _make_pipeline(backend, sh_degree=3, width=W, height=H)
    cam = simple_camera(W, H, eye=(-1.0, -1.0, 0.0), target=(0.0, 0.0, 3.0),
                        fovy_degrees=60.0)
    sh0 = np.full(3, 1.42, np.float32)          # bright DC so channels stay > 0
    # Strong-ish coeffs so the view-dependent (dir->mean) term is non-trivial.
    rng = np.random.default_rng(1)
    rest0 = (rng.standard_normal(p.buffers.sh.rest_floats) * 0.2).astype(np.float32)

    mean0 = np.array([[0.0, 0.0, 3.0]], np.float32)
    log_scales = np.log(np.full((1, 3), 0.5, np.float32))
    rot = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    opac = np.array([0.99], np.float32)

    target = np.zeros((H, W, 3), np.float32)
    target[..., 0] = 0.4; target[..., 1] = 0.6; target[..., 2] = 0.5
    p.targets.set_image(target)

    def place(mean: np.ndarray) -> None:
        p.set_primitive_from_numpy(means=mean.astype(np.float32),
                                   log_scales=log_scales, rot_quats=rot,
                                   opacities=opac, color_logits=sh0.reshape(1, 3))
        _upload_rest(p, rest0)
        p.set_active_sh_degree(3)

    place(mean0)
    p.forward_and_loss(cam)
    p.optimizer.zero_grad()
    p.run_backward()
    grads = p.read_gradients()
    analytic = np.array([grads["mean_x"][0], grads["mean_y"][0], grads["mean_z"][0]])

    eps = 1e-3
    for axis in range(3):
        mp = mean0.copy(); mp[0, axis] += eps
        mm = mean0.copy(); mm[0, axis] -= eps
        place(mp); p.forward_and_loss(cam); lp = p.read_loss()
        place(mm); p.forward_and_loss(cam); lm = p.read_loss()
        numerical = (lp - lm) / (2 * eps)
        assert abs(numerical - analytic[axis]) <= 0.06 * abs(numerical) + 3e-2, (
            f"mean axis {axis}: analytic {analytic[axis]:.4f} vs "
            f"numerical {numerical:.4f}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_training_updates_rest_and_reduces_loss(backend: str) -> None:
    """A short SH-on training run must drive the rest coefficients off zero (the
    rest-Adam is stepping) and reduce the loss toward a colored target."""
    W = H = 24
    p = _make_pipeline(backend, sh_degree=3, width=W, height=H)
    cam = simple_camera(W, H, eye=(-1.0, -1.0, 0.0), target=(0.0, 0.0, 3.0),
                        fovy_degrees=60.0)

    means = np.array([[0.0, 0.0, 3.0]], np.float32)
    log_scales = np.log(np.full((1, 3), 0.7, np.float32))
    rot = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    opac = np.array([0.99], np.float32)
    p.set_primitive_from_numpy(means=means, log_scales=log_scales, rot_quats=rot,
                               opacities=opac,
                               color_logits=np.zeros((1, 3), np.float32))
    p.set_active_sh_degree(3)

    target = np.zeros((H, W, 3), np.float32)
    target[..., 0] = 0.9; target[..., 1] = 0.1; target[..., 2] = 0.2
    p.targets.set_image(target)

    rest_floats = p.buffers.sh.rest_floats
    # features_rest starts at zero (set_gaussians clears it).
    assert np.abs(_read_features_rest(p, 1)).max() == 0.0

    p.forward_and_loss(cam)
    loss0 = p.read_loss()

    lrs = AdamLRs()
    for _ in range(60):
        p.train_step(cam, lrs=lrs)

    p.forward_and_loss(cam)
    loss1 = p.read_loss()
    rest = _read_features_rest(p, 1)[0]

    assert np.abs(rest).max() > 1e-4, "rest coefficients never moved off zero"
    assert loss1 < loss0, f"loss did not decrease: {loss0:.3f} -> {loss1:.3f}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_active_band_adam_matches_full_width(backend: str) -> None:
    """The rest-Adam active-band optimization (iterate only the active SH
    coefficients) must be BIT-IDENTICAL to the full-width step.

    At a partial active degree the inactive higher bands have exactly-zero
    gradient (autodiff of evalSHColor never touches them), so their Adam update
    is a parameter no-op. Capping the loop to the active head must therefore
    leave features_rest byte-for-byte equal to a reference run that steps the
    full width — and the inactive bands must stay exactly zero throughout.
    """
    W = H = 24
    cam = simple_camera(W, H, eye=(-1.0, -1.0, 0.0), target=(0.0, 0.0, 3.0),
                        fovy_degrees=60.0)
    means = np.array([[0.0, 0.0, 3.0]], np.float32)
    log_scales = np.log(np.full((1, 3), 0.7, np.float32))
    rot = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    opac = np.array([0.99], np.float32)
    target = np.zeros((H, W, 3), np.float32)
    target[..., 0] = 0.9; target[..., 1] = 0.1; target[..., 2] = 0.2
    active_deg = 1  # only band 1 (9 of 45 rest floats) is active

    def _run(adam_degree: int) -> np.ndarray:
        # forward + backward ALWAYS run at active_deg, so only band 1 receives a
        # non-zero gradient (higher bands are provably zero). ``adam_degree``
        # independently sets how many rest floats the Adam step iterates:
        #   active_deg  -> capped run (Adam touches only the 9 active floats)
        #   full degree -> reference run (Adam iterates all 45 rest floats)
        # Because the inactive-band gradients are exactly zero, the two runs must
        # produce byte-identical features_rest — that's the property under test.
        # The phases are driven separately (not via train_step) precisely because
        # production ties forward/backward/adam to a single sh_degree; the test
        # needs them decoupled to exercise the active-band cap in isolation.
        p = _make_pipeline(backend, sh_degree=3, width=W, height=H)
        p.set_primitive_from_numpy(
            means=means, log_scales=log_scales, rot_quats=rot, opacities=opac,
            color_logits=np.zeros((1, 3), np.float32))
        p.set_active_sh_degree(active_deg)
        p.targets.set_image(target)
        lrs = AdamLRs()
        for _ in range(40):
            p.step_split_for_active_band_test(cam, lrs=lrs, adam_degree=adam_degree)
        p.device.wait_for_idle()
        return _read_features_rest(p, 1)[0]

    capped = _run(adam_degree=active_deg)
    full = _run(adam_degree=3)

    rest_per_band = 3
    n_active = ((active_deg + 1) ** 2 - 1) * rest_per_band  # 9 floats
    # Active head must match the full-width reference (up to 1-2 fp16 ULPs on Metal atomic_float).
    if backend == "metal":
        np.testing.assert_allclose(capped[:n_active], full[:n_active], rtol=1e-3, atol=1e-4)
    else:
        np.testing.assert_array_equal(capped[:n_active], full[:n_active])
    # Inactive higher bands must be untouched (exactly zero) in BOTH runs.
    assert np.all(capped[n_active:] == 0.0), "capped run moved an inactive band"
    assert np.all(full[n_active:] == 0.0), "full run moved an inactive band"
    # And the active band must actually have moved (the step is doing work).
    assert np.abs(capped[:n_active]).max() > 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_features_rest_survives_morton_sort(backend: str) -> None:
    """features_rest + its Adam moments must be permuted in lockstep with the
    gaussians by a Morton re-sort, or the coefficients desync
    from their gaussians."""
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    max_splats = 64
    p = TestPipeline(device=device, width=32, height=32, max_splats=max_splats,
                     loss_config=LossConfig(ssim_weight=0.0),
                     appearance_config=AppearanceConfig(sh_degree=3))

    n = 16
    rng = np.random.default_rng(3)
    means = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    log_scales = np.log(np.full((n, 3), 0.05, np.float32))
    rot = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))
    opac = np.linspace(0.1, 0.9, n).astype(np.float32)
    # sh0.r = slot index fingerprint (survives the permute via the gaussian move).
    color = np.stack([np.arange(n, dtype=np.float32)] * 3, axis=1)
    p.set_primitive_from_numpy(means=means, log_scales=log_scales, rot_quats=rot,
                               opacities=opac, color_logits=color)

    # Fingerprint coeff-0 of features_rest and adam_rest (m) with the slot index
    # (set AFTER set_gaussians, which zero-clears both). fp16(0..15) is exact.
    rest_floats = p.buffers.sh.rest_floats
    rest = np.zeros((max_splats, rest_floats), np.float32)
    rest[:n, 0] = np.arange(n)
    p.buffers.sh.features_rest.copy_from_numpy(pack_features_rest(rest))
    m = np.zeros((max_splats, rest_floats), np.float32)
    m[:n, 0] = np.arange(n)
    p.buffers.sh.adam_rest.copy_from_numpy(
        _pack_adam_mv(m, np.zeros_like(m), p.max_splats))

    world_min = means.min(axis=0) - 1.0
    world_max = means.max(axis=0) + 1.0
    p.morton.reorder(world_min, world_max)
    p.device.wait_for_idle()

    records = np.frombuffer(p.buffers.primitive.values.to_numpy().tobytes(),
                            dtype=p.buffers.primitive.layout.dtype)
    sorted_sh0_r = records["sh0"][:n, 0]   # original slot index carried in sh0.r

    rest_out = _read_features_rest(p, n)[:, 0]
    arest_out = _read_adam_m(p, n)[:, 0]

    assert np.allclose(rest_out, sorted_sh0_r), (
        "features_rest not permuted in lockstep with gaussians")
    assert np.allclose(arest_out, sorted_sh0_r), (
        "adam_rest not permuted in lockstep with gaussians")


def _force_clone_eta(p: TestPipeline, idx: int = 0) -> None:
    """Inject eta accumulators + grad signal to force a CLONE of gaussian idx
    (mirrors tests/test_structure_densify.py::_set_eta)."""
    b = p.buffers
    N = p.max_splats

    def w(buf, val, dtype=np.uint32):
        a = np.zeros(N, dtype=dtype)
        a[idx] = val
        buf.copy_from_numpy(a)

    w(b.eta.n_total, 10)
    w(b.eta.components["n_high"], 10)
    w(b.eta.components["n_low"], 0)
    for key, val in (("max_x", 4.0), ("max_y", 2.0), ("max_z", 0.05)):
        w(b.eta.components[key], int(val * 65536.0))
    from trainer.backend.memory.buffers import encode_grad_array
    w(b.densification.visibility_count, 1)
    a = np.zeros(N, dtype=np.float64)
    a[idx] = 1.0
    b.densification.grad_mean_2d_mag.copy_from_numpy(
        encode_grad_array(b.densification.grad_mean_2d_mag, a))


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_densify_clone_inherits_rest(backend: str) -> None:
    """A structure-densify CLONE must duplicate the gaussian AND its
    features_rest coefficients, with the child getting fresh (zero) rest-Adam."""
    from trainer.config.config import StructureConfig

    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    p = TestPipeline(device=device, width=64, height=64, max_splats=64,
                     loss_config=LossConfig(ssim_weight=0.0),
                     appearance_config=AppearanceConfig(sh_degree=3))

    p.set_primitive_from_numpy(
        means=np.array([[0.0, 0.0, 3.0]], np.float32),
        log_scales=np.log(np.full((1, 3), 0.1, np.float32)),
        rot_quats=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        opacities=np.array([0.9], np.float32),
        color_logits=np.zeros((1, 3), np.float32))

    rest_floats = p.buffers.sh.rest_floats
    rest = np.zeros((64, rest_floats), np.float32)
    parent_rest = (np.arange(rest_floats, dtype=np.float32) + 1.0) * 0.01
    rest[0] = parent_rest
    p.buffers.sh.features_rest.copy_from_numpy(pack_features_rest(rest))
    # Stamp the parent's rest-Adam non-zero so we can prove the child gets fresh.
    m = np.zeros((64, rest_floats), np.float32)
    m[0] = 7.0
    p.buffers.sh.adam_rest.copy_from_numpy(_pack_adam_mv(m, m, p.max_splats))

    _force_clone_eta(p, 0)
    cfg = StructureConfig(tau_split=0.8, tau_prune=0.8, split_power=0.5,
                          max_split_factor=4, dense_scale=0.2)  # small -> CLONE
    p.structure_densifier.densify(cfg)
    p.device.wait_for_idle()

    count = int(p.buffers.primitive.active_count.to_numpy().view(np.uint32)[0])
    assert count == 2, f"expected 1 clone (count 2), got {count}"

    rest_out = _read_features_rest(p, 2)
    arest_m = _read_adam_m(p, 2)

    # Child (index 1) is a bit-exact fp16 copy of the parent's coefficients;
    # the parent matches the input within fp16 quantization.
    np.testing.assert_array_equal(rest_out[1], rest_out[0])
    np.testing.assert_allclose(rest_out[0], parent_rest, rtol=1e-2, atol=2e-3)
    assert np.all(arest_m[1] == 0.0), "cloned child's rest-Adam must be fresh (zero)"


def test_sh_ply_roundtrip_features_rest(tmp_path) -> None:
    """features_rest survives a PLY write/read round-trip through the INRIA
    channel-major f_rest_* ordering (pure CPU, no GPU)."""
    from trainer.data.ply import load_ply, write_gaussian_ply

    n, rest_floats = 5, 45
    rng = np.random.default_rng(0)
    g = np.zeros(n, dtype=[("mean", "<f4", 3), ("log_scale", "<f4", 3), ("rot_quat", "<f4", 4),
                           ("opacity", "<f4"), ("sh0", "<f4", 3)])
    g["mean"] = rng.standard_normal((n, 3)).astype(np.float32)
    g["log_scale"] = rng.standard_normal((n, 3)).astype(np.float32)
    g["rot_quat"][:, 0] = 1.0
    g["opacity"] = 0.0
    g["sh0"] = rng.standard_normal((n, 3)).astype(np.float32)
    features_rest = rng.standard_normal((n, rest_floats)).astype(np.float32)

    path = tmp_path / "sh.ply"
    write_gaussian_ply(g, path, features_rest=features_rest)
    loaded = load_ply(path)

    assert loaded["features_rest"] is not None
    np.testing.assert_allclose(loaded["features_rest"], features_rest,
                               rtol=0, atol=1e-6)

    # The SH0 PLY (no features_rest) stays a 14-field file with no f_rest.
    path0 = tmp_path / "sh0.ply"
    write_gaussian_ply(g, path0)
    assert load_ply(path0)["features_rest"] is None


@pytest.mark.parametrize("backend", BACKENDS)
def test_sh_viewer_mirrors_view_dependent_color(backend: str) -> None:
    """The in-process viewer must allocate SH buffers, receive features_rest via
    update_gaussians (GPU->GPU copy), and render view-dependent color \u2014 i.e. the
    same forward as the trainer, from the viewer's own pipeline."""
    from trainer.frontend.viewer.viewer import Viewer

    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    train = TestPipeline(device=device, width=48, height=48, max_splats=4,
                         loss_config=LossConfig(ssim_weight=0.0),
                         appearance_config=AppearanceConfig(sh_degree=1))
    train.set_primitive_from_numpy(
        means=np.array([[0.0, 0.0, 0.0]], np.float32),
        log_scales=np.log(np.full((1, 3), 0.6, np.float32)),
        rot_quats=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        opacities=np.array([0.99], np.float32),
        color_logits=np.zeros((1, 3), np.float32))
    rest = np.zeros(train.buffers.sh.rest_floats, np.float32)
    rest[1 * 3 + 0] = 5.0   # band-1 z-term on R
    train.primitives.write_features_rest(rest.reshape(1, -1))
    train.set_active_sh_degree(1)

    # Viewer builds its own SH-enabled viewer_only pipeline; no window until start().
    viewer = Viewer(device, width=48, height=48, max_splats=4, sh_degree=1)
    assert viewer._pipeline.buffers.sh_enabled
    viewer.update_gaussians(train)
    device.wait_for_idle()

    def center(eye) -> np.ndarray:
        cam = simple_camera(48, 48, eye=eye, target=(0.0, 0.0, 0.0))
        viewer._pipeline.render(cam)
        img = viewer._pipeline.read_output_image()
        return img[24, 24, :3].astype(np.float64)

    front = center((0.0, 0.0, -4.0))   # dir ~ +z (bright R)
    side = center((4.0, 0.0, 0.0))     # dir ~ -x (base)
    assert front[0] - side[0] > 0.3, f"viewer not view-dependent: {front} vs {side}"







