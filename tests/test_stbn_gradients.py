# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""STBN sparse pixel supervision, on the GPU.

The design rests on one claim: the backward may SKIP the pixels this slice does
not sample, because the loss already zeroed their gradient, so skipping them
changes nothing but the time it takes. That claim is what these tests exist to
check, and it is checkable directly — the same run has a slow reference form and
a fast form:

  reference : mask applied in the loss only. The backward keeps its exact
              full-pixel loop and reaches the unsampled pixels, finds a zero
              gradient there, and adds nothing.  (no STBN_ENABLED define)
  compacted : the backward walks only the k sampled pixels of each tile.
              (STBN_ENABLED=1, the whole point of the feature)

Those must agree. If they ever stop agreeing, the compacted walk is dropping or
double-counting pixel terms, which no amount of PSNR-watching would localize.

The other two things worth pinning are that the loss's mask is exactly the mask
(zeroed where unsampled, scaled by TILE_AREA/k where sampled), and that the
sparse gradient really is centred on the dense one when averaged over a camera's
repeat visits — the property that lets the learning rate stay untouched.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from _pipeline_testkit import TestPipeline
from trainer.backend.render import stbn_table as stbn
from trainer.config.config import LossConfig, StbnConfig, StbnVariables
from trainer.backend.compute.execution import TILE_AREA, TILE_HEIGHT, TILE_WIDTH
from trainer.data.dataset import simple_camera
from trainer.backend.compute.device import create_device

BACKENDS = ["vulkan", "d3d12", "metal"]

GRAD_NAMES = [
    "mean_x", "mean_y", "mean_z",
    "log_scale_x", "log_scale_y", "log_scale_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
    "opacity", "color_r", "color_g", "color_b",
]

W = H = 64
N_SPLATS = 100
BUDGET = 6          # 6/64 ~ 9.4%, the low-density regime the feature targets


def _asset_or_skip() -> stbn.StbnSpec:
    spec = stbn.StbnSpec()
    if not spec.resolved_path().is_file():
        pytest.skip("NVIDIA STBN asset absent; run scripts/data/download_stbn.py")
    return spec


def _deep_scene(seed: int = 3):
    """Splats clustered in front of the camera so the central tiles span several
    checkpoint buckets — the case where the backward's pixel loop dominates and
    where a compaction bug would show up."""
    rng = np.random.default_rng(seed)
    means = np.zeros((N_SPLATS, 3), dtype=np.float32)
    means[:, 0] = rng.uniform(-0.2, 0.2, N_SPLATS)
    means[:, 1] = rng.uniform(-0.2, 0.2, N_SPLATS)
    means[:, 2] = rng.uniform(2.0, 4.0, N_SPLATS)
    return {
        "means": means,
        "log_scales": np.log(np.full((N_SPLATS, 3), 0.35, dtype=np.float32)),
        "rot_quats": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                             (N_SPLATS, 1)),
        "opacities": rng.uniform(0.3, 0.8, N_SPLATS).astype(np.float32),
        "color_logits": rng.uniform(-2.0, 2.0, (N_SPLATS, 3)).astype(np.float32),
    }


def _target() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    yy /= max(H - 1, 1)
    xx /= max(W - 1, 1)
    return np.stack([xx, yy, (1.0 - xx) * yy], axis=-1).astype(np.float32)


def _camera():
    return simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                         target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                         fovy_degrees=60.0)


def _pipeline(backend: str, *, compacted: bool, budget: int = BUDGET,
              ssim: float = 0.0, source: str = stbn.SOURCE_NVIDIA,
              select_mode: str = stbn.SELECT_GLOBAL) -> TestPipeline:
    """A pipeline whose SHADER DEFINES and StbnConfig agree.

    ``compacted`` is the only difference between the two paths under test: it
    injects STBN_ENABLED and tells the pipeline the
    backward now takes the mask parameters.

    ``select_mode`` picks how the density chooses pixels. The reference-vs-
    compacted equivalence must hold for BOTH, since the layout deliberately hides
    the mode from the shader — under a global threshold the per-tile count varies
    and the backward takes its loop bound from the table.
    """
    defines = {"STBN_ENABLED": "1"} if compacted else None
    try:
        device = create_device(backend=backend, defines=defines)
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"backend {backend!r} unavailable: {exc}")
    return TestPipeline(
        device=device, width=W, height=H, max_splats=128,
        loss_config=LossConfig(ssim_weight=ssim, alpha_weight=0.0),
        stbn_config=StbnConfig(
            source=source, pixel_fraction=budget / TILE_AREA,
            backward_compacted=compacted,
            select_mode=select_mode,
        ),
    )


def _expected_mask(select_mode: str, budget: int, slice_index: int):
    """The frame-sized mask the shader should have applied.

    Derived from the SAME helper the production table is built from, so the test
    pins the shader against the host's definition of the selection rather than
    against a re-implementation that could drift from it.
    """
    spec = dataclasses.replace(_asset_or_skip(), select_mode=select_mode,
                               fraction=budget / TILE_AREA)
    selected = stbn.selected_volume(spec, budget)[slice_index % spec.slices]
    return np.take(np.take(selected, np.arange(H) % spec.height, axis=0),
                   np.arange(W) % spec.width, axis=1)


def _gradients(pipeline: TestPipeline, scene, camera,
               variables: "StbnVariables | None"):
    pipeline.set_primitive_from_numpy(**scene)
    pipeline.targets.set_image(_target())
    pipeline.forward_and_loss(camera, stbn=variables)
    pipeline.optimizer.zero_grad()
    pipeline.run_backward(stbn=variables)
    pipeline.device.wait_for_idle()
    return pipeline.read_gradients()


def _compare(a: dict, b: dict, *, label_a: str, label_b: str,
             rel_tol: float = 0.02, abs_floor: float = 1e-4) -> None:
    """Per-component comparison with the tolerance convention the other backward
    cross-checks use: relative to the component's own magnitude, with an absolute
    floor so near-zero components don't trip on fixed-point LSBs."""
    mismatches = []
    for name in GRAD_NAMES:
        left, right = a[name], b[name]
        assert left.shape == right.shape, f"{name}: {left.shape} vs {right.shape}"
        scale = max(1e-6, float(np.max(np.abs(left))))
        abs_err = np.abs(left - right)
        worst = int(np.argmax(abs_err / scale))
        if abs_err[worst] / scale > rel_tol and abs_err[worst] > abs_floor:
            mismatches.append(
                f"{name}[g{worst}]: {label_a}={left[worst]:.6e} "
                f"{label_b}={right[worst]:.6e} rel={abs_err[worst] / scale:.4f}")
    if mismatches:
        pytest.fail(f"{len(mismatches)} gradient mismatches ({label_a} vs "
                    f"{label_b}):\n" + "\n".join(mismatches))


# ------------------------------------------------------------------ the loss

@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("select_mode",
                         [stbn.SELECT_GLOBAL, stbn.SELECT_TILE_RANK])
def test_loss_zeroes_unsampled_pixels_and_scales_the_sampled_ones(
        backend: str, select_mode: str) -> None:
    """The mask is DEFINED here, so pin it against the mask table directly.

    Unsampled pixels must be exactly zero (the backward's licence to skip them),
    and sampled ones must be the dense gradient times TILE_AREA/k. That weight is
    UNIFORM in both modes: a global threshold selects each pixel with probability
    rho, so 1/rho is the correct importance weight for the image sum, and a
    per-tile weight would re-impose exactly the tile-grid structure the global
    mode exists to remove. Comparing against a full-density run of the same scene
    makes this a statement about the mask alone, the photometric math cancelling.
    """
    slice_index = 11
    expected = _expected_mask(select_mode, BUDGET, slice_index)

    pipeline = _pipeline(backend, compacted=False, select_mode=select_mode)
    scene, camera = _deep_scene(), _camera()

    pipeline.set_primitive_from_numpy(**scene)
    pipeline.targets.set_image(_target())
    pipeline.forward_and_loss(camera, stbn=None)
    pipeline.device.wait_for_idle()
    dense = pipeline.buffers.loss.image_gradient.to_numpy()[..., :3].astype(np.float64)

    pipeline.forward_and_loss(camera, stbn=StbnVariables(
        slice_index=slice_index, pixels_per_tile=BUDGET))
    pipeline.device.wait_for_idle()
    sparse = pipeline.buffers.loss.image_gradient.to_numpy()[..., :3].astype(np.float64)

    assert (sparse[~expected] == 0.0).all(), "unsampled pixels are not zeroed"

    per_tile = expected.reshape(H // TILE_HEIGHT, TILE_HEIGHT,
                                W // TILE_WIDTH, TILE_WIDTH).sum(axis=(1, 3))
    if select_mode == stbn.SELECT_TILE_RANK:
        # Exactly k per tile, by construction — that IS the mode.
        assert per_tile.min() == per_tile.max() == BUDGET
    else:
        # A global threshold constrains the SLICE, not the tile, so the counts
        # must be free to vary while still averaging to the requested density.
        assert per_tile.min() <= BUDGET <= per_tile.max()

    weight = TILE_AREA / BUDGET
    np.testing.assert_allclose(sparse[expected], dense[expected] * weight,
                               rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_mask_off_in_a_compacted_build_matches_a_plain_dense_backward(
        backend: str) -> None:
    """A mask-OFF backward inside an STBN_ENABLED build must be the dense gradient.

    This is the dense REFERENCE that any sparse-vs-dense comparison is measured
    against, and it is easy to get wrong in a way nothing else notices: the
    compacted path takes its pixel-loop bound from the mask table, which still
    describes some density even when the mask is disabled. Reading it
    unconditionally made the "dense" backward visit only rho of the pixels while
    the loss weighted all of them by 1.0 — a gradient exactly 1/rho too small,
    with the sparse runs then looking 1/rho too LARGE.

    ``test_full_budget_is_bit_identical_to_no_mask_at_all`` cannot catch this:
    both of its arms take the mask-off path, so both would be wrong together.
    This compares against a build that has no STBN code at all.
    """
    scene, camera = _deep_scene(), _camera()

    plain = _pipeline(backend, compacted=False, source=stbn.SOURCE_WHITE)
    reference = _gradients(plain, scene, camera, None)

    compacted = _pipeline(backend, compacted=True, source=stbn.SOURCE_WHITE)
    measured = _gradients(compacted, scene, camera, None)

    _compare(measured, reference, label_a="compacted(mask off)",
             label_b="plain dense")


# --------------------------------------------------- reference vs compacted

@pytest.mark.parametrize("backend", BACKENDS)
def test_compacted_backward_matches_the_full_pixel_reference(backend: str) -> None:
    """THE correctness gate for the fast path.

    Same scene, same slice, same budget; the only difference is whether the
    backward walks the k-entry visit list or all TILE_AREA pixels and adds zeros
    for the unsampled ones. Agreement means the compaction enumerates exactly the
    sampled pixels, pairs each with the right checkpoint slot, and keeps the wave
    scans in lockstep. Residual disagreement is float summation ORDER (rank order
    vs raster order) plus fixed-point LSBs, hence the relative tolerance rather
    than bit-equality.
    """
    _asset_or_skip()
    scene, camera = _deep_scene(), _camera()
    variables = StbnVariables(slice_index=7, pixels_per_tile=BUDGET)

    reference = _gradients(_pipeline(backend, compacted=False), scene, camera,
                           variables)
    compacted = _gradients(_pipeline(backend, compacted=True), scene, camera,
                           variables)
    _compare(reference, compacted, label_a="reference", label_b="compacted")


@pytest.mark.parametrize("backend", BACKENDS)
def test_compacted_backward_matches_reference_with_ssim(backend: str) -> None:
    """Same gate with the SSIM term on, since SSIM masks at a different point in
    the loss (pass 2's output, not pass 1's maps) and is the default objective."""
    _asset_or_skip()
    scene, camera = _deep_scene(), _camera()
    variables = StbnVariables(slice_index=3, pixels_per_tile=BUDGET)

    reference = _gradients(_pipeline(backend, compacted=False, ssim=0.2),
                           scene, camera, variables)
    compacted = _gradients(_pipeline(backend, compacted=True, ssim=0.2),
                           scene, camera, variables)
    _compare(reference, compacted, label_a="reference", label_b="compacted")


@pytest.mark.parametrize("backend", BACKENDS)
def test_full_budget_is_bit_identical_to_no_mask_at_all(backend: str) -> None:
    """The off switch must be inert, not merely cheap.

    A budget of TILE_AREA means "sample everything", and the kernels take that as
    the early-out before ever reading the table — so it has to reproduce the
    unmasked gradient EXACTLY, whatever slice is nominally selected. Anything
    less and every default run would be paying for a feature it disabled.
    """
    scene, camera = _deep_scene(), _camera()
    pipeline = _pipeline(backend, compacted=False, source=stbn.SOURCE_WHITE)

    unmasked = _gradients(pipeline, scene, camera, None)
    full = _gradients(pipeline, scene, camera,
                      StbnVariables(slice_index=5, pixels_per_tile=TILE_AREA))
    for name in GRAD_NAMES:
        np.testing.assert_array_equal(
            unmasked[name], full[name],
            err_msg=f"{name}: a full budget perturbed the gradient")


# ------------------------------------------------------------- unbiasedness

@pytest.mark.parametrize("backend", BACKENDS)
def test_sparse_gradient_is_centred_on_the_dense_one_over_a_visit_cycle(
        backend: str) -> None:
    """The claim that keeps the learning rate untouched.

    A single sparse step is a noisy estimate — that is the trade. What must hold
    is that averaging over a camera's successive visits converges on the dense
    gradient, since that is the sequence the mask's temporal axis walks. So:
    average the estimator over a run of slices and require it to land far closer
    to the dense gradient than any individual slice does.

    The mean is not exact even over a full cycle: a pixel's exposure across the
    mask's slices is deterministic and spread ~+-2% absolute around rho (pinned in
    test_stbn.py), so what is asserted is the CONVERGENCE, not equality.
    """
    spec = _asset_or_skip()
    scene, camera = _deep_scene(), _camera()
    pipeline = _pipeline(backend, compacted=True)

    dense_pipeline = _pipeline(backend, compacted=False)
    dense = _gradients(dense_pipeline, scene, camera, None)

    n_slices = min(spec.slices, 32)
    per_slice = [
        _gradients(pipeline, scene, camera,
                   StbnVariables(slice_index=t, pixels_per_tile=BUDGET))
        for t in range(n_slices)
    ]

    for name in ("mean_x", "mean_y", "opacity", "color_r"):
        reference = dense[name]
        scale = max(1e-6, float(np.abs(reference).max()))
        averaged = np.mean([g[name] for g in per_slice], axis=0)
        single = per_slice[0][name]

        averaged_err = float(np.abs(averaged - reference).max()) / scale
        single_err = float(np.abs(single - reference).max()) / scale
        assert averaged_err < single_err / 2.0, (
            f"{name}: averaging {n_slices} visits did not converge "
            f"(mean err {averaged_err:.3f} vs single-slice {single_err:.3f})")
        assert averaged_err < 0.35, (
            f"{name}: mean over {n_slices} visits is {averaged_err:.3f} off the "
            f"dense gradient — the 1/rho weighting looks miscalibrated")


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_budget_of_a_whole_tile_is_representable(backend: str) -> None:
    """The backward stages a FULL tile and clamps the runtime budget to it, so
    every budget in [1, TILE_AREA] is expressible - there is no capacity a
    schedule can outgrow, and therefore nothing to validate against. (There used
    to be: the staged size was a CLI knob, on the theory that shrinking it bought
    occupancy. At an 8x8 tile those arrays are under 2 KB of ~100 KB per SM, so it
    bought nothing and the knob is now a hand-only diagnostic.)"""
    scene, camera = _deep_scene(), _camera()
    pipeline = _pipeline(backend, compacted=True, source=stbn.SOURCE_WHITE)
    grads = _gradients(pipeline, scene, camera,
                       StbnVariables(slice_index=0, pixels_per_tile=TILE_AREA))
    assert np.isfinite(grads["opacity"]).all()
