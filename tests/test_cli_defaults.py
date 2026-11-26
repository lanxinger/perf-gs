# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Characterization test: freeze the v2 trainer CLI surface + effective defaults.

The refactor brief lists "CLI flag names or acceptance surface" and "effective
defaults" as NON-NEGOTIABLE. This asserts the FULL parsed-default namespace of
``build_parser()``, so any added / removed / renamed flag or changed default
fails loudly and forces a conscious update.

The parser now lives in ``trainer.frontend.cli`` (it was a scripts shim), so the
import below points there rather than at ``v2_train`` — which keeps this contract
test independent of the whole application's import graph. ``EXPECTED_DEFAULTS``
stays the contract the extracted parser must reproduce byte-for-byte, including for
the flags ``cli.py`` mounts from the shared ``trainer.frontend.settings`` table.
"""

from __future__ import annotations

import sys

from trainer.frontend.cli import build_parser

# dest -> default, exactly as build_parser() defines today. This is the frozen
# CLI contract. The former quirky `store_true, default=True` flags were fixed by
# the CLI/config-correctness refactor: cluster_cull / sad_relative_reset are now
# BooleanOptionalAction (still default True, but --no-... can disable them), and
# the three overlapping densify booleans (simple / importance / structure) were
# replaced by the single --densify-mode choice (default "structure_aware").
EXPECTED_DEFAULTS = {
    "scene": "datasets/360_v2/bicycle",
    "preset": "balanced",
    "iters": 3000,
    "lr_recipe": "short",
    "stereo_init": True,
    # 0 == follow --iters (app._resolve_position_lr_max_steps). It used to be a
    # hard 3000 that silently disagreed with any other --iters.
    "position_lr_max_steps": 0,
    "downsample": None,
    "width": None,
    "height": None,
    "preview_every": 0,
    # Loss/PSNR sampling for the viewer's LOSS graph and log lines; each tick re-renders. 0 = off.
    "log_every": 0,
    "render_all_every": 0,
    "evaluation": False,
    "eval_every": None,
    "eval_holdout": 1,
    "eval_holdout_stride": 8,
    "psnr_curve": False,
    "psnr_curve_every": 0,
    "densify_every": 100,
    "densify_from": 100,
    "densify_to": 1900,
    "n_scoring_views": 10,
    "t_opacity": 0.005,
    "t_big_scale": 0.1,
    "prune_scale3d": 1.0,
    "prune_blend_weight": 0.2,
    "prune_blend_from": 900,
    "reset_every": 1200,
    "opacity_reg": 0.0,
    "scale_reg": 0.0,
    "w_alpha": 0.0,
    "lambda_ssim": 0.3,
    "max_splats": 2_600_000,
    "max_sort_pairs": None,
    "final_splat_fraction": 1.0,
    "z_near_cull": 0.01,
    "profile_every": 0,
    "morton_every": 0,
    "cluster_cull": True,
    "densify_mode": "structure_aware",
    "sad_tau_split": 0.64,
    "sad_tau_prune": 0.8,
    "sad_grad_threshold": 1e-5,
    "sad_accum_every": 10,
    "sad_structure_every": 500,
    "sad_freq_grad_threshold": 0.00002,
    "sad_split_power": 0.5,
    "sad_max_split_factor": 2,
    "sad_ks_scale_power": 1.0,
    "sad_structure_min_opacity": 0.1,
    "sad_opacity_cap": 0.8,
    "sad_relative_reset": True,
    "sad_reset_decay": 0.1,
    "densify_diag": False,
    "sad_match_loss_lrs": 1,
    "sad_lambda_l2": 6.0,
    "lambda_l2": 0.0,
    "grad_thresh": 0.00058,
    "no_viewer": False,
    "launch_viewer": False,
    "wait_for_viewer": False,
    "hub": False,
    "glomap_bin": None,
    "colmap_bin": None,
    # install-time colmap/glomap build choice; CUDA by default (the only GPU path)
    "recon_cuda": True,
    "stream_every": 10,
    "vram_budget_gib": 0.0,
    "image_residency": "stream",
    "adam_moments": "bf16",
    "splat_kernel": "gaussian",
    "splat_profile_power": 1.0,
    "splat_profile_onset": 0.0,
    "backend": "metal" if sys.platform == "darwin" else "vulkan",
    "seed": 0,
    "mip_splatting": True,
    "mip_kernel_size": 0.1,
    "sh_degree": 2,
    "sh_band_interval": 0,
    "sample_mode": "sadgs",
    "run_name": None,
    "natural_vignetting": False, "lens_strength": None, "auto_lens": True,
    "init_opacity": 0.1,
    "grow_scale3d": 0.001,
    "opacity_reset_value": 0.01,
    "profile_run": False,
    "capture_iter": 0,
    "capture_external": False,
    "capture_frames": 240,
    "nsight_debug_names": None,
    "flat_bucket_backward": True,
    "backward_pixel_stride": 1,
    # STBN sparse supervision: 1.0 = every pixel (bit-identical to no STBN); see docs/stbn_pixel_sampling.md.
    # stbn_max_fraction 0 = the diagnostic staged-array override off.
    "stbn_pixel_fraction": 1.0,
    "stbn_warmup_iters": 0,
    "stbn_max_fraction": 0.0,
    "stbn_source": "nvidia",
    "stbn_file": "",
    # Global slice threshold: the faithful reading of the NVIDIA asset. The
    # per-tile-rank variant is kept as a selectable ablation.
    "stbn_select_mode": "global",
    "stbn_seed": 0,
    "stbn_independent": False,
    "extinction_hardness": 1.0,
    "extinction_hardness_warmup": 500,
    "final_test_psnr": False,
    "compute_final_metrics": False,
    "save_ply": False,
    # Evaluation and profiling are opt-in.
    "run_report": False,
    "save_test_renders": False,
    "render_fps_frames": 0,
    "stage_timing_every": 0,
    # None (not ()) so "no explicit ladder" is distinguishable from "an empty
    # one was passed"; ScheduleConfig.from_args normalizes both to ().
    "eval_iters": None,
    "eval_lpips": False,
    "write_ground_truths": False,
}


def test_cli_defaults_frozen() -> None:
    """Every CLI default matches the frozen contract (values AND flag set)."""
    args = build_parser().parse_args([])
    assert vars(args) == EXPECTED_DEFAULTS


def test_cli_flag_set_frozen() -> None:
    """The exact set of dest names is frozen — catches a flag added or removed
    even independently of its default value."""
    args = build_parser().parse_args([])
    assert set(vars(args)) == set(EXPECTED_DEFAULTS)


def test_cli_backend_choices() -> None:
    """Backend is restricted to the supported RHI backends."""
    assert build_parser().parse_args(["--backend", "d3d12"]).backend == "d3d12"
    assert build_parser().parse_args(["--backend", "vulkan"]).backend == "vulkan"
    assert build_parser().parse_args(["--backend", "metal"]).backend == "metal"
