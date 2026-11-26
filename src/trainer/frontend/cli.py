# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""CLI parser, the single boundary that produces the untyped Namespace. Flags the hub panel
also offers are declared once as shared ``settings.SETTINGS`` rows and mounted via ``_shared``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from trainer.common.lens import parse_strength
from trainer.config.config import (
    DensifyConfig,
    LossConfig,
    RasterConfig,
    RegularizationConfig,
    StbnConfig,
    StructureConfig,
)
from trainer.frontend.settings import PRESET, SETTINGS_MAP, preset_values


def _shared(parser: argparse.ArgumentParser, dest: str) -> None:
    SETTINGS_MAP[dest].add_to(parser)


class _TrainerParser(argparse.ArgumentParser):
    def parse_args(self, args: Sequence[str] | None = None,
                   namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        """Fill in the chosen preset before argparse applies explicit flags."""
        namespace = argparse.Namespace() if namespace is None else namespace
        preset = self.parse_known_args(args, argparse.Namespace(**vars(namespace)))[0].preset
        for key, value in preset_values(preset).items():
            vars(namespace).setdefault(key, value)
        vars(namespace).setdefault("evaluation", None)  # Preserve an explicit --no-evaluation.
        result = super().parse_args(args, namespace)
        if result.evaluation is None:
            # A requested report or checkpoint enables only the requested measurements.
            requested = result.run_report or result.final_test_psnr or result.compute_final_metrics
            result.evaluation = bool(requested or result.eval_iters or (result.eval_every or 0) > 0
                                     or result.eval_lpips)
            if result.eval_every is None and (requested or result.eval_iters):
                result.eval_every = 0
        if result.scene is None and not result.hub:
            result.scene = "datasets/360_v2/bicycle"
        if result.lens_strength is not None or result.natural_vignetting:
            result.auto_lens = False
        return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = _TrainerParser()
    parser.add_argument("--scene", help="scene folder to train or open in the hub")
    _shared(parser, PRESET)
    _shared(parser, "iters")
    _shared(parser, "lr_recipe")
    parser.add_argument("--position-lr-max-steps", type=int, default=0,
                    help="horizon for exponential position LR decay. 0 follows the preset schedule "
                         "(--iters, or 100 camera passes with a 3,500-step minimum for Quality+).")
    parser.add_argument("--downsample", type=int, default=None,
                    help="downsample factor (1=full res, 2=half, 4=quarter). "
                         "Defaults to 2 for indoor Mip-NeRF 360 scenes and 4 otherwise. "
                         "Uses images_N if available, otherwise resizes on load.")
    parser.add_argument("--width", type=int, default=None,
                    help="force render width (overrides --downsample)")
    parser.add_argument("--height", type=int, default=None,
                    help="force render height (overrides --downsample)")
    parser.add_argument("--preview-every", type=int, default=0,
                    help="sample training-view loss and PSNR every N iterations (default: off). "
                         "With evaluation enabled, previews also render every camera.")
    _shared(parser, "log_every")
    parser.add_argument("--render-all-every", type=int, default=0,
                    help="dump PNGs for ALL cameras every N iters")
    _shared(parser, "evaluation")
    parser.add_argument("--eval-every", type=int, default=None,
                    help="every N iters, render ALL cameras and compute the honest "
                         "average PSNR (same metric as the final_psnr.json summary). "
                         "With --evaluation, defaults to the last iteration; 0 = off. "
                         "Writes results to eval_curve.jsonl in the run dir with "
                         "iter, elapsed_s (pure training wall-clock, excluding eval "
                         "time), and avg_psnr_lin/srgb/srgb_masked.")
    parser.add_argument("--eval-iters", type=int, nargs="*", default=None,
                    help="explicit extra iterations to run the --eval-every "
                         "sweep at, UNION-ed with the --eval-every cadence "
                         "(e.g. --eval-iters 250 500 750 1000 1500 2000 2500 "
                         "3000). For an irregular convergence ladder: a single "
                         "interval that hit all of those would also fire on "
                         "every other multiple, and each sweep renders and "
                         "scores the entire held-out set. Default none.")
    parser.add_argument("--eval", dest="eval_holdout", type=int, default=1,
                    help="hold out every Nth camera from training (1, default) or train on all (0). "
                         "The split is independent of metric collection.")
    parser.add_argument("--eval-holdout-stride", type=int, default=8,
                    help="holdout stride for --eval (3DGS llffhold=8). Test set is "
                         "cameras where index %% stride == 0.")
    parser.add_argument("--psnr-curve", action="store_true",
                    help="capture held-out TEST-set PSNR every --psnr-curve-every iters, "
                         "sampled one iteration BEFORE each densify step (densify-1) so the "
                         "reading reflects the settled pre-densify state. Writes "
                         "psnr_curve.jsonl (streamed) + psnr_curve.json (iters[]/psnr_srgb[], "
                         "built for cross-run averaging) and renders psnr_curve.png (PSNR vs "
                         "iter) into the run dir at the end. Independent of --eval-every. "
                         "Average many runs with scripts/plots/plot_psnr_curves.py.", default=False)
    parser.add_argument("--psnr-curve-every", type=int, default=0,
                    help="capture cadence for --psnr-curve. A sample is taken at iter `it` "
                         "when (it+1) %% this == 0, i.e. one iter before each densify boundary "
                         "when --densify-every divides this (default 100 aligns with the "
                         "default densify schedule).")
    _shared(parser, "densify_every")
    _shared(parser, "densify_from")
    _shared(parser, "densify_to")
    parser.add_argument("--n-scoring-views", type=int, default=StructureConfig.n_scoring_views)
    parser.add_argument("--t-opacity", type=float, default=DensifyConfig.t_opacity,
                    help="prune splats with sigmoid(opacity) below this. MUST stay BELOW "
                         "--opacity-reset-value (default 0.01): opacity reset clamps every "
                         "splat down to the reset value, so a floor above it prunes the "
                         "entire scene the instant reset fires. 3DGS: prune 0.005 < reset 0.01.")
    parser.add_argument("--t-big-scale", type=float, default=0.1)
    parser.add_argument("--prune-scale3d", type=float, default=1.0,
                    help="prune splats whose max 3D scale exceeds this * scene_extent "
                         "(gsplat prune_scale3d). Only active after the first opacity reset.")
    _shared(parser, "prune_blend_weight")
    _shared(parser, "prune_blend_from")
    _shared(parser, "reset_every")
    parser.add_argument("--opacity-reg", type=float,
                        default=RegularizationConfig.opacity_weight)
    parser.add_argument("--scale-reg", type=float, default=RegularizationConfig.scale_weight)
    parser.add_argument("--w-alpha", type=float, default=LossConfig.alpha_weight,
                    help="alpha loss weight; set > 0 for RGBA targets (e.g. 1.0)")
    _shared(parser, "lambda_ssim")
    _shared(parser, "max_splats")
    _shared(parser, "final_splat_fraction")
    parser.add_argument("--max-sort-pairs", type=int, help="Training renderer's splat–tile pair capacity; defaults to a budget derived from --max-splats.")
    parser.add_argument("--z-near-cull", type=float, default=RasterConfig.z_near_cull,
                    help="view-space depth floor for bin cull; raise for scenes "
                         "where points can land near a camera pose (e.g. 0.2)")
    parser.add_argument("--profile-every", type=int, default=0,
                    help="profile one full step every N iters (0 to disable)")
    parser.add_argument("--morton-every", type=int, default=0,
                    help="Extra Morton-order (Z-curve) re-sorts of the gaussian + Adam "
                         "buffers every N iters, on top of the one after each densify. "
                         "0 (default) = only after densify. Improves cache locality of "
                         "subsequent per-splat reads (bin_count, blend backward, "
                         "adam) as the array grows large. Follows clear_tracking "
                         "(and clear_structure_scores on SAD runs) since window "
                         "accumulators are slot-indexed and must be reset when "
                         "slots are permuted.")
    parser.add_argument("--cluster-cull", action=argparse.BooleanOptionalAction,
                    default=RasterConfig.cluster_cull,
                    help="Cluster-frustum cull before each forward: bound the "
                         "128-splat clusters (Morton-adjacent) by a per-cluster "
                         "AABB and skip clusters fully outside the view frustum. "
                         "On by default; pass --no-cluster-cull to disable. "
                         "Relies on the Morton re-sort after each densify for the "
                         "clusters to be spatially coherent. Mask-in-bin_count design (no compaction, "
                         "no slot remap) — every downstream pass is unaware.")
    _shared(parser, "densify_mode")
    parser.add_argument("--sad-tau-split", type=float, default=0.64,
                    help="SAD split_ratio_threshold: split if high_ratio (fraction of views "
                         "with scalar eta>1) exceeds this AND the gradient is high (SADGS 0.8)")
    parser.add_argument("--sad-tau-prune", type=float, default=StructureConfig.tau_prune,
                    help="SAD prune_ratio_threshold: prune if low_ratio (fraction of views "
                         "with eta<0.1) exceeds this (SADGS 0.8)")
    parser.add_argument("--sad-grad-threshold", type=float,
                    default=StructureConfig.grad_threshold,
                    help="SAD is_grad_high floor: avg 2D-grad magnitude a gaussian must exceed "
                         "to be split-eligible (SADGS hardcodes 1e-5)")
    parser.add_argument("--sad-accum-every", type=int, default=10,
                    help="accumulate an eta sample every N iters (SADGS iteration%%10==0). "
                         "eta_max is a running max, so sampling every iter (=1) inflates it "
                         "and oversizes splits. Default 10 matches SADGS.")
    parser.add_argument("--sad-structure-every", type=int, default=500,
                    help="DUAL SCHEDULE (paper Sec 4): run the aggressive structure-aware "
                         "densify every N iters (paper: 500), while the gentle AbsGS gradient "
                         "densify runs every --densify-every (paper: 100) to populate empty "
                         "regions in between. Set equal to --densify-every to disable the dual "
                         "schedule (structure densify every step, no AbsGS interleave).")
    parser.add_argument("--sad-freq-grad-threshold", type=float, default=0.00002,
                    help="SAD is_active_vis per-view grad floor (SADGS freq_grad_threshold). "
                         "eta is only folded in from views where the gaussian has a high "
                         "per-view screen gradient; rescaled by W*H*3 into our accumulator's "
                         "sum-of-abs convention. 0 disables the per-view grad gate.")
    parser.add_argument("--sad-split-power", type=float, default=StructureConfig.split_power,
                    help="SAD: per-axis split factor n_k = ceil(eta_max ** power); 0.5 = sqrt (paper)")
    parser.add_argument("--sad-max-split-factor", type=int,
                    default=StructureConfig.max_split_factor,
                    help="SAD: per-axis cap on the grid split factor n_k. The default 2 trains 17%% "
                         "faster than SADGS's max_clones_per_axis=8 for ~0.04 dB less PSNR")
    parser.add_argument("--sad-ks-scale-power", type=float,
                    default=StructureConfig.ks_scale_power,
                    help="SAD: child scale along axis k = sigma_k / n_k**power (SADGS "
                         "ks_scale_power). 1.0 for Mip-NeRF 360 (variance-preserving with the "
                         "sqrt(12) child spacing); 1.2 for Tanks & Temples / Deep Blending.")
    parser.add_argument("--sad-structure-min-opacity", type=float, default=0.1,
                    help="structure-pass opacity prune floor. SADGS uses 0.1 (20x the "
                         "warmup 0.005), pruning the post-split population hard so weak "
                         "children don't survive to compound into the next structure pass. "
                         "Default 0.005 = current behavior; set 0.1 to match the reference.")
    parser.add_argument("--sad-opacity-cap", type=float, default=StructureConfig.opacity_cap,
                    help="post-densify opacity cap applied to ALL survivors after the "
                         "structure prune (SADGS inverse_sigmoid(min(opacity,0.8))). "
                         "0 = off (current). Set 0.8 to match the reference (measured: "
                         "ref it500-post opacity p90=p99=max=0.8).")
    parser.add_argument("--sad-relative-reset", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="use SADGS RELATIVE opacity reset (opacity *= decay) instead of "
                         "the gsplat CLAMP-to-0.01. Preserves strong/weak ordering through "
                         "reset. On by default; pass --no-sad-relative-reset to disable. "
                         "REQUIRED when --sad-structure-min-opacity=0.1 so the floor "
                         "doesn't wipe the scene at the first post-reset structure pass.")
    parser.add_argument("--sad-reset-decay", type=float, default=DensifyConfig.reset_decay,
                    help="RELATIVE-reset multiplier (SADGS opacity_reset_decay=0.1). "
                         "Only used with --sad-relative-reset.")
    parser.add_argument("--densify-diag", action="store_true",
                    help="when a structure-aware densify tick overflows --max-splats, "
                         "also print its split-gate breakdown (view counts, high-view "
                         "ratio, gradient and eta percentiles). Opt-in: it reads back "
                         "the full eta buffers, which can be hundreds of MB.")
    parser.add_argument("--sad-match-loss-lrs", type=int, default=1,
                    help="when --densify-mode structure_aware: 1 (default) = use SADGS's loss (lambda_l2) "
                         "and per-parameter LRs so we train exactly as they do; 0 = keep our "
                         "own loss/LR defaults and only swap the densification strategy.")
    parser.add_argument("--sad-lambda-l2", type=float, default=6.0,
                    help="SADGS additive L2 loss weight (their opt.lambda_l2). Applied when "
                         "--densify-mode structure_aware and --sad-match-loss-lrs (else --lambda-l2 is used).")
    parser.add_argument("--lambda-l2", type=float, default=LossConfig.l2_weight,
                    help="additive L2 loss weight for non-SAD runs (0 = pure L1(+SSIM))")
    _shared(parser, "grad_thresh")
    parser.add_argument("--no-viewer", action="store_true", help="disable live viewer stream")
    # Older launch scripts pass this flag. The viewer now starts automatically.
    parser.add_argument("--launch-viewer", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wait-for-viewer", action="store_true",
                    help="after initializing the gaussians, stream the initial "
                         "state and BLOCK until the viewer's 'Start Training' "
                         "button is pressed (or Enter here). Lets you inspect "
                         "the init before training begins. No-op with --no-viewer.")
    parser.add_argument("--hub", action="store_true",
                    help="open the interactive training hub to choose a dataset and preset, "
                         "train and export a PLY. Add --scene PATH to open a dataset directly.")
    parser.add_argument("--glomap-bin", type=str, default=None,
                    help="Path to a glomap executable (global SfM mapper). Falls "
                         "back to $GLOMAP_PATH, the copy under .recon-tools/, then "
                         "PATH; unresolved, mapping uses colmap's own mapper.")
    parser.add_argument("--colmap-bin", type=str, default=None,
                    help="Path to a colmap executable — the reconstruction "
                         "front-end. Falls back to $COLMAP_PATH, the copy under "
                         ".recon-tools/, then PATH.")
    parser.add_argument("--recon-cuda", "--glomap-cuda",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Use the CUDA builds of colmap/glomap (also a checkbox in "
                         "the hub's DATASET panel). On by default: it is what puts "
                         "SIFT extraction/matching and bundle adjustment on the GPU. "
                         "Pass --no-recon-cuda for the CPU-only zips. The two "
                         "variants install side by side, so switching re-downloads.")
    _shared(parser, "stream_every")
    parser.add_argument("--vram-budget-gib", type=float, default=0.0,
                    help="GPU memory this process may use, in GiB. 0 (default) = "
                         "the budget Windows (DXGI) grants the process; no limit "
                         "where that is unavailable. --max-splats is lowered "
                         "(with a warning) when it would not fit.")
    _shared(parser, "image_residency")
    _shared(parser, "adam_moments")
    _shared(parser, "splat_kernel")
    _shared(parser, "splat_profile_power")
    _shared(parser, "splat_profile_onset")
    parser.add_argument(
        "--backend",
        type=str,
        default="metal" if sys.platform == "darwin" else "vulkan",
        choices=["vulkan", "d3d12", "metal"],
    )
    parser.add_argument("--seed", type=int, default=0)
    _shared(parser, "mip_splatting")
    parser.add_argument("--mip-kernel-size", type=float, default=0.1,
                    help="Mip-Splatting 2D low-pass dilation (mip canonical 0.1). Only used "
                         "with --mip-splatting; vanilla uses 0.3 with no compensation.")
    _shared(parser, "sh_degree")
    _shared(parser, "sh_band_interval")
    parser.add_argument("--sample-mode", type=str, default="sadgs", choices=["ours", "sadgs"],
                    help="per-iteration train-camera draw. 'ours': "
                         "np.random.default_rng(seed).integers over the name-sorted "
                         "train indices (with replacement, 1/iter). 'sadgs' (default): replicate "
                         "the reference SADGS trainer's EXACT draw — Python random.seed(seed), "
                         "a load-time random.shuffle of train then test cameras (INRIA "
                         "Scene shuffle=True), first epoch consumed in that load-shuffled "
                         "order, then random.shuffle on every refill (epoch cycling, "
                         "without replacement), batch_size=1. Use for a like-for-like "
                         "PSNR comparison that isolates the renderer from the data order.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--stereo-init", action=argparse.BooleanOptionalAction, default=True,
                        help="triangulate additional seeds from training views")
    lens = parser.add_mutually_exclusive_group()
    lens.add_argument("--auto-lens", action=argparse.BooleanOptionalAction, default=True,
                      help="fit lens falloff from shared training landmarks by default")
    lens.add_argument("--lens-strength", type=parse_strength, default=None,
                        help="strength of cos-fourth lens falloff: 0 = flat, 1 = natural; overrides --natural-vignetting")
    parser.add_argument("--natural-vignetting", action="store_true", help="render the lens cos-fourth response in sRGB")
    parser.add_argument("--init-opacity", type=float, default=0.1, help="initial opacity for all splats")
    parser.add_argument("--grow-scale3d", type=float, default=0.001,
                    help="Scale factor for 3D growth of Gaussians (for split/clone routing).")
    parser.add_argument("--opacity-reset-value", type=float,
                    default=DensifyConfig.t_opacity_reset_value, help="value to reset opacity to when doing opacity reset")
    parser.add_argument("--profile-run", action="store_true",
                    help="Nsight Graphics profiling mode. Runs a NORMAL training loop "
                         "(densify + logs still on, per your schedule) but also opens "
                         "a Vulkan window and Present()s one frame per train_step so "
                         "Nsight has a frame boundary to capture on. Auto-disables the "
                         "viewer stream (a second device pollutes the capture) and the "
                         "final I/O-heavy metrics pass.")
    parser.add_argument("--capture-iter", type=int, default=0,
                    help="Iteration at which scripts/running/capture_gpu_trace.py "
                         "freezes the workload for the GPU-Trace collection window "
                         "(external capture, see --capture-external). Presents a "
                         "bit-identical frozen frame from here so ngfx.exe can sample "
                         "HW counters over a stable steady state. 0 (default) = no "
                         "capture.")
    parser.add_argument("--capture-external", action="store_true", default=False,
                    help="Headless capture driven by the Nsight CLI launcher "
                         "(ngfx.exe), NOT by self-injection. In this mode we do "
                         "NOT load NGFX_Injection (ngfx.exe injects for us) and "
                         "do NOT trigger a capture; ngfx.exe starts/stops the "
                         "trace via --start-after-frames/--limit-to-frames and "
                         "AUTO-SAVES the .ngfx-gputrace to its --output-dir (no "
                         "HUD Save/Open needed). The trainer just presents a "
                         "stable frozen workload at --capture-iter for exactly "
                         "--capture-frames frames, then exits. Launch this via "
                         "scripts/running/capture_gpu_trace.py, which computes the frame "
                         "counts and invokes ngfx.exe. Implies --profile-run.")
    parser.add_argument("--capture-frames", type=int, default=240,
                    help="External-capture only: number of frozen, bit-identical "
                         "frames to Present() at --capture-iter before the "
                         "process exits. Must comfortably exceed the CLI's "
                         "start-after + limit-to window so the trace lands "
                         "entirely inside the frozen steady state.")
    parser.add_argument("--nsight-debug-names",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="Enable Vulkan debug layers so buffer/texture debug "
                         "labels (VK_EXT_debug_utils object names) reach the "
                         "driver and show up in Nsight's Object Browser / Memory "
                         "view as e.g. 'primitive.values' instead of "
                         "'Buffer_217'. sgl only enables VK_EXT_debug_utils when "
                         "debug layers are on, so without this every resource is "
                         "synthetic. Adds Vulkan validation overhead, so it is "
                         "OFF for normal runs and defaults ON under "
                         "--capture-external (a Graphics Capture is the whole "
                         "point of naming). Pass --no-nsight-debug-names to "
                         "force it off even during capture.")
    parser.add_argument("--flat-bucket-backward",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="Use the flat warp-per-bucket backward (live-bucket "
                         "compaction + indirect one-warp-per-bucket dispatch) "
                         "instead of the per-tile per-splat backward. "
                         "Gradient-equivalent. Pass --no-flat-bucket-backward "
                         "for the per-tile path (e.g. A/B timing).")
    _shared(parser, "backward_pixel_stride")
    _shared(parser, "stbn_pixel_fraction")
    _shared(parser, "stbn_warmup_iters")
    parser.add_argument("--stbn-max-fraction", type=float, default=0.0,
                    help="DIAGNOSTIC. Compile-time per-tile capacity for the "
                         "STBN backward's staged arrays (STBN_MAX_TILE_PIXELS). "
                         "0 (default) = auto, which is the only value worth "
                         "using: at an 8x8 tile the arrays it sizes total 1.5 KB "
                         "of the ~100 KB shared memory per SM, so shrinking them "
                         "changes nothing measurable. It exists to A/B that claim "
                         "- the speedup comes from the shorter pixel loop "
                         "(--stbn-pixel-fraction), not from the staging.")
    parser.add_argument("--stbn-source", type=str, default=StbnConfig.source,
                    choices=["nvidia", "white", "file"],
                    help="Where the spatiotemporal blue noise comes from. "
                         "'nvidia' (default): the pregenerated scalar STBN from "
                         "github.com/NVIDIA-RTX/STBN, fetched by "
                         "scripts/data/download_stbn.py (NON-COMMERCIAL license). "
                         "'white': independent random per-tile permutations — the "
                         "CONTROL that isolates what the blue-noise arrangement "
                         "buys, since it shares the exact per-tile count and every "
                         "downstream code path. 'file': rank your own scalar "
                         "volume from --stbn-file. Only used when "
                         "--stbn-pixel-fraction < 1.")
    parser.add_argument("--stbn-file", type=str, default=StbnConfig.file_path,
                    help="Path to a (slices, height, width) .npy scalar STBN "
                         "volume for --stbn-source file. Also overrides the "
                         "cached asset path for --stbn-source nvidia.")
    parser.add_argument("--stbn-select-mode", type=str, default=StbnConfig.select_mode,
                    choices=["global", "tile_rank"],
                    help="Which pixels a density selects. 'global' (default): "
                         "threshold the WHOLE 128x128 slice and keep its lowest "
                         "rho fraction — the point set NVIDIA's construction is "
                         "about, since their spatial energy function couples "
                         "samples across the entire slice. The 8x8 execution "
                         "tile then merely LOOKS UP that mask, and per-tile "
                         "counts vary (measured 16+/-1.4 at a nominal 16). "
                         "'tile_rank': rank within each 8x8 tile and keep "
                         "exactly k per tile — uniform per-tile work, but a "
                         "different sampling process whose per-tile threshold "
                         "lets pixels cluster across tile boundaries. Measured "
                         "at rho=0.25 it costs ~30%% of the blue mask's "
                         "low-frequency suppression AND makes the white-noise "
                         "control ~24%% bluer than true white, compressing the "
                         "blue-vs-white contrast from both ends. Kept as an "
                         "ablation. Incompatible with --stbn-warmup-iters > 0.")
    parser.add_argument("--stbn-seed", type=int, default=StbnConfig.seed,
                    help="Seed for --stbn-source white (ignored otherwise). Kept "
                         "separate from --seed so the mask can be re-rolled "
                         "without changing the camera draw order.")
    parser.add_argument("--stbn-independent",
                    action=argparse.BooleanOptionalAction, default=StbnConfig.independent,
                    help="Draw a fresh RANDOM mask slice each step instead of "
                         "advancing the temporal axis per camera visit. This "
                         "turns the spatiotemporal mask into an INDEPENDENT-"
                         "per-step mask (same slice pool, temporal ordering "
                         "broken) — the ablation baseline that isolates what the "
                         "temporal axis buys over independently drawn masks. The "
                         "slice RNG is seeded from --stbn-seed, kept separate "
                         "from the camera draw so it can't shift the sampling "
                         "order. Only used when --stbn-pixel-fraction < 1.")
    _shared(parser, "extinction_hardness")
    _shared(parser, "extinction_hardness_warmup")
    parser.add_argument("--final-test-psnr", action="store_true", default=False,
                    help="After the timed loop, compute ONLY the held-out "
                         "TEST-set PSNR (mean per-image sRGB, the paper metric "
                         "behind sadgs_vs_slang) and write test_psnr.json. "
                         "Lightweight: renders just the test cams, no full "
                         "per-camera PNG dump / PLY. Does not affect train_wall_s.")
    parser.add_argument("--compute-final-metrics", action="store_true", default=False)
    parser.add_argument("--save-ply", action="store_true",
                        help="write output.ply after training, even with --no-evaluation")
    parser.add_argument("--run-report", action="store_true", default=False,
                    help="After the timed loop, write run_report.json: PER-VIEW "
                         "held-out PSNR/SSIM (not just the mean), the timing "
                         "split (train_active_s = train_wall_s minus every "
                         "mid-loop observation), GPU allocations, image caches, process RAM, "
                         "the adapter/API the run bound, the resolved train/test "
                         "split, and the STBN asset's content hash. Additive: "
                         "test_psnr.json keeps its schema. Renders only the "
                         "held-out cams (no .ply, no full camera dump), and runs "
                         "after train_wall_s is frozen, so it cannot move a "
                         "training number.")
    parser.add_argument("--save-test-renders", action="store_true", default=False,
                    help="With --run-report, also write each held-out render to "
                         "<run>/test_renders/cam_NNNN.png. Lets LPIPS (and "
                         "contact sheets / error maps) be computed later, out of "
                         "process, without re-training. Metrics themselves are "
                         "always computed from the float render, never the PNG.")
    parser.add_argument("--render-fps-frames", type=int, default=0,
                    help="With --run-report, measure steady-state render "
                         "throughput over N serialized frames of one held-out "
                         "camera (0 = skip). Serialized (one wait_for_idle per "
                         "frame), so it reports per-frame latency rather than "
                         "pipelined throughput.")
    parser.add_argument("--eval-lpips", action="store_true", default=False,
                    help="Also compute VGG LPIPS on the held-out split — "
                         "in the --eval-every/--eval-iters sweeps and in "
                         "--run-report. Needs torch + the lpips package; when "
                         "they are missing the column is null (never silently "
                         "substituted) and the run is unaffected. Runs on the "
                         "CPU so it cannot contend with the GPU workload being "
                         "timed, ~50 ms/view, and it lands inside the "
                         "already-excluded observation window.")
    parser.add_argument("--write-ground-truths",
                    action=argparse.BooleanOptionalAction, default=False,
                    help="write a reference PNG for each camera at startup (default: off).")
    parser.add_argument("--stage-timing-every", type=int, default=0,
                    help="SAMPLED per-stage GPU timing: install the timestamp "
                         "timer on one iteration in N and write "
                         "stage_timing.{jsonl,json} (raw per-marker ms per "
                         "sample, plus per-window means). 0 (default) = off. "
                         "Reading timestamps needs a host drain the hot path "
                         "never does, so this PERTURBS the loop; the measured "
                         "cost is reported as sample_overhead_s so it can be "
                         "checked rather than assumed. 25 samples ~120 "
                         "iterations of a 3k run.")
    return parser
