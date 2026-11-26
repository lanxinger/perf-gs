# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Structure-aware densification (SAD-GS, arXiv 2604.28016): per-view eta samples accumulate
per splat, and a densify tick splits splats whose views agree they under-resolve the image."""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.backend.compute.algorithms import PrefixScan
from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.config.config import StructureConfig
from trainer.backend.densify.opacity import OpacityReset
from trainer.backend.memory.arena import ArenaCommitter
from trainer.backend.memory.buffers import (
    AdamResources,
    DensificationResources,
    EtaResources,
    GradientResources,
    PrimitiveStorage,
    RasterResources,
    SphericalHarmonicsResources,
    StructureSplitResources,
)

_DISPATCH = "densify/densify_dispatch.slang"
_COMPACT = "densify/scatter_compact.slang"
_SCORE = "densify/score_structure_aware.slang"
_SPLIT = "densify/split_anisotropic.slang"
_RANK = "densify/rank_families.slang"
FIXED_POINT_SCALE = 65536.0


class StructureDensifier:
    KERNELS = {
        "score": (_SCORE, "scoreStructureAware"),
        "clear_scores": (_SCORE, "clearStructureScores"),
        "clear_uint": ("optimizers/grad_utils.slang", "clearUintBuffer"),
        "snapshot_count": (_DISPATCH, "snapshotCount"),
        "prepare_dispatch": (_DISPATCH, "prepareDensifyArgs"),
        "prepare_split": (_DISPATCH, "prepareStructureSplitArgs"),
        "decide_actions": ("densify/decide_actions_structure.slang", "decideActionsStructure"),
        "split": (_SPLIT, "splitAnisotropic"),
        "split_sh": (_SPLIT, "splitAnisotropicSH"),
        "build_prune_mask": ("densify/build_prune_mask.slang", "buildPruneMask"),
        "scatter_compact_gaussian": (_COMPACT, "scatterCompactGaussian"),
        "scatter_compact_adam": (_COMPACT, "scatterCompactAdam"),
        "scatter_compact_features": (_COMPACT, "scatterCompactFeatures"),
        "scatter_compact_adam_rest": (_COMPACT, "scatterCompactAdamRestPlanes"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 primitive: PrimitiveStorage, adam: AdamResources,
                 resources: DensificationResources, eta: EtaResources,
                 split: StructureSplitResources, gradients: GradientResources,
                 raster: RasterResources, sh: SphericalHarmonicsResources | None,
                 scan: PrefixScan, arena: ArenaCommitter, opacity: OpacityReset,
                 width: int, height: int, capacity: int) -> None:
        self._commands = commands
        self._kernel_library = kernels
        self._kernels = kernels.load(self.KERNELS)
        self._primitive = primitive
        self._adam_resources = adam
        self._densification_resources = resources
        self.eta_resources = eta
        self._split_resources = split
        self._gradient_resources = gradients
        self._raster_resources = raster
        self._sh_resources = sh
        self._scan = scan
        self._arena = arena
        self._opacity = opacity
        self._width = width
        self._height = height
        self._capacity = capacity

    def clear_scores(self) -> None:
        """Start a new accumulation window (after any slot permutation that dropped eta)."""
        self._commands.dispatch(self._kernels.clear_scores, "clear_structure_scores",
            thread_count=[self._capacity, 1, 1],
            **self.eta_resources.kwargs(),
            max_splats=self._capacity)

    def accumulate_scores(self, structure_tensor: spy.Texture, camera: spy.Buffer,
                          pass_idx: int, opacity_threshold: float = 0.05,
                          per_view_grad_thresh: float = 0.0) -> None:
        """Fold this view's eta sample into the accumulators.

        Must run after the backward and before the statistics accumulation clears the screen
        gradient. The opacity/gradient gates matter: ungated, densification explodes.
        """
        self._commands.dispatch(self._kernels.score, "score_structure_aware",
            thread_count=[self._capacity, 1, 1],
            gaussians=self._primitive.values.storage,
            num_gaussians_counter=self._primitive.active_count.storage,
            camera=camera,
            structure_tensor=structure_tensor,
            tile_count_per_gaussian=self._raster_resources.tile_count_per_gaussian.storage,
            grad_2d_x_iter=self._gradient_resources.screen_x.storage,
            grad_2d_y_iter=self._gradient_resources.screen_y.storage,
            **self.eta_resources.kwargs(),
            opacity_threshold=opacity_threshold,
            per_view_grad_thresh=float(per_view_grad_thresh),
            pass_idx=int(pass_idx) & 0xFFFF,
            max_splats=self._capacity)

    def densify(self, cfg: StructureConfig, min_blend_weight: float = 0.0) -> dict:
        """Split/clone the splats the window flagged, prune, compact, cap opacity.

        Read back the requested growth to report capacity pressure.
        """
        self._record_densify(cfg, min_blend_weight)
        self._commands.runtime.wait_for_submit(self._commands.last_submit_id)
        children = int(self._split_resources.children_sum.storage.to_numpy().view(np.uint32)[0])
        pre_count = int(self._densification_resources.pre_count.storage.to_numpy().view(np.uint32)[0])
        headroom = max(self._capacity - pre_count, 0)
        overbudget = children > headroom
        if overbudget:
            share = children / pre_count if pre_count > 0 else 0.0
            policy = ("ranked complete split families" if cfg.rank_split_families
                      else "overflow children DROPPED at the tail")
            print(f"Structure densify overbudget at {pre_count} splats — "
                  f"requested {children} new children ({share:.0%} of the "
                  f"population) but only {headroom} fit (max_splats="
                  f"{self._capacity}); {policy}.")
            if cfg.detailed_diagnostics:
                self._print_gate_breakdown(cfg, pre_count)
        return {"pre_count": None, "post_split_count": None, "final_count": None,
                "overbudget": overbudget}

    def _grad_threshold(self, cfg: StructureConfig) -> float:
        # Same convention scaling as the simple densifier.
        return cfg.grad_threshold * (self._width * self._height * 3.0)

    def _record_densify(self, cfg: StructureConfig, min_blend_weight: float) -> None:
        commands, k, capacity = self._commands, self._kernels, self._capacity
        primitive, adam, r, split = self._primitive, self._adam_resources, self._densification_resources, self._split_resources
        eta, sh, arena = self.eta_resources.components, self._sh_resources, self._arena
        selected_count, scan_offsets = self._scan.resources.total, self._scan.resources.work

        with commands.recording("densify_structure"), commands.marker_group("densify_structure"):
            commands.dispatch(k.clear_uint, "clear_children_sum",
                thread_count=[1, 1, 1], buffer=split.children_sum.storage, count=1)
            commands.dispatch(k.clear_uint, "clear_selected_count",
                thread_count=[1, 1, 1], buffer=selected_count.storage, count=1)
            commands.dispatch(k.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1], src=primitive.active_count.storage,
                dst=r.pre_count.storage)
            commands.dispatch(k.prepare_dispatch, "densify_prepare_args",
                thread_count=[1, 1, 1], count_buf=r.pre_count.storage,
                dispatch_args=r.dispatch_args.storage, max_splats=capacity)

            # Decide: action + per-axis split factors; selected parents go to the scan scratch.
            commands.dispatch_indirect(k.decide_actions, "decide_actions_structure",
                r.dispatch_args,
                densify_children_sum=split.children_sum.storage,
                gaussians=primitive.values.storage,
                num_gaussians_counter=primitive.active_count.storage,
                eta_n_total=self.eta_resources.n_total.storage,
                eta_n_high=eta["n_high"].storage,
                eta_n_low=eta["n_low"].storage,
                eta_max_x=eta["max_x"].storage,
                eta_max_y=eta["max_y"].storage,
                eta_max_z=eta["max_z"].storage,
                grad_mean_2d_mag_accum=r.grad_mean_2d_mag.storage,
                visibility_count_accum=r.visibility_count.storage,
                blend_weight_accum=r.blend_weight.storage,
                action=r.actions.storage,
                split_factor_x=split.split_factor_x.storage,
                split_factor_y=split.split_factor_y.storage,
                split_factor_z=split.split_factor_z.storage,
                tau_split=cfg.tau_split,
                tau_prune=cfg.tau_prune,
                grad_threshold_internal=self._grad_threshold(cfg),
                split_power=cfg.split_power,
                max_split_factor=int(cfg.max_split_factor),
                t_opacity=cfg.t_opacity,
                dense_scale=cfg.dense_scale,
                min_blend_weight=float(min_blend_weight),
                max_splats=capacity,
                selected_parents=r.valid_mask.storage,
                selected_count=selected_count.storage)

            if cfg.rank_split_families:
                # The depth sort has finished and clears its own histogram on next use.
                histogram = self._raster_resources.depth_sort.global_hist.storage
                commands.dispatch(k.clear_uint, "clear_family_histogram",
                    thread_count=[512, 1, 1], buffer=histogram, count=512)
                common = dict(histogram=histogram, pre_count=r.pre_count.storage, capacity=capacity)
                args = dict(**common, action=r.actions.storage,
                            factor_x=split.split_factor_x.storage, factor_y=split.split_factor_y.storage,
                            factor_z=split.split_factor_z.storage, eta_x=eta["max_x"].storage,
                            eta_y=eta["max_y"].storage, eta_z=eta["max_z"].storage,
                            views=r.visibility_count.storage, gradient=r.grad_mean_2d_mag.storage,
                            gradient_scale=self._grad_threshold(cfg) or 1.0)
                rank = self._kernel_library.kernel(_RANK, "rankFamilies")
                commands.dispatch_indirect(rank, "score_family_upgrades", r.dispatch_args, apply=0, **args)
                commands.dispatch(self._kernel_library.kernel(_RANK, "chooseCutoff"), "budget_family_upgrades",
                    thread_count=[1, 1, 1], **common)
                commands.dispatch_indirect(rank, "admit_family_upgrades", r.dispatch_args, apply=1, **args)

            # One cooperative workgroup per selected parent; children go to the tail.
            commands.dispatch(k.prepare_split, "structure_split_prepare_args",
                thread_count=[1, 1, 1],
                count_buf=selected_count.storage,
                dispatch_args=r.dispatch_args.storage,
                max_splats=capacity)
            split_kwargs = dict(
                gaussians=primitive.values.storage,
                adam_state=adam.state.storage,
                num_gaussians_counter=primitive.active_count.storage,
                action=r.actions.storage,
                split_factor_x=split.split_factor_x.storage,
                split_factor_y=split.split_factor_y.storage,
                split_factor_z=split.split_factor_z.storage,
                pre_count_buf=r.pre_count.storage,
                selected_parents=r.valid_mask.storage,
                selected_count=selected_count.storage,
                ks_scale_power=float(cfg.ks_scale_power),
                max_splats=capacity)
            if sh is not None:
                split_kwargs.update(features_rest=sh.features_rest.storage,
                                    adam_rest=sh.adam_rest.storage,
                                    rest_floats=sh.rest_floats)
            commands.dispatch_indirect(k.split_sh if sh is not None else k.split,
                "split_anisotropic", r.dispatch_args, **split_kwargs)
            commands.dispatch(k.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1], src=primitive.active_count.storage,
                dst=r.post_count.storage)

            # Prune: the selected-parent scratch becomes the keep mask, then compact in place.
            commands.dispatch(k.build_prune_mask, "build_prune_mask",
                thread_count=[capacity, 1, 1],
                action=r.actions.storage,
                valid_mask=r.valid_mask.storage,
                pre_count_buf=r.pre_count.storage,
                post_count_buf=r.post_count.storage,
                max_splats=capacity)
            self._scan.scan(capacity, r.valid_mask)
            live_count = selected_count
            commands.dispatch(k.scatter_compact_gaussian, "scatter_compact_gaussian",
                thread_count=[capacity, 1, 1],
                gaussians_in=primitive.values.storage,
                valid_mask=r.valid_mask.storage,
                scan=scan_offsets.storage,
                gaussians_out=arena.storage,
                max_splats=capacity)
            arena.commit(primitive.values, live_count)
            commands.dispatch(k.scatter_compact_adam, "scatter_compact_adam",
                thread_count=[capacity, 1, 1],
                adam_state_in=adam.state.storage,
                valid_mask=r.valid_mask.storage,
                scan=scan_offsets.storage,
                adam_state_out=arena.storage,
                max_splats=capacity)
            arena.commit(adam.state, live_count)
            if sh is not None:
                commands.dispatch(k.scatter_compact_features, "scatter_compact_features",
                    thread_count=[capacity, 1, 1],
                    features_rest_in=sh.features_rest.storage,
                    valid_mask=r.valid_mask.storage,
                    scan=scan_offsets.storage,
                    features_rest_out=arena.storage,
                    rest_floats=sh.rest_floats,
                    max_splats=capacity)
                arena.commit(sh.features_rest, live_count)
                arena.reorder_planes(
                    sh.adam_rest, k.scatter_compact_adam_rest, "scatter_compact_adam_rest",
                    sh.moment_planes, live_count,
                    valid_mask=r.valid_mask.storage, scan=scan_offsets.storage)
            commands.dispatch(k.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1], src=live_count.storage,
                dst=primitive.active_count.storage)
            primitive.cached_count = None

            # SADGS caps every survivor's opacity: inverse_sigmoid(min(opacity, cap)).
            if cfg.opacity_cap and cfg.opacity_cap > 0.0:
                self._opacity.cap(cfg.opacity_cap)

    def _print_gate_breakdown(self, cfg: StructureConfig, pre_count: int) -> None:
        """Full-capacity eta readbacks (opt-in via --densify-diag; can be hundreds of MB)."""
        def u32(buffer):
            return np.frombuffer(buffer.storage.to_numpy().tobytes(), dtype=np.uint32)[:pre_count]

        eta = self.eta_resources.components
        n_total = u32(self.eta_resources.n_total).astype(np.float64)
        n_high = u32(eta["n_high"]).astype(np.float64)
        eta_max = np.maximum(u32(eta["max_x"]), np.maximum(u32(eta["max_y"]), u32(eta["max_z"])))
        eta_max = eta_max.astype(np.float64) / FIXED_POINT_SCALE
        visible = u32(self._densification_resources.visibility_count).astype(np.float64)
        from trainer.backend.memory.buffers import decode_grad_buffer
        grad = decode_grad_buffer(self._densification_resources.grad_mean_2d_mag, pre_count)
        seen = n_total > 0
        if not seen.any():
            return
        high_ratio = (n_high / np.maximum(n_total, 1.0))[seen]
        avg_grad = np.where(visible > 0, grad / np.maximum(visible, 1.0), 0.0)[seen]
        eta_max = eta_max[seen]
        print(f"  eta-gate breakdown (over {int(seen.sum())} seen splats): "
              f"n_total p50={np.percentile(n_total[seen], 50):.0f} "
              f"p90={np.percentile(n_total[seen], 90):.0f}; "
              f"eta_max p50={np.percentile(eta_max, 50):.2f} "
              f"p90={np.percentile(eta_max, 90):.2f}; "
              f"high_ratio>{cfg.tau_split:.2f}: {np.mean(high_ratio > cfg.tau_split):.0%}; "
              f"grad>thresh: {np.mean(avg_grad >= self._grad_threshold(cfg)):.0%}. "
              f"(low n_total => multiview gate meaningless; high eta_max => "
              f"splats too big / structure field too hot.)")
