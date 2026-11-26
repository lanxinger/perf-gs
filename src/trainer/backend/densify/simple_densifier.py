# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Reference 3DGS densification: gradient-driven clone/split, then prune and compact."""

from __future__ import annotations

from trainer.backend.compute.algorithms import PrefixScan
from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.arena import ArenaCommitter
from trainer.backend.memory.buffers import (
    AdamResources,
    DensificationResources,
    EtaResources,
    PrimitiveStorage,
    SphericalHarmonicsResources,
)

_DISPATCH = "densify/densify_dispatch.slang"
_COMPACT = "densify/scatter_compact.slang"


class SimpleDensifier:
    """One wait-free tick: decide, split/clone into the tail, prune, compact in place."""

    KERNELS = {
        "decide_actions": ("densify/decide_actions_simple.slang", "decideActionsSimple"),
        "snapshot_count": (_DISPATCH, "snapshotCount"),
        "prepare_dispatch": (_DISPATCH, "prepareDensifyArgs"),
        "split_and_clone": ("densify/split_and_clone.slang", "splitAndClone"),
        "split_and_clone_sh": ("densify/split_and_clone.slang", "splitAndCloneSH"),
        "build_prune_mask": ("densify/build_prune_mask.slang", "buildPruneMask"),
        "scatter_compact_gaussian": (_COMPACT, "scatterCompactGaussian"),
        "scatter_compact_adam": (_COMPACT, "scatterCompactAdam"),
        "scatter_compact_features": (_COMPACT, "scatterCompactFeatures"),
        "scatter_compact_adam_rest": (_COMPACT, "scatterCompactAdamRestPlanes"),
        "scatter_compact_eta_only": (_COMPACT, "scatterCompactEtaOnly"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 primitive: PrimitiveStorage, adam: AdamResources,
                 resources: DensificationResources, eta: EtaResources | None,
                 sh: SphericalHarmonicsResources | None, scan: PrefixScan,
                 arena: ArenaCommitter, width: int, height: int,
                 max_primitives: int) -> None:
        self.commands = commands
        self.kernels = kernels.load(self.KERNELS)
        self.primitive = primitive
        self.adam = adam
        self.resources = resources
        self.eta = eta
        self.sh = sh
        self.scan = scan
        self.arena = arena
        self.width = width
        self.height = height
        self.max_primitives = max_primitives

    def run(
        self,
        cfg,
        grad_thresh: float = 0.0002,
        scale_thresh: float | None = None,
        prune_big_scale: float | None = None,
        allow_scale_prune: bool = True,
        carry_eta: bool = False,
        min_blend_weight: float = 0.0,
    ) -> dict:
        if carry_eta and self.eta is None:
            raise RuntimeError("carry_eta requires EtaResources")
        if scale_thresh is None:
            scale_thresh = cfg.t_big_scale * 0.1
        # The screen gradient sums an un-normalized per-pixel loss, so scale by W * H * channels.
        grad_thresh_internal = grad_thresh * (self.width * self.height * 3.0)
        big_scale = prune_big_scale if prune_big_scale is not None else cfg.t_big_scale
        primitive, adam, r = self.primitive, self.adam, self.resources
        k, commands, capacity = self.kernels, self.commands, self.max_primitives

        with commands.recording("densify_simple") as recording, \
                commands.marker_group("densify_simple"):
            commands.dispatch(k.decide_actions, "decide_actions_simple",
                thread_count=[capacity, 1, 1],
                gaussians=primitive.values.storage,
                num_gaussians_counter=primitive.active_count.storage,
                grad_mean_2d_mag_accum=r.grad_mean_2d_mag.storage,
                visibility_count_accum=r.visibility_count.storage,
                blend_weight_accum=r.blend_weight.storage,
                action=r.actions.storage,
                grad_thresh=grad_thresh_internal,
                scale_thresh=scale_thresh,
                t_opacity=cfg.t_opacity,
                t_big_scale=big_scale,
                allow_scale_prune=1 if allow_scale_prune else 0,
                min_blend_weight=float(min_blend_weight),
                max_splats=capacity)
            commands.dispatch(k.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1],
                src=primitive.active_count.storage,
                dst=r.pre_count.storage)
            commands.dispatch(k.prepare_dispatch, "densify_prepare_args",
                thread_count=[1, 1, 1],
                count_buf=r.pre_count.storage,
                dispatch_args=r.dispatch_args.storage,
                max_splats=capacity)
            split_kwargs = dict(
                gaussians=primitive.values.storage,
                adam_state=adam.state.storage,
                num_gaussians_counter=primitive.active_count.storage,
                action=r.actions.storage,
                pre_count_buf=r.pre_count.storage,
                max_splats=capacity)
            if self.sh is not None:
                split_kwargs.update(
                    features_rest=self.sh.features_rest.storage,
                    adam_rest=self.sh.adam_rest.storage,
                    rest_floats=self.sh.rest_floats)
            commands.dispatch_indirect(
                k.split_and_clone_sh if self.sh is not None else k.split_and_clone,
                "split_and_clone", r.dispatch_args, **split_kwargs)
            commands.dispatch(k.snapshot_count, "densify_snapshot",
                thread_count=[1, 1, 1],
                src=primitive.active_count.storage,
                dst=r.post_count.storage)
            commands.dispatch(k.build_prune_mask, "build_prune_mask",
                thread_count=[capacity, 1, 1],
                action=r.actions.storage,
                valid_mask=r.valid_mask.storage,
                pre_count_buf=r.pre_count.storage,
                post_count_buf=r.post_count.storage,
                max_splats=capacity)
            self._compact(carry_eta)
            # Eta keeps a dedicated scratch and swaps once the submit succeeded.
            if carry_eta:
                recording.after_submit(self.eta.swap)

        return {"pre_count": None, "post_split_count": None, "final_count": None}

    def compact_mask(self, mask) -> None:
        """Compact a host-selected survivor mask after the last growth tick."""
        self.resources.valid_mask.storage.copy_from_numpy(mask)
        with self.commands.recording("prune_compact"):
            self._compact(carry_eta=False)

    def _compact(self, carry_eta: bool) -> None:
        primitive, adam, r = self.primitive, self.adam, self.resources
        k, commands, capacity = self.kernels, self.commands, self.max_primitives
        live_count, scan_offsets = self.scan.resources.total, self.scan.resources.work
        self.scan.scan(capacity, r.valid_mask, total_buf=live_count)

        commands.dispatch(k.scatter_compact_gaussian, "scatter_compact_gaussian",
            thread_count=[capacity, 1, 1],
            gaussians_in=primitive.values.storage,
            valid_mask=r.valid_mask.storage,
            scan=scan_offsets.storage,
            gaussians_out=self.arena.storage,
            max_splats=capacity)
        self.arena.commit(primitive.values, live_count)
        commands.dispatch(k.scatter_compact_adam, "scatter_compact_adam",
            thread_count=[capacity, 1, 1],
            adam_state_in=adam.state.storage,
            valid_mask=r.valid_mask.storage,
            scan=scan_offsets.storage,
            adam_state_out=self.arena.storage,
            max_splats=capacity)
        self.arena.commit(adam.state, live_count)

        if carry_eta:
            commands.dispatch(k.scatter_compact_eta_only, "scatter_compact_eta_only",
                thread_count=[capacity, 1, 1],
                valid_mask=r.valid_mask.storage,
                scan=scan_offsets.storage,
                action=r.actions.storage,
                pre_count_buf=r.pre_count.storage,
                **self.eta.scatter_kwargs(),
                max_splats=capacity)

        if self.sh is not None:
            commands.dispatch(k.scatter_compact_features, "scatter_compact_features",
                thread_count=[capacity, 1, 1],
                features_rest_in=self.sh.features_rest.storage,
                valid_mask=r.valid_mask.storage,
                scan=scan_offsets.storage,
                features_rest_out=self.arena.storage,
                rest_floats=self.sh.rest_floats,
                max_splats=capacity)
            self.arena.commit(self.sh.features_rest, live_count)
            self.arena.reorder_planes(
                self.sh.adam_rest, k.scatter_compact_adam_rest,
                "scatter_compact_adam_rest", self.sh.moment_planes, live_count,
                valid_mask=r.valid_mask.storage, scan=scan_offsets.storage)

        commands.dispatch(k.snapshot_count, "densify_snapshot",
            thread_count=[1, 1, 1],
            src=live_count.storage,
            dst=primitive.active_count.storage)
        primitive.cached_count = None
