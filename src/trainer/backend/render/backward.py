# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Backward rasterization: image gradient -> per-splat parameter gradients."""

from __future__ import annotations

import slangpy as spy

from trainer.backend.compute.algorithms import PrefixScan
from trainer.backend.compute.execution import TILE_AREA, TILE_HEIGHT, TILE_WIDTH
from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import (
    BackwardResources,
    DensificationResources,
    GradientResources,
    PrimitiveStorage,
    RasterResources,
    SphericalHarmonicsResources,
    max_raster_buckets,
)
from trainer.backend.render.rasterizer import Rasterizer
from trainer.config.config import IterationContextualVariables

_LIVE_BUCKETS = "rasterize/tile_binned/live_buckets.slang"
_PER_SPLAT = "rasterize/tile_binned/backward_persplat.slang"


class RasterBackward:
    """Backpropagates dL/dC through the last frame the rasterizer recorded.

    Blend variants: per-pixel (reference), per-splat, and warp-per-bucket (default, fastest).
    """

    KERNELS = {
        "projection_backward": ("rasterize/tile_binned/projection_backward.slang",
                                "projectionBackward"),
        "sh_backward": ("rasterize/tile_binned/sh_backward.slang", "shBackward"),
    }
    PER_PIXEL_KERNELS = {"backward": ("rasterize/tile_binned/backward.slang", "backwardRasterize")}
    PER_SPLAT_KERNELS = {"backward_persplat": (_PER_SPLAT, "backwardRasterizePerSplat")}
    PER_BUCKET_KERNELS = {
        "tile_max_processed": (_LIVE_BUCKETS, "tileMaxProcessed"),
        "scatter_live_buckets": (_LIVE_BUCKETS, "scatterLiveBuckets"),
        "bucket_prepare_dispatch": (_LIVE_BUCKETS, "prepareBucketDispatchArgs"),
        "backward_persbucket": (_PER_SPLAT, "backwardRasterizePerBucket"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 rasterizer: Rasterizer, primitive: PrimitiveStorage,
                 raster: RasterResources, backward: BackwardResources,
                 gradients: GradientResources, image_gradient: spy.Texture,
                 tracking: DensificationResources,
                 sh: SphericalHarmonicsResources | None, scan: PrefixScan,
                 capacity: int, max_sort_pairs: int, per_splat: bool = True,
                 per_bucket: bool = True, stbn_compacted: bool = False) -> None:
        assert per_splat or not per_bucket, \
            "use_flat_bucket_backward=True requires use_persplat_backward=True"
        if rasterizer.config.mip_enabled and not per_splat:
            raise RuntimeError(
                "mip_enabled requires the per-splat backward: the per-pixel "
                "backward does not propagate opacity-compensation gradients.")
        variant = (self.PER_BUCKET_KERNELS if per_bucket
                   else self.PER_SPLAT_KERNELS if per_splat else self.PER_PIXEL_KERNELS)
        self._commands = commands
        self._k = kernels.load({**self.KERNELS, **variant})
        self._rasterizer = rasterizer
        self._primitive = primitive
        self._raster = raster
        self._backward = backward
        self._gradients = gradients
        self._image_gradient = image_gradient
        self._tracking = tracking
        self._sh = sh
        self._scan = scan
        self._capacity = capacity
        self._max_sort_pairs = max_sort_pairs
        self.per_splat = per_splat
        self.per_bucket = per_bucket
        self._stbn_compacted = stbn_compacted
        self._pixel_phase = 0   # rotates the visited pixel subset (BACKWARD_PIXEL_STRIDE > 1)

    def tracks_screen_gradient(self, contextual: IterationContextualVariables) -> bool:
        """Whether this step writes the per-step screen gradient (the per-pixel backward always does)."""
        variables = contextual.densification_variables
        return variables is None or variables.track_screen_grad or not self.per_splat

    def record(self, contextual: IterationContextualVariables, stbn_kwargs: dict) -> None:
        variables = contextual.densification_variables
        track_blend_weight = variables is not None and variables.track_blend_weight
        if track_blend_weight and not self.per_splat:
            raise RuntimeError("blend-weight tracking needs the per-splat backward")
        tracking = {
            "blend_weight_accum": self._tracking.blend_weight.storage,
            "track_blend_weight": int(track_blend_weight),
            "track_screen_grad": int(self.tracks_screen_gradient(contextual)),
        }
        # The loss already applied the mask; the compacted backward skips unsampled pixels.
        backward_stbn = stbn_kwargs if self._stbn_compacted else {}
        hardness = contextual.hardness_value
        with self._commands.marker_group("rasterize_backward"):
            if self.per_splat:
                self._record_blend(stbn_kwargs, backward_stbn, tracking, hardness)
                with self._commands.marker_group("projection_backward"):
                    self._record_projection()
            else:
                self._record_per_pixel()
        if self._sh is not None:
            with self._commands.marker_group("sh_backward"):
                self._record_sh(contextual.sh_degree)

    def _blend_kwargs(self) -> dict:
        r, grads = self._raster, self._gradients.params
        return dict(
            blend_data=r.blend_data.storage,
            tile_start=r.tile_start.storage,
            tile_end=r.tile_end.storage,
            sorted_gaussian_ids=self._rasterizer.sorted_values.storage,
            bucket_offsets=r.bucket_offsets.storage,
            raster_checkpoints=r.raster_checkpoints.storage,
            camera=self._rasterizer.camera_buffer,
            output_image=r.output_image,
            gaussian_count_per_pixel=r.gaussian_count_per_pixel,
            dL_dC_texture=self._image_gradient,
            **self._gradients.seed_kwargs(),
            grad_opacity=grads["opacity"].storage,
            grad_color_r=grads["color_r"].storage,
            grad_color_g=grads["color_g"].storage,
            grad_color_b=grads["color_b"].storage,
            grad_2d_x_iter=self._gradients.screen_x.storage,
            grad_2d_y_iter=self._gradients.screen_y.storage,
        )

    def _record_blend(self, stbn_kwargs: dict, backward_stbn: dict, tracking: dict,
                      hardness: float) -> None:
        commands, k = self._commands, self._k
        r, live = self._raster, self._backward
        num_tiles = self._rasterizer.num_tiles
        if self.per_bucket:
            with commands.marker_group("live_bucket_compaction"):
                commands.dispatch(k.tile_max_processed, "tile_max_processed",
                    thread_count=[num_tiles * TILE_AREA, 1, 1],
                    gaussian_count_per_pixel=r.gaussian_count_per_pixel,
                    live_bucket_count=live.live_bucket_count.storage,
                    frame_dim_px=spy.uint2(self._rasterizer.width, self._rasterizer.height),
                    num_tiles_x=self._rasterizer.num_tiles_x,
                    num_tiles=num_tiles,
                    **stbn_kwargs)
                self._scan.scan(n=num_tiles, in_buf=live.live_bucket_count,
                                out_buf=live.live_bucket_offsets,
                                total_buf=live.live_bucket_total)
                commands.dispatch(k.scatter_live_buckets, "scatter_live_buckets",
                    thread_count=[num_tiles, 1, 1],
                    live_bucket_count=live.live_bucket_count.storage,
                    live_bucket_offsets=live.live_bucket_offsets.storage,
                    live_buckets=live.live_buckets.storage,
                    num_tiles=num_tiles)
                commands.dispatch(k.bucket_prepare_dispatch, "bucket_prepare_dispatch",
                    thread_count=[1, 1, 1],
                    live_bucket_total=live.live_bucket_total.storage,
                    dispatch_args=live.dispatch_args.storage,
                    max_buckets=max_raster_buckets(self._max_sort_pairs, num_tiles))
            commands.dispatch_indirect(k.backward_persbucket, "backward_persbucket",
                args_buf=live.dispatch_args,
                live_buckets=live.live_buckets.storage,
                live_bucket_total=live.live_bucket_total.storage,
                **self._blend_kwargs(),
                **tracking,
                pixel_phase=self._pixel_phase,
                **backward_stbn,
                extinction_hardness=hardness)
        else:
            commands.dispatch(k.backward_persplat, "backward_persplat",
                thread_count=[num_tiles * TILE_AREA, 1, 1],
                **self._blend_kwargs(),
                **tracking,
                pixel_phase=self._pixel_phase,
                **backward_stbn,
                extinction_hardness=hardness)
        self._pixel_phase += 1

    def _active_threads(self) -> int:
        live = self._primitive.cached_count
        return max(1, min(live, self._capacity)) if live is not None else self._capacity

    def _record_projection(self) -> None:
        grads = self._gradients.params
        config = self._rasterizer.config
        self._commands.dispatch(self._k.projection_backward, "projection_backward",
            thread_count=[self._active_threads(), 1, 1],
            gaussians=self._primitive.values.storage,
            num_gaussians_counter=self._primitive.active_count.storage,
            camera=self._rasterizer.camera_buffer,
            **self._gradients.seed_kwargs(),
            grad_opacity=grads["opacity"].storage,
            grad_mean_x=grads["mean_x"].storage,
            grad_mean_y=grads["mean_y"].storage,
            grad_mean_z=grads["mean_z"].storage,
            grad_log_scale_x=grads["log_scale_x"].storage,
            grad_log_scale_y=grads["log_scale_y"].storage,
            grad_log_scale_z=grads["log_scale_z"].storage,
            grad_rot_w=grads["rot_w"].storage,
            grad_rot_x=grads["rot_x"].storage,
            grad_rot_y=grads["rot_y"].storage,
            grad_rot_z=grads["rot_z"].storage,
            filter_3D=self._primitive.filter_3d.storage,
            mip_enabled=1 if config.mip_enabled else 0,
            kernel_size=config.kernel_size,
            cluster_visible=self._raster.cluster_visible.storage,
            max_splats=self._capacity)

    def _record_per_pixel(self) -> None:
        r = self._raster
        self._commands.dispatch(self._k.backward, "backward",
            thread_count=[self._rasterizer.num_tiles_x * TILE_WIDTH,
                          self._rasterizer.num_tiles_y * TILE_HEIGHT, 1],
            gaussians=self._primitive.values.storage,
            projected_2d=r.projected_2d.storage,
            tile_start=r.tile_start.storage,
            tile_end=r.tile_end.storage,
            sorted_gaussian_ids=self._rasterizer.sorted_values.storage,
            camera=self._rasterizer.camera_buffer,
            output_image=r.output_image,
            gaussian_count_per_pixel=r.gaussian_count_per_pixel,
            dL_dC_texture=self._image_gradient,
            **self._gradients.param_kwargs(),
            grad_2d_x_iter=self._gradients.screen_x.storage,
            grad_2d_y_iter=self._gradients.screen_y.storage)

    def _record_sh(self, sh_degree: int) -> None:
        grads = self._gradients.params
        self._commands.dispatch(self._k.sh_backward, "sh_backward",
            thread_count=[self._active_threads(), 1, 1],
            gaussians=self._primitive.values.storage,
            num_gaussians_counter=self._primitive.active_count.storage,
            camera=self._rasterizer.camera_buffer,
            features_rest=self._sh.features_rest.storage,
            features_rest_grad=self._sh.features_rest_grad.storage,
            grad_color_r=grads["color_r"].storage,
            grad_color_g=grads["color_g"].storage,
            grad_color_b=grads["color_b"].storage,
            grad_mean_x=grads["mean_x"].storage,
            grad_mean_y=grads["mean_y"].storage,
            grad_mean_z=grads["mean_z"].storage,
            cluster_visible=self._raster.cluster_visible.storage,
            active_sh_degree=sh_degree,
            rest_floats=self._sh.rest_floats,
            max_splats=self._capacity)
