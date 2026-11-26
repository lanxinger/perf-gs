# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Per-splat statistics a densification window accumulates."""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import (
    DensificationResources,
    GradientResources,
    PrimitiveStorage,
    decode_grad_buffer,
)

FIXED_POINT_SCALE = 65536.0


class DensifyStatistics:
    """Screen-gradient magnitude, visibility count and blend weight per splat."""

    KERNELS = {
        "clear": ("densify/clear_tracking.slang", "clearTracking"),
        "accumulate": ("densify/accumulate_iter.slang", "accumulateIter"),
    }

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 primitive: PrimitiveStorage, resources: DensificationResources,
                 gradients: GradientResources, cluster_visible: spy.NDBuffer,
                 capacity: int, width: int, height: int) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self._primitive = primitive
        self.resources = resources
        self._gradients = gradients
        self._cluster_visible = cluster_visible
        self._capacity = capacity
        self._frame = spy.float2(width, height)

    def clear(self) -> None:
        """Start a new window (also required after any slot permutation)."""
        r = self.resources
        self._commands.dispatch(self._k.clear, "clear_tracking",
            thread_count=[self._capacity, 1, 1],
            grad_mean_2d_mag_accum=r.grad_mean_2d_mag.storage,
            visibility_count_accum=r.visibility_count.storage,
            grad_2d_x_iter=self._gradients.screen_x.storage,
            grad_2d_y_iter=self._gradients.screen_y.storage,
            blend_weight_accum=r.blend_weight.storage,
            max_splats=self._capacity)

    def accumulate(self) -> None:
        """Fold this step's screen gradient into the window and reset it."""
        r = self.resources
        live = self._primitive.cached_count
        active_n = max(1, min(live, self._capacity)) if live is not None else self._capacity
        self._commands.dispatch(self._k.accumulate, "accumulate_iter",
            thread_count=[active_n, 1, 1],
            grad_2d_x_iter=self._gradients.screen_x.storage,
            grad_2d_y_iter=self._gradients.screen_y.storage,
            grad_mean_2d_mag_accum=r.grad_mean_2d_mag.storage,
            visibility_count_accum=r.visibility_count.storage,
            num_gaussians_counter=self._primitive.active_count.storage,
            cluster_visible=self._cluster_visible.storage,
            frame_dim=self._frame,
            max_splats=self._capacity)

    def read(self) -> dict[str, np.ndarray]:
        r = self.resources
        visible = np.frombuffer(r.visibility_count.to_numpy().tobytes(), dtype=np.uint32)
        return {
            "grad_mean_2d_mag": decode_grad_buffer(r.grad_mean_2d_mag),
            "visibility_count": visible.astype(np.int64),
        }
