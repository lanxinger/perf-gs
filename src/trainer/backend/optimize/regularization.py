# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Opacity and scale regularization added onto the parameter gradients."""

from __future__ import annotations

import slangpy as spy

from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import GradientResources, PrimitiveStorage
from trainer.config.config import RegularizationConfig


class Regularizer:
    KERNELS = {"regularization": ("losses/regularization.slang", "addRegularization")}

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary, *,
                 primitive: PrimitiveStorage, gradients: GradientResources,
                 cluster_visible: spy.NDBuffer, config: RegularizationConfig,
                 capacity: int) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self._primitive = primitive
        self._gradients = gradients
        self._cluster_visible = cluster_visible
        self.config = config
        self._capacity = capacity

    def record(self) -> None:
        config = self.config
        if config.opacity_weight <= 0 and config.scale_weight <= 0:
            return
        live = self._primitive.cached_count
        active_n = max(1, min(live, self._capacity)) if live is not None else self._capacity
        grads = self._gradients.params
        self._commands.dispatch(self._k.regularization, "regularization",
            thread_count=[active_n, 1, 1],
            gaussians=self._primitive.values.storage,
            num_gaussians_counter=self._primitive.active_count.storage,
            grad_opacity=grads["opacity"].storage,
            grad_log_scale_x=grads["log_scale_x"].storage,
            grad_log_scale_y=grads["log_scale_y"].storage,
            grad_log_scale_z=grads["log_scale_z"].storage,
            cluster_visible=self._cluster_visible.storage,
            w_opacity=config.opacity_weight,
            w_scale=config.scale_weight)
