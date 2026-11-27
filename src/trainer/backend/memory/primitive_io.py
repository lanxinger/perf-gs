# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Host <-> GPU transfer of primitive records in their reflected struct layout."""

from __future__ import annotations

import numpy as np

from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import PrimitiveStorage, SphericalHarmonicsResources
from trainer.common.sh import pack_features_rest, unpack_features_rest


def gaussian_records(dtype: np.dtype, *, means: np.ndarray, log_scales: np.ndarray,
                     rot_quats: np.ndarray, opacities: np.ndarray,
                     color_logits: np.ndarray) -> np.ndarray:
    """Gaussian3D records from per-field arrays; ``opacities`` are probabilities."""
    n = means.shape[0]
    assert means.shape == (n, 3), f"means shape {means.shape}"
    assert log_scales.shape == (n, 3), f"log_scales shape {log_scales.shape}"
    assert rot_quats.shape == (n, 4), f"rot_quats shape {rot_quats.shape}"
    assert opacities.shape == (n,), f"opacities shape {opacities.shape}"
    assert color_logits.shape == (n, 3), f"color_logits shape {color_logits.shape}"
    p = np.clip(opacities, 1e-5, 1.0 - 1e-5).astype(np.float32)
    records = np.zeros(n, dtype=dtype)
    records["mean"] = means
    records["log_scale"] = log_scales
    records["rot_quat"] = rot_quats
    records["opacity"] = np.log(p / (1.0 - p))
    records["sh0"] = color_logits
    return records


class PrimitiveIO:
    """Uploads and reads back the live primitives and their SH coefficients."""

    KERNELS = {"clear_uint": ("optimizers/grad_utils.slang", "clearUintBuffer")}

    def __init__(self, commands: CommandRecorder, kernels: KernelLibrary,
                 storage: PrimitiveStorage, capacity: int,
                 sh: SphericalHarmonicsResources | None) -> None:
        self._commands = commands
        self._k = kernels.load(self.KERNELS)
        self.storage = storage
        self.capacity = capacity
        self.sh = sh

    @property
    def dtype(self) -> np.dtype:
        return self.storage.layout.dtype

    def count(self) -> int:
        n = int(self.storage.active_count.to_numpy().view(np.uint32)[0])
        self.storage.cached_count = n
        return n

    def set_count(self, n: int) -> None:
        assert 0 <= n <= self.capacity, f"primitive count {n} out of range [0, {self.capacity}]"
        self.storage.active_count.copy_from_numpy(np.array([n], dtype=np.uint32))
        self.storage.cached_count = int(n)

    def write(self, records: np.ndarray) -> None:
        """Replace the live primitives with ``records`` (the layout dtype)."""
        assert records.dtype == self.dtype, f"records dtype {records.dtype} != {self.dtype}"
        n = records.shape[0]
        assert n <= self.capacity, f"{n} primitives exceed capacity {self.capacity}"
        self.storage.values.copy_from_numpy(
            np.ascontiguousarray(records).view(np.float32).reshape(-1))
        self.set_count(n)

    def read(self) -> np.ndarray:
        """The live primitives as layout-dtype records."""
        raw = self.storage.values.to_numpy()
        return np.frombuffer(raw, dtype=self.dtype)[:self.count()].copy()

    def clear_features_rest(self) -> None:
        count = self.capacity * self.sh.feat_uints
        self._commands.dispatch(self._k.clear_uint, "clear_features_rest",
            thread_count=[count, 1, 1],
            buffer=self.sh.features_rest.storage, count=count)

    def write_features_rest(self, features_rest: np.ndarray | None) -> None:
        """Upload ``(N, rest_floats)`` SH rest coefficients (coeff-major) for the first N slots."""
        if self.sh is None or features_rest is None or features_rest.size == 0:
            return
        assert features_rest.shape[1] == self.sh.rest_floats, (
            f"features_rest width {features_rest.shape[1]} != rest_floats {self.sh.rest_floats}")
        self.sh.features_rest.copy_from_numpy(
            pack_features_rest(np.ascontiguousarray(features_rest, dtype=np.float32)))

    def read_features_rest(self, n: int) -> np.ndarray:
        """``(n, rest_floats)`` SH rest coefficients of the first n slots."""
        packed = np.frombuffer(self.sh.features_rest.to_numpy(),
                               dtype=np.uint32)[:n * self.sh.feat_uints]
        return unpack_features_rest(packed, n, self.sh.rest_floats)
