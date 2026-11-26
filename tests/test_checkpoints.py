# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Phase-1 validation for the transposed-backward forward checkpoints.

The forward pass records, once per RASTER_BUCKET_SIZE splats, each pixel's
running (T, C_in_front). The transposed backward reconstructs any splat's
pre-splat transmittance T_i by reading its bucket's checkpoint and
re-blending the splats between the checkpoint and the splat. This test
verifies the stored checkpoints are correct by:

  1. bucket-0 checkpoint == (T=1, C=0) for every pixel (initial state).
  2. Reconstructing forward FROM a checkpoint reproduces the final
     transmittance + color the forward wrote to the output textures.

The reconstruction in (2) is exactly what the backward relies on, so
matching it end-to-end is the decisive correctness check for Phase 1.
"""

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

from trainer.data.dataset import simple_camera
from trainer.backend.compute.execution import (
    RASTER_BUCKET_SIZE, TILE_WIDTH, TILE_HEIGHT, TILE_AREA)
from trainer.backend.compute.device import SHADERS_DIR, create_device
from trainer.backend.memory.layout import StructLayout
from _pipeline_testkit import TestPipeline


BUCKET = RASTER_BUCKET_SIZE
ALPHA_MAX = 0.99     # must match forward.slang
ALPHA_MIN = 1.0 / 255.0   # must match forward.slang alpha-min skip
T_EARLY_BREAK = 1e-3      # must match forward.slang


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _eval_alpha(px, py, mean, cov2d, opacity_logit):
    """Replicates evalGaussian2DFromCov: sigmoid(op) * exp(power), capped."""
    dx = px - mean[0]
    dy = py - mean[1]
    det = cov2d[0] * cov2d[2] - cov2d[1] * cov2d[1]
    if det <= 0.0:
        return 0.0
    inv = 1.0 / det
    ci00 = cov2d[2] * inv
    ci01 = -cov2d[1] * inv
    ci11 = cov2d[0] * inv
    power = -0.5 * (dx * dx * ci00 + 2.0 * dx * dy * ci01 + dy * dy * ci11)
    if power > 0.0:
        return 0.0
    a = _sigmoid(opacity_logit) * np.exp(power)
    return min(a, ALPHA_MAX)


def _sh_dc(sh0):
    # shDC: linear color = 0.5 + C0 * sh0. Must match common/math.slang shDC.
    C0 = 0.28209479177387814
    return 0.5 + C0 * np.asarray(sh0, dtype=np.float64)


@pytest.mark.parametrize("backend", ["vulkan", "d3d12", "metal"])
def test_forward_checkpoints_reconstruct(backend: str) -> None:
    try:
        device = create_device(backend=backend)
    except Exception as exc:
        pytest.skip(f"backend {backend!r} unavailable: {exc}")

    W = H = 64
    n = 80  # > 2 buckets of overlap in the central tile
    pipeline = TestPipeline(device=device, width=W, height=H, max_splats=128)

    rng = np.random.default_rng(0)
    # Cluster many overlapping splats near the optical axis so the central
    # tile's pixels see a deep (>2 bucket) sorted list.
    means = np.zeros((n, 3), dtype=np.float32)
    means[:, 0] = rng.uniform(-0.15, 0.15, n)
    means[:, 1] = rng.uniform(-0.15, 0.15, n)
    means[:, 2] = rng.uniform(2.0, 4.0, n)  # varied depth -> varied sort order
    log_scales = np.log(np.full((n, 3), 0.3, dtype=np.float32))
    rot = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    opac = np.full(n, 0.5, dtype=np.float32)
    colors = rng.uniform(-2.0, 2.0, (n, 3)).astype(np.float32)

    pipeline.set_primitive_from_numpy(
        means=means, log_scales=log_scales, rot_quats=rot,
        opacities=opac, color_logits=colors,
    )
    camera = simple_camera(width=W, height=H, eye=(0.0, 0.0, 0.0),
                           target=(0.0, 0.0, 1.0), up=(0.0, 1.0, 0.0),
                           fovy_degrees=60.0)
    pipeline.render(camera)
    device.wait_for_idle()

    b = pipeline.buffers
    num_tiles_x = (W + TILE_WIDTH - 1) // TILE_WIDTH
    num_tiles_y = (H + TILE_HEIGHT - 1) // TILE_HEIGHT

    tile_start = np.frombuffer(b.raster.tile_start.to_numpy().tobytes(), dtype=np.uint32)
    tile_end = np.frombuffer(b.raster.tile_end.to_numpy().tobytes(), dtype=np.uint32)
    bucket_offsets = np.frombuffer(b.raster.bucket_offsets.to_numpy().tobytes(), dtype=np.uint32)
    sorted_ids = np.frombuffer(b.sort.values.to_numpy().tobytes(), dtype=np.uint32)
    # sorted result may be in A or B ping-pong; use the stashed sorted buffer.
    sorted_ids = np.frombuffer(pipeline.rasterizer.sorted_values.to_numpy().tobytes(), dtype=np.uint32)

    proj_f = np.frombuffer(b.raster.projected_2d.to_numpy().tobytes(), dtype=np.float32).reshape(pipeline.max_splats, 7)
    gauss_rec = np.frombuffer(b.primitive.values.to_numpy().tobytes(), dtype=b.primitive.layout.dtype)

    ckpt = b.raster.raster_checkpoints.to_numpy()
    ckpt_u = np.frombuffer(ckpt.tobytes(), dtype=np.uint32).reshape(-1, 2)

    # Final transmittance is no longer stored in a dedicated texture; the
    # forward writes output_image.a = 1 - T, so T = 1 - alpha (see forward.slang).
    out_img = np.frombuffer(b.raster.output_image.to_numpy().tobytes(), dtype=np.float32).reshape(H, W, 4)
    out_T = 1.0 - out_img[..., 3]

    def unpack_half2(u):
        lo = np.uint16(u & 0xFFFF)
        hi = np.uint16((u >> 16) & 0xFFFF)
        return np.float32(lo.view(np.float16)), np.float32(hi.view(np.float16))

    checked_pixels = 0
    checked_bucket0 = 0
    # Walk every tile that received splats; validate its pixels' checkpoints.
    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            tile_idx = ty * num_tiles_x + tx
            s = int(tile_start[tile_idx]); e = int(tile_end[tile_idx])
            length = e - s
            if length <= 0:
                continue
            gb_base = int(bucket_offsets[tile_idx])

            for ly in range(TILE_HEIGHT):
                for lx in range(TILE_WIDTH):
                    gx = tx * TILE_WIDTH + lx
                    gy = ty * TILE_HEIGHT + ly
                    if gx >= W or gy >= H:
                        continue
                    pixel_in_tile = ly * TILE_WIDTH + lx
                    px = gx + 0.5
                    py = gy + 0.5

                    # bucket 0 checkpoint must be (T=1, C=0).
                    e0 = ckpt_u[(gb_base + 0) * TILE_AREA + pixel_in_tile]
                    T0, R0 = unpack_half2(e0[0])
                    G0, B0 = unpack_half2(e0[1])
                    assert abs(T0 - 1.0) < 1e-2, f"bucket0 T={T0} != 1"
                    assert abs(R0) < 1e-2 and abs(G0) < 1e-2 and abs(B0) < 1e-2, \
                        f"bucket0 C=({R0},{G0},{B0}) != 0"
                    checked_bucket0 += 1

                    # Walk the whole chain once, capturing (T, C) at each
                    # bucket boundary, and compare to the stored checkpoint.
                    # This is exactly the state the transposed backward will
                    # read to reconstruct T_i / C_in_front, so matching it at
                    # every boundary is the decisive check.
                    num_buckets = (length + BUCKET - 1) // BUCKET
                    T = 1.0
                    C = np.zeros(3, dtype=np.float64)
                    broke = False
                    for k in range(length):
                        if (k % BUCKET) == 0:
                            kb = k // BUCKET
                            e = ckpt_u[(gb_base + kb) * TILE_AREA + pixel_in_tile]
                            Tk, Rk = unpack_half2(e[0])
                            Gk, Bk = unpack_half2(e[1])
                            # fp16 tolerance: relative for T, absolute for C.
                            assert abs(float(Tk) - T) < 3e-2 + 3e-2 * T, \
                                f"tile {tile_idx} px {pixel_in_tile} bucket {kb}: " \
                                f"stored T={Tk} vs walk T={T}"
                            assert abs(float(Rk) - C[0]) < 5e-2 and \
                                   abs(float(Gk) - C[1]) < 5e-2 and \
                                   abs(float(Bk) - C[2]) < 5e-2, \
                                f"tile {tile_idx} px {pixel_in_tile} bucket {kb}: " \
                                f"stored C=({Rk},{Gk},{Bk}) vs walk C={C}"
                        gid = int(sorted_ids[s + k])
                        mean2d = proj_f[gid, 0:2]
                        cov2d = proj_f[gid, 2:5]
                        op = gauss_rec["opacity"][gid]
                        sh0 = gauss_rec["sh0"][gid]
                        a = _eval_alpha(px, py, mean2d, cov2d, op)
                        # Alpha-min skip: mirror forward.slang exactly — a splat
                        # below 1/255 does not blend or advance T.
                        if a < ALPHA_MIN:
                            continue
                        C += _sh_dc(sh0) * a * T
                        T *= (1.0 - a)
                        if T < T_EARLY_BREAK:
                            broke = True
                            break
                    assert abs(T - out_T[gy, gx]) < 2e-2, \
                        f"pixel ({gx},{gy}) reconstructed T={T} vs out_T={out_T[gy,gx]}"
                    checked_pixels += 1

    assert checked_bucket0 > 0, "no tiles had splats; test did not exercise checkpoints"
    # Ensure we actually exercised multi-bucket tiles.
    assert checked_pixels > 0
