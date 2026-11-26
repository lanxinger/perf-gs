# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Tile-binned forward rasterization of the Gaussian primitives."""

from __future__ import annotations

import numpy as np
import slangpy as spy

from trainer.backend.compute.algorithms import PrefixScan, RadixSort
from trainer.backend.compute.execution import CLUSTER_SIZE, TILE_HEIGHT, TILE_WIDTH
from trainer.backend.compute.gpu import CommandRecorder
from trainer.backend.compute.kernels import KernelLibrary
from trainer.backend.memory.buffers import (
    PrimitiveStorage,
    RasterResources,
    SphericalHarmonicsResources,
)
from trainer.common.lens import COS4_SRGB_EXPONENT
from trainer.config.config import IterationContextualVariables, RasterConfig
from trainer.data.dataset import Camera

_CAMERA_BYTES = 144   # CameraData: two 4x4 matrices + image size, padded to 16 bytes


class Rasterizer:
    """Projects, bins, depth-sorts and blends the splats of one view.

    Splats are depth-sorted once and their tile pairs emitted front to back, so the pair sort
    only orders tile indices. Keeps the uploaded camera and per-tile splat ids for the backward.
    """

    KERNELS = {
        "clear_textures": ("rasterize/tile_binned/clear.slang", "clearTextures"),
        "clear_tile_ranges": ("rasterize/tile_binned/tile_ranges.slang", "clearTileRanges"),
        "bin_count": ("rasterize/tile_binned/bin_count.slang", "binCount"),
        "gather_tile_counts": ("rasterize/tile_binned/depth_order.slang",
                               "gatherTileCountsByDepth"),
        "bin_emit": ("rasterize/tile_binned/bin_emit.slang", "binEmit"),
        "prepare_pair_args": ("densify/copy_active.slang", "prepareActiveCopyArgs"),
        "identify_tile_ranges": ("rasterize/tile_binned/tile_ranges.slang", "identifyTileRanges"),
        "bucket_count": ("rasterize/tile_binned/bucket_count.slang", "bucketCount"),
        "forward": ("rasterize/tile_binned/forward.slang", "forwardRasterize"),
    }
    SH_KERNELS = {"sh_forward": ("rasterize/tile_binned/sh_forward.slang", "shForward")}
    CULL_KERNELS = {
        "build_cluster_aabb": ("rasterize/cluster_cull.slang", "buildClusterAABB"),
        "cull_clusters_frustum": ("rasterize/cluster_cull.slang", "cullClustersFrustum"),
    }

    def __init__(self, device: spy.Device, commands: CommandRecorder,
                 kernels: KernelLibrary, *, primitive: PrimitiveStorage,
                 raster: RasterResources, sh: SphericalHarmonicsResources | None,
                 scan: PrefixScan, sort: RadixSort, config: RasterConfig,
                 width: int, height: int, num_tiles_x: int, num_tiles_y: int,
                 capacity: int) -> None:
        self._commands = commands
        self._k = kernels.load({**self.KERNELS,
                                **(self.SH_KERNELS if sh is not None else {}),
                                **(self.CULL_KERNELS if config.cluster_cull else {})})
        self._primitive = primitive
        self.raster = raster
        self._sh = sh
        self._scan = scan
        self._sort = sort
        self._depth_sort = RadixSort(commands, kernels, raster.depth_sort)
        self.config = config
        self.natural_vignetting: float = 0.0  # Existing False/True map to flat/full strength.
        self.width = width
        self.height = height
        self.num_tiles_x = num_tiles_x
        self.num_tiles_y = num_tiles_y
        self._capacity = capacity
        self.camera_buffer = device.create_buffer(
            size=_CAMERA_BYTES,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
            label="camera_data",
        )
        self.sorted_values = sort.resources.values

    @property
    def num_tiles(self) -> int:
        return self.num_tiles_x * self.num_tiles_y

    def upload_camera(self, camera: Camera) -> None:
        if camera.image_size != (self.width, self.height):
            raise ValueError(
                f"camera resolution {camera.image_size} does not match pipeline "
                f"resolution {(self.width, self.height)}")
        data = np.empty(36, dtype=np.float32)
        data[0:16] = camera.view_matrix.flatten()
        data[16:32] = camera.proj_matrix.flatten()
        data[32:34] = np.array([self.width, self.height], dtype=np.float32)
        data[34:36] = COS4_SRGB_EXPONENT * self.natural_vignetting, 0.0
        self.camera_buffer.copy_from_numpy(np.ascontiguousarray(data.view(np.uint8)))

    def record(self, camera: Camera, contextual: IterationContextualVariables) -> None:
        self.upload_camera(camera)
        commands, k, r = self._commands, self._k, self.raster
        primitive, config = self._primitive, self.config
        num_tiles, capacity = self.num_tiles, self._capacity

        with commands.marker_group("clear"):
            commands.dispatch(k.clear_textures, "clear_textures",
                thread_count=[self.width, self.height, 1],
                output_image=r.output_image,
                gaussian_count_per_pixel=r.gaussian_count_per_pixel,
                frame_dim=spy.uint2(self.width, self.height))
            commands.dispatch(k.clear_tile_ranges, "clear_tile_ranges",
                thread_count=[num_tiles, 1, 1],
                tile_start=r.tile_start.storage,
                tile_end=r.tile_end.storage,
                num_tiles=num_tiles)

        live = primitive.cached_count
        active_n = max(1, min(live, capacity)) if live is not None else capacity

        if config.cluster_cull:
            num_clusters = (active_n + CLUSTER_SIZE - 1) // CLUSTER_SIZE
            with commands.marker_group("cluster_cull"):
                commands.dispatch(k.build_cluster_aabb, "build_cluster_aabb",
                    thread_count=[num_clusters * CLUSTER_SIZE, 1, 1],
                    gaussians=primitive.values.storage,
                    num_gaussians_counter=primitive.active_count.storage,
                    cluster_aabb_min=r.cluster_aabb_min.storage,
                    cluster_aabb_max=r.cluster_aabb_max.storage,
                    max_splats=capacity,
                    num_clusters=num_clusters)
                commands.dispatch(k.cull_clusters_frustum, "cull_clusters_frustum",
                    thread_count=[num_clusters, 1, 1],
                    cluster_aabb_min=r.cluster_aabb_min.storage,
                    cluster_aabb_max=r.cluster_aabb_max.storage,
                    camera=self.camera_buffer,
                    cluster_visible=r.cluster_visible.storage,
                    num_clusters=num_clusters)

        with commands.marker_group("bin_count"):
            commands.clear_buffer(r.depth_count)
            commands.dispatch(k.bin_count, "bin_count",
                thread_count=[active_n, 1, 1],
                gaussians=primitive.values.storage,
                num_gaussians_counter=primitive.active_count.storage,
                camera=self.camera_buffer,
                projected_2d=r.projected_2d.storage,
                blend_data=r.blend_data.storage,
                tile_count_per_gaussian=r.tile_count_per_gaussian.storage,
                depth_keys=r.depth_sort.keys_lo.storage,
                depth_ids=r.depth_sort.values.storage,
                depth_count=r.depth_count.storage,
                cluster_visible=r.cluster_visible.storage,
                filter_3D=primitive.filter_3d.storage,
                mip_enabled=1 if config.mip_enabled else 0,
                kernel_size=config.kernel_size,
                z_near=config.z_near_cull,
                max_splats=capacity)

        # On Metal, reading visible_count once after bin_count lets sh_forward, depth_sort,
        # gather_tile_counts and scan_tile_offsets operate strictly on visible splats (~45% of live)
        # with zero additional sync inside depth_sort.
        vis_count = commands.read_uint(r.depth_count) if commands._emulate_indirect else None
        vis_n = max(1, min(vis_count, capacity)) if vis_count is not None else active_n

        if self._sh is not None and (vis_count is None or vis_count > 0):
            with commands.marker_group("sh_forward"):
                commands.dispatch(k.sh_forward, "sh_forward",
                    thread_count=[vis_n, 1, 1],
                    gaussians=primitive.values.storage,
                    visible_ids=r.depth_sort.values.storage,
                    visible_count=r.depth_count.storage,
                    camera=self.camera_buffer,
                    features_rest=self._sh.features_rest.storage,
                    blend_data=r.blend_data.storage,
                    active_sh_degree=contextual.sh_degree,
                    rest_floats=self._sh.rest_floats,
                    max_splats=capacity)

        with commands.marker_group("depth_sort"):
            _, _, depth_sorted_ids = self._depth_sort.sort(
                r.depth_count, key_bits=32, known_count=vis_count)
            commands.dispatch(k.gather_tile_counts, "gather_tile_counts",
                thread_count=[vis_n, 1, 1],
                depth_sorted_ids=depth_sorted_ids.storage,
                depth_count=r.depth_count.storage,
                tile_count_per_gaussian=r.tile_count_per_gaussian.storage,
                tile_count_by_depth=r.tile_count_by_depth.storage,
                max_splats=capacity)

        with commands.marker_group("scan_tile_offsets"):
            self._scan.scan(n=vis_n, in_buf=r.tile_count_by_depth,
                            out_buf=r.pair_offset_by_depth)

        pairs = self._sort.resources
        pair_count = self._scan.resources.total
        with commands.marker_group("bin_emit"):
            args = r.depth_sort.dispatch_args  # depth sort is finished for this frame
            # Prepare tile-sort dispatch args before bin_emit so Metal prefetches both
            # pair_count-derived indirect buffers in a single flush.
            self._sort.prepare_dispatch(pair_count)
            commands.dispatch(k.prepare_pair_args, "pair_dispatch_args", thread_count=[1, 1, 1],
                count_buf=pair_count.storage, dispatch_args=args.storage,
                record_words=1, max_splats=pairs.capacity)
            commands.dispatch_indirect(k.bin_emit, "bin_emit", args,
                blend_data=r.blend_data.storage,
                camera=self.camera_buffer,
                depth_sorted_ids=depth_sorted_ids.storage,
                depth_count=r.depth_count.storage,
                pair_offset_by_depth=r.pair_offset_by_depth.storage,
                pair_total=pair_count.storage,
                tile_keys=pairs.keys_lo.storage,
                values=pairs.values.storage,
                max_sort_pairs=pairs.capacity)

        with commands.marker_group("radix_sort"):
            # Pairs arrive front to back, so the stable sort only needs the tile index's bits.
            tile_bits = max(1, (num_tiles - 1).bit_length())
            sorted_tiles, _, sorted_values = self._sort.sort(
                pair_count, key_bits=tile_bits, args_prepared=True)

        with commands.marker_group("identify_tile_ranges"):
            commands.dispatch_indirect(k.identify_tile_ranges, "identify_tile_ranges", args,
                tile_keys=sorted_tiles.storage,
                tile_start=r.tile_start.storage,
                tile_end=r.tile_end.storage,
                n_buf=pair_count.storage,
                num_tiles=num_tiles)
        self.sorted_values = sorted_values

        with commands.marker_group("bucket_scan"):
            commands.dispatch(k.bucket_count, "bucket_count",
                thread_count=[num_tiles, 1, 1],
                tile_start=r.tile_start.storage,
                tile_end=r.tile_end.storage,
                bucket_count=r.bucket_count.storage,
                num_tiles=num_tiles)
            self._scan.scan(n=num_tiles, in_buf=r.bucket_count,
                            out_buf=r.bucket_offsets, total_buf=r.bucket_offset_total)

        with commands.marker_group("rasterize"):
            # One warp per tile, two pixels per lane; Metal uses 1 tile/workgroup for
            # cooperative 32-splat bucket preloads, Vulkan/D3D12 pack 2 tiles/workgroup.
            fwd_y = (self.num_tiles_y * (TILE_HEIGHT // 2)
                     if commands._emulate_indirect
                     else (self.num_tiles_y + 1) // 2 * TILE_HEIGHT)
            commands.dispatch(k.forward, "forward",
                thread_count=[self.num_tiles_x * TILE_WIDTH, fwd_y, 1],
                blend_data=r.blend_data.storage,
                tile_start=r.tile_start.storage,
                tile_end=r.tile_end.storage,
                sorted_gaussian_ids=sorted_values.storage,
                camera=self.camera_buffer,
                bucket_offsets=r.bucket_offsets.storage,
                raster_checkpoints=r.raster_checkpoints.storage,
                output_image=r.output_image,
                gaussian_count_per_pixel=r.gaussian_count_per_pixel,
                extinction_hardness=contextual.hardness_value)
