# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""GPU resources grouped by owner, including backend-sized fixed-point gradient accumulators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy
from slangpy.core.native import NativeNDBuffer, NativeNDBufferDesc, Shape
from slangpy.reflection.lookup import resolve_element_type, resolve_program_layout

from trainer.backend.compute.execution import (
    CLUSTER_SIZE,
    RADIX_GLOBAL_HIST_SIZE,
    RADIX_PART_SIZE,
    RASTER_BUCKET_SIZE,
    SCAN_BLOCK_SIZE,
    TILE_AREA,
    TILE_HEIGHT,
    TILE_WIDTH,
)
from trainer.backend.compute.fixed_point import FIXED_POINT_INT64, fixed_point_format
from trainer.backend.memory.layout import StructLayout

# One fixed-point gradient accumulator per optimized scalar, in GaussianChannel order.
GRAD_SCALAR_COMPONENTS = (
    "mean_x", "mean_y", "mean_z",
    "log_scale_x", "log_scale_y", "log_scale_z",
    "rot_w", "rot_x", "rot_y", "rot_z",
    "opacity",
    "color_r", "color_g", "color_b",
)
GRAD_2D_SEED_COMPONENTS = ("mean2d_x", "mean2d_y", "conic_a", "conic_b", "conic_c", "amplitude")
ETA_ACCUM_COMPONENTS = ("n_high", "n_low", "max_x", "max_y", "max_z")

# Full-resolution indoor training can exceed 15.5M splat/tile pairs in one view.
DEFAULT_MAX_SORT_PAIRS = 16_776_960


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def max_raster_buckets(max_sort_pairs: int, num_tiles: int) -> int:
    """Upper bound on sum over tiles of ceil(len / RASTER_BUCKET_SIZE)."""
    return ceil_div(max_sort_pairs, RASTER_BUCKET_SIZE) + num_tiles


_DEFAULT_BUFFER_USAGE = (
    spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access)


def _make_labeled_ndbuffer(device, dtype, shape, label, usage):
    """An NDBuffer whose storage carries a debug label (Nsight memory view).

    Vulkan names a buffer only at creation, so this rebuilds NDBuffer around a labeled
    buffer, falling back to an unlabeled NDBuffer if slangpy's internals change.
    """
    try:
        program_layout = resolve_program_layout(device, dtype, None)
        element_type = resolve_element_type(program_layout, dtype)
        element_count = 1
        for dim in shape:
            element_count *= dim
        shp = Shape(tuple(shape))
        desc = NativeNDBufferDesc()
        desc.usage = usage
        desc.memory_type = spy.MemoryType.device_local
        desc.shape = shp
        desc.strides = shp.calc_contiguous_strides()
        desc.dtype = element_type
        desc.element_layout = element_type.buffer_layout.reflection
        size = element_count * element_type.buffer_layout.reflection.stride
        buffer = device.create_buffer(
            size=size, usage=usage,
            memory_type=spy.MemoryType.device_local, label=label)
        return NativeNDBuffer(device, desc, buffer)
    except Exception:
        return spy.NDBuffer(device=device, dtype=dtype, shape=shape, usage=usage)


@dataclass
class PrimitiveStorage:
    values: spy.NDBuffer
    active_count: spy.NDBuffer
    filter_3d: spy.NDBuffer
    layout: StructLayout
    cached_count: int | None = None


@dataclass
class AdamResources:
    state: spy.NDBuffer
    dispatch_args: spy.NDBuffer


@dataclass
class ScratchArena:
    """Raw buffer every compaction / Morton scatter stages through, one resource at a time.

    Sized for the largest whole-resource scatter; SH moment planes go in batches that fit.
    """
    buffer: spy.NDBuffer
    capacity_bytes: int
    copy_dispatch_args: spy.NDBuffer

    @property
    def storage(self):
        return self.buffer.storage


@dataclass
class ScanResources:
    work: spy.NDBuffer | None
    block_sums: spy.NDBuffer
    block_sums_l2: spy.NDBuffer
    total: spy.NDBuffer


@dataclass
class SortResources:
    """Ping-pong (key_lo, key_hi, value) pairs plus radix-sort scratch."""
    keys_lo: spy.NDBuffer
    keys_hi: spy.NDBuffer
    values: spy.NDBuffer
    keys_lo_alt: spy.NDBuffer
    keys_hi_alt: spy.NDBuffer
    values_alt: spy.NDBuffer
    global_hist: spy.NDBuffer
    pass_hist: spy.NDBuffer
    dispatch_args: spy.NDBuffer
    stats: spy.NDBuffer         # [peak requested pairs, sorts that dropped pairs]
    capacity: int


@dataclass
class RasterResources:
    """Per-frame forward intermediates, kept for the backward pass."""
    projected_2d: spy.NDBuffer
    blend_data: spy.NDBuffer
    tile_count_per_gaussian: spy.NDBuffer
    # Splats sorted front to back once per view; pairs are then emitted in that order.
    depth_sort: SortResources
    depth_count: spy.NDBuffer
    tile_count_by_depth: spy.NDBuffer
    pair_offset_by_depth: spy.NDBuffer
    tile_start: spy.NDBuffer
    tile_end: spy.NDBuffer
    bucket_count: spy.NDBuffer
    bucket_offsets: spy.NDBuffer
    bucket_offset_total: spy.NDBuffer
    raster_checkpoints: spy.NDBuffer
    output_image: spy.Texture     # .a = 1 - final transmittance
    gaussian_count_per_pixel: spy.Texture
    cluster_visible: spy.NDBuffer
    cluster_aabb_min: spy.NDBuffer | None = None
    cluster_aabb_max: spy.NDBuffer | None = None


@dataclass
class BackwardResources:
    """Live-bucket compaction for the warp-per-bucket backward."""
    live_bucket_count: spy.NDBuffer
    live_bucket_offsets: spy.NDBuffer
    live_bucket_total: spy.NDBuffer
    live_buckets: spy.NDBuffer
    dispatch_args: spy.NDBuffer


@dataclass
class GradientResources:
    params: dict[str, spy.NDBuffer]   # GRAD_SCALAR_COMPONENTS
    seeds: dict[str, spy.NDBuffer]    # GRAD_2D_SEED_COMPONENTS, blend -> projection
    screen_x: spy.NDBuffer            # this step's screen-space gradient
    screen_y: spy.NDBuffer

    def param_kwargs(self) -> dict:
        return {f"grad_{name}": buffer.storage for name, buffer in self.params.items()}

    def seed_kwargs(self) -> dict:
        return {f"grad_{name}": buffer.storage for name, buffer in self.seeds.items()}


@dataclass
class LossResources:
    target_image: spy.Texture         # rgba8_unorm sRGB
    image_gradient: spy.Texture       # dL/dC
    ssim_m1: spy.Texture
    ssim_m2: spy.Texture
    ssim_m3: spy.Texture


@dataclass
class DensificationResources:
    grad_mean_2d_mag: spy.NDBuffer
    visibility_count: spy.NDBuffer
    blend_weight: spy.NDBuffer
    scratch_arena: ScratchArena
    actions: spy.NDBuffer
    valid_mask: spy.NDBuffer
    pre_count: spy.NDBuffer
    post_count: spy.NDBuffer
    dispatch_args: spy.NDBuffer


@dataclass
class EtaResources:
    """Structure-aware multiview accumulators, double-buffered for slot permutations."""
    components: dict[str, spy.NDBuffer]
    n_total: spy.NDBuffer
    scratch_components: dict[str, spy.NDBuffer]
    scratch_n_total: spy.NDBuffer

    def kwargs(self) -> dict:
        kwargs = {f"eta_{name}": buffer.storage for name, buffer in self.components.items()}
        kwargs["eta_n_total"] = self.n_total.storage
        return kwargs

    def scatter_kwargs(self) -> dict:
        primary = {"n_total": self.n_total, **self.components}
        scratch = {"n_total": self.scratch_n_total, **self.scratch_components}
        kwargs = {f"eta_{name}_in": buffer.storage for name, buffer in primary.items()}
        kwargs.update(
            {f"eta_{name}_out": buffer.storage for name, buffer in scratch.items()})
        return kwargs

    def swap(self) -> None:
        self.n_total, self.scratch_n_total = self.scratch_n_total, self.n_total
        self.components, self.scratch_components = \
            self.scratch_components, self.components


@dataclass
class StructureSplitResources:
    split_factor_x: spy.NDBuffer
    split_factor_y: spy.NDBuffer
    split_factor_z: spy.NDBuffer
    children_sum: spy.NDBuffer        # requested tail growth, read back for the overflow warning


@dataclass
class MipResources:
    dispatch_args: spy.NDBuffer       # compute + tail-clear records
    previous_count: spy.NDBuffer      # [live count, clear end]


@dataclass
class SphericalHarmonicsResources:
    """SH rest coefficients (bands 1..degree), stored apart from Gaussian3D.

    Features are fp16 pairs (``feat_uints`` words per splat). Adam moments are
    ``moment_planes`` SoA planes of max_splats words each.
    """
    rest_floats: int
    feat_uints: int
    features_rest: spy.NDBuffer
    features_rest_grad: spy.NDBuffer
    adam_rest: spy.NDBuffer
    moment_planes: int


@dataclass
class GaussianBuffers:
    """Every GPU resource of one pipeline. Training-only groups are None on the viewer."""
    max_splats: int
    max_sort_pairs: int
    width: int
    height: int
    num_tiles_x: int
    num_tiles_y: int
    primitive: PrimitiveStorage
    scan: ScanResources
    sort: SortResources
    raster: RasterResources
    sh: SphericalHarmonicsResources | None = None
    backward: BackwardResources | None = None
    gradients: GradientResources | None = None
    loss: LossResources | None = None
    adam: AdamResources | None = None
    densification: DensificationResources | None = None
    mip: MipResources | None = None
    eta: EtaResources | None = None
    structure: StructureSplitResources | None = None

    @property
    def num_tiles(self) -> int:
        return self.num_tiles_x * self.num_tiles_y

    @property
    def sh_enabled(self) -> bool:
        return self.sh is not None


FIXED_POINT_SCALE = 65536.0


def is_float_grad_buffer(buf: spy.NDBuffer) -> bool:
    """True when ``buf`` stores fp32 gradients (Metal) rather than fixed-point integers."""
    dtype = getattr(buf, "dtype", None)
    if dtype is not None:
        name = getattr(dtype, "name", None) or str(dtype)
        if name == "float":
            return True
        if name in ("int", "int32_t", "int64_t"):
            return False
    n = 1
    for dim in buf.shape:
        n *= int(dim)
    return n > 0 and buf.storage.size == n * 4


def _fixed_format_for_buffer(buf: spy.NDBuffer, fixed_point=None):
    if fixed_point is not None:
        return fixed_point
    dtype = getattr(buf, "dtype", None)
    name = (getattr(dtype, "name", None) or str(dtype)) if dtype is not None else ""
    if name in ("int", "int32_t"):
        from trainer.backend.compute.fixed_point import FIXED_POINT_INT32
        return FIXED_POINT_INT32
    return FIXED_POINT_INT64


def decode_grad_buffer(buf: spy.NDBuffer, count: int | None = None, *, fixed_point=None) -> np.ndarray:
    """Read a gradient/accumulator buffer as float64 regardless of backend storage (fp32, int32, int64)."""
    raw = buf.storage.to_numpy().tobytes()
    if is_float_grad_buffer(buf):
        vals = np.frombuffer(raw, dtype=np.float32)
        if count is not None:
            vals = vals[:count]
        return vals.astype(np.float64)
    fmt = _fixed_format_for_buffer(buf, fixed_point)
    ints = np.frombuffer(raw, dtype=fmt.dtype)
    if count is not None:
        ints = ints[:count]
    return ints.astype(np.float64) / fmt.scale


def encode_grad_array(buf: spy.NDBuffer, values: np.ndarray, *, fixed_point=None) -> np.ndarray:
    """Encode float gradient values into ``buf``'s backend storage dtype (fp32, int32, or int64)."""
    arr = np.asarray(values, dtype=np.float64)
    if is_float_grad_buffer(buf):
        return arr.astype(np.float32)
    fmt = _fixed_format_for_buffer(buf, fixed_point)
    return np.round(arr * fmt.scale).astype(fmt.dtype)


@dataclass
class _ShaderTypes:
    gaussian: object
    projected: object
    blend: object
    adam: object
    sh_moments: object
    gaussian_layout: StructLayout
    grad_dtype: str = "int64_t"


def _reflect_types(module: spy.Module) -> _ShaderTypes:
    def need(name: str) -> object:
        slang_type = module.layout.find_type_by_name(name)
        if slang_type is None:
            raise RuntimeError(f"shader module does not expose type '{name}'")
        return slang_type

    gaussian = need("Gaussian3D")
    session = getattr(module, "session", None)
    defines = getattr(getattr(getattr(session, "desc", None), "compiler_options", None), "defines", None) or {}
    device_api = getattr(getattr(getattr(module, "device", None), "info", None), "api_name", None)
    if "FIXED_POINT_INT32" not in defines and ("TRAINER_METAL_BACKEND" in defines or device_api == "Metal"):
        grad_dtype = "float"
    else:
        grad_dtype = fixed_point_format(module).buffer_dtype
    return _ShaderTypes(
        gaussian=gaussian,
        projected=need("Projected2DGaussian"),
        blend=need("BlendData"),
        adam=need("AdamState"),
        sh_moments=need("ShRestMoments"),
        gaussian_layout=StructLayout.of(gaussian),
        grad_dtype=grad_dtype,
    )


@dataclass
class _Alloc:
    """Buffer/texture factory bound to the device and frame size."""
    device: spy.Device
    width: int
    height: int
    fixed_dtype: str = FIXED_POINT_INT64.buffer_dtype

    def typed_buffer(self, label: str, dtype, *shape) -> spy.NDBuffer:
        return _make_labeled_ndbuffer(
            self.device, dtype, shape, label, _DEFAULT_BUFFER_USAGE)

    def uint_buffer(self, label: str, *shape) -> spy.NDBuffer:
        return self.typed_buffer(label, "uint", *shape)

    def fixed_buffer(self, label: str, *shape) -> spy.NDBuffer:
        return self.typed_buffer(label, self.fixed_dtype, *shape)

    def indirect_buffer(self, label: str, records: int = 1) -> spy.NDBuffer:
        return _make_labeled_ndbuffer(
            self.device, "uint", (3 * records,), label,
            _DEFAULT_BUFFER_USAGE | spy.BufferUsage.indirect_argument)

    def tex(self, fmt: spy.Format, label: str,
            usage: spy.TextureUsage | None = None) -> spy.Texture:
        return self.device.create_texture(
            type=spy.TextureType.texture_2d,
            format=fmt,
            width=self.width,
            height=self.height,
            usage=usage if usage is not None else (
                spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access),
            label=label,
        )


def scratch_resource_bytes(types: _ShaderTypes, max_splats: int,
                           sh_rest_floats: int) -> int:
    """Largest whole-resource scatter: gaussians, main Adam, or SH features."""
    g_stride = int(types.gaussian.buffer_layout.reflection.stride)
    a_stride = int(types.adam.buffer_layout.reflection.stride)
    candidates = [max_splats * g_stride, max_splats * a_stride]
    if sh_rest_floats > 0:
        feat_uints = (sh_rest_floats + 1) // 2
        candidates.append(max_splats * feat_uints * 4)
    return max(candidates)


def _make_scratch_arena(allocator: _Alloc, types: _ShaderTypes,
                        max_splats: int, sh_rest_floats: int) -> ScratchArena:
    nbytes = scratch_resource_bytes(types, max_splats, sh_rest_floats)
    n_uints = (nbytes + 3) // 4
    buf = allocator.uint_buffer("densification.scratch_arena", n_uints)
    return ScratchArena(
        buffer=buf, capacity_bytes=n_uints * 4,
        copy_dispatch_args=allocator.indirect_buffer("densification.arena_copy_args"))


def _allocate_sort(allocator: _Alloc, capacity: int, name: str = "sort",
                   key_words: int = 2) -> SortResources:
    max_blocks = ceil_div(capacity, RADIX_PART_SIZE)
    # 32-bit keys never touch keys_hi, so it shrinks to a placeholder.
    high_words = capacity if key_words > 1 else 1
    return SortResources(
        keys_lo=allocator.uint_buffer(f"{name}_keys_lo_a", capacity),
        keys_hi=allocator.uint_buffer(f"{name}_keys_hi_a", high_words),
        values=allocator.uint_buffer(f"{name}_values_a", capacity),
        keys_lo_alt=allocator.uint_buffer(f"{name}_keys_lo_b", capacity),
        keys_hi_alt=allocator.uint_buffer(f"{name}_keys_hi_b", high_words),
        values_alt=allocator.uint_buffer(f"{name}_values_b", capacity),
        global_hist=allocator.uint_buffer(f"{name}_global_hist", RADIX_GLOBAL_HIST_SIZE),
        pass_hist=allocator.uint_buffer(f"{name}_pass_hist", 256 * max_blocks),
        dispatch_args=allocator.indirect_buffer(f"{name}_dispatch_args"),
        stats=allocator.uint_buffer(f"{name}_stats", 2),
        capacity=capacity,
    )


def _allocate_depth_order(allocator: _Alloc, max_splats: int) -> dict:
    """Front-to-back splat order and the depth-ordered pair offsets bin_emit writes at."""
    return {
        "depth_sort": _allocate_sort(allocator, max_splats, "depth_sort", key_words=1),
        "depth_count": allocator.uint_buffer("depth_count", 1),
        "tile_count_by_depth": allocator.uint_buffer("tile_count_by_depth", max_splats),
        "pair_offset_by_depth": allocator.uint_buffer("pair_offset_by_depth", max_splats),
    }


def _allocate_scan(allocator: _Alloc, max_splats: int, with_work: bool) -> ScanResources:
    num_l1_blocks = max(1, ceil_div(max_splats, SCAN_BLOCK_SIZE))
    num_l2_blocks = max(1, ceil_div(num_l1_blocks, SCAN_BLOCK_SIZE))
    return ScanResources(
        work=allocator.uint_buffer("scan.work", max_splats) if with_work else None,
        block_sums=allocator.uint_buffer("scan.block_sums", num_l1_blocks),
        block_sums_l2=allocator.uint_buffer("scan.block_sums_l2", num_l2_blocks),
        total=allocator.uint_buffer("scan.total", 1),
    )


def _all_clusters_visible(allocator: _Alloc, max_splats: int) -> spy.NDBuffer:
    """bin_count reads it every frame; all-ones until a cluster cull overwrites it."""
    max_clusters = ceil_div(max_splats, CLUSTER_SIZE)
    visible = allocator.uint_buffer("cluster_visible", max_clusters)
    visible.copy_from_numpy(np.ones(max_clusters, dtype=np.uint32))
    return visible


def _tile_grid(width: int, height: int) -> tuple[int, int]:
    return ceil_div(width, TILE_WIDTH), ceil_div(height, TILE_HEIGHT)


def allocate(
    device: spy.Device,
    shared_data_module: spy.Module,
    max_splats: int,
    width: int,
    height: int,
    max_sort_pairs: int = DEFAULT_MAX_SORT_PAIRS,
    sh_rest_floats: int = 0,
    structure_aware: bool = False,
    _allocator=None,
) -> GaussianBuffers:
    """Allocate everything forward, backward, Adam and densification use."""
    types = _reflect_types(shared_data_module)
    fixed_dtype = fixed_point_format(shared_data_module or device).buffer_dtype
    allocator = _allocator if _allocator is not None else _Alloc(device, width, height, fixed_dtype)
    num_tiles_x, num_tiles_y = _tile_grid(width, height)
    num_tiles = num_tiles_x * num_tiles_y
    max_buckets = max_raster_buckets(max_sort_pairs, num_tiles)

    primitive_values = allocator.typed_buffer("primitive.values", types.gaussian, max_splats)
    active_count = allocator.uint_buffer("primitive.active_count", 1)
    projected_2d = allocator.typed_buffer("projected_2d", types.projected, max_splats)
    blend_data = allocator.typed_buffer("blend_data", types.blend, max_splats)
    tile_count = allocator.uint_buffer("tile_count_per_gaussian", max_splats)
    depth_order = _allocate_depth_order(allocator, max_splats)
    sort = _allocate_sort(allocator, max_sort_pairs)
    tile_start = allocator.uint_buffer("tile_start", num_tiles)
    tile_end = allocator.uint_buffer("tile_end", num_tiles)
    bucket_count = allocator.uint_buffer("bucket_count", num_tiles)
    bucket_offsets = allocator.uint_buffer("bucket_offsets", num_tiles)
    bucket_offset_total = allocator.uint_buffer("bucket_offset_total", 1)
    raster_checkpoints = allocator.typed_buffer(
        "raster_checkpoints", "uint2", max_buckets * TILE_AREA)
    backward = BackwardResources(
        live_bucket_count=allocator.uint_buffer("live_bucket_count", num_tiles),
        live_bucket_offsets=allocator.uint_buffer("live_bucket_offsets", num_tiles),
        live_bucket_total=allocator.uint_buffer("live_bucket_total", 1),
        live_buckets=allocator.typed_buffer("live_buckets", "uint2", max_buckets),
        dispatch_args=allocator.indirect_buffer("bucket_dispatch_args"),
    )
    output_image = allocator.tex(spy.Format.rgba32_float, "output_image")
    gaussian_count_per_pixel = allocator.tex(spy.Format.r32_uint, "gaussian_count_per_pixel")

    loss = LossResources(
        target_image=allocator.tex(spy.Format.rgba8_unorm, "target_image"),
        image_gradient=allocator.tex(spy.Format.rgba32_float, "dL_dC"),
        ssim_m1=allocator.tex(spy.Format.rgba32_float, "ssim_m1"),
        ssim_m2=allocator.tex(spy.Format.rgba32_float, "ssim_m2"),
        ssim_m3=allocator.tex(spy.Format.rgba32_float, "ssim_m3"),
    )
    grad_dtype = getattr(types, "grad_dtype", fixed_dtype)
    grad_params = {name: allocator.typed_buffer(f"grad_buffers[{name}]", grad_dtype, max_splats)
                   for name in GRAD_SCALAR_COMPONENTS}
    grad_seeds = {name: allocator.typed_buffer(f"grad_2d_seed_buffers[{name}]", grad_dtype, max_splats)
                  for name in GRAD_2D_SEED_COMPONENTS}
    adam_state = allocator.typed_buffer("adam.state", types.adam, max_splats)

    grad_mean_2d_mag = allocator.typed_buffer("densification.grad_mean_2d_mag", grad_dtype, max_splats)
    visibility_count = allocator.uint_buffer("densification.visibility_count", max_splats)
    blend_weight = allocator.fixed_buffer("densification.blend_weight", max_splats)
    gradients = GradientResources(
        params=grad_params, seeds=grad_seeds,
        screen_x=allocator.typed_buffer("grad_2d_x_iter", grad_dtype, max_splats),
        screen_y=allocator.typed_buffer("grad_2d_y_iter", grad_dtype, max_splats),
    )
    filter_3d = allocator.typed_buffer("primitive.filter_3d", "float", max_splats)
    primitive = PrimitiveStorage(primitive_values, active_count, filter_3d, types.gaussian_layout)

    scratch_arena = _make_scratch_arena(allocator, types, max_splats, sh_rest_floats)
    actions = allocator.uint_buffer("densification.actions", max_splats)
    valid_mask = allocator.uint_buffer("densification.valid_mask", max_splats)
    scan = _allocate_scan(allocator, max_splats, with_work=True)
    densification = DensificationResources(
        grad_mean_2d_mag=grad_mean_2d_mag,
        visibility_count=visibility_count,
        blend_weight=blend_weight,
        scratch_arena=scratch_arena,
        actions=actions,
        valid_mask=valid_mask,
        pre_count=allocator.uint_buffer("densification.pre_count", 1),
        post_count=allocator.uint_buffer("densification.post_count", 1),
        dispatch_args=allocator.indirect_buffer("densification.dispatch_args"),
    )
    adam = AdamResources(adam_state, allocator.indirect_buffer("adam_dispatch_args"))
    mip = MipResources(
        dispatch_args=allocator.indirect_buffer("mip_filter_dispatch_args", records=2),
        previous_count=allocator.uint_buffer("mip_filter_previous_count", 2),
    )
    # The first recomputation clears the whole dead tail; later ones only the slots that died.
    mip.previous_count.copy_from_numpy(np.array([max_splats, max_splats], dtype=np.uint32))

    eta = structure = None
    if structure_aware:
        eta = EtaResources(
            components={name: allocator.uint_buffer(f"eta.components[{name}]", max_splats)
                        for name in ETA_ACCUM_COMPONENTS},
            n_total=allocator.uint_buffer("eta.n_total", max_splats),
            scratch_components={
                name: allocator.uint_buffer(f"eta.scratch_components[{name}]", max_splats)
                for name in ETA_ACCUM_COMPONENTS},
            scratch_n_total=allocator.uint_buffer("eta.scratch_n_total", max_splats),
        )
        structure = StructureSplitResources(
            split_factor_x=allocator.uint_buffer("split_factor_x", max_splats),
            split_factor_y=allocator.uint_buffer("split_factor_y", max_splats),
            split_factor_z=allocator.uint_buffer("split_factor_z", max_splats),
            children_sum=allocator.uint_buffer("densify_children_sum", 1),
        )

    sh = None
    if sh_rest_floats > 0:
        feat_uints = (sh_rest_floats + 1) // 2
        moment_planes = sh_rest_floats * int(types.sh_moments.buffer_layout.reflection.stride) // 4
        sh = SphericalHarmonicsResources(
            rest_floats=sh_rest_floats,
            feat_uints=feat_uints,
            features_rest=allocator.uint_buffer("sh.features_rest", max_splats * feat_uints),
            features_rest_grad=allocator.uint_buffer("sh.features_rest_grad", max_splats * feat_uints),
            # fp32 (or bf16-pair) moments: fp16 underflows the tiny SH v = g^2.
            adam_rest=allocator.uint_buffer("sh.adam_rest", max_splats * moment_planes),
            moment_planes=moment_planes,
        )

    max_clusters = ceil_div(max_splats, CLUSTER_SIZE)
    raster = RasterResources(
        projected_2d=projected_2d,
        blend_data=blend_data,
        tile_count_per_gaussian=tile_count,
        **depth_order,
        tile_start=tile_start,
        tile_end=tile_end,
        bucket_count=bucket_count,
        bucket_offsets=bucket_offsets,
        bucket_offset_total=bucket_offset_total,
        raster_checkpoints=raster_checkpoints,
        output_image=output_image,
        gaussian_count_per_pixel=gaussian_count_per_pixel,
        cluster_aabb_min=allocator.typed_buffer("cluster_aabb_min", "float4", max_clusters),
        cluster_aabb_max=allocator.typed_buffer("cluster_aabb_max", "float4", max_clusters),
        cluster_visible=_all_clusters_visible(allocator, max_splats),
    )

    active_count.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    sort.stats.copy_from_numpy(np.zeros(2, dtype=np.uint32))
    raster.depth_sort.stats.copy_from_numpy(np.zeros(2, dtype=np.uint32))

    return GaussianBuffers(
        max_splats=max_splats, max_sort_pairs=max_sort_pairs, width=width, height=height,
        num_tiles_x=num_tiles_x, num_tiles_y=num_tiles_y,
        primitive=primitive, scan=scan, sort=sort, raster=raster, sh=sh,
        backward=backward, gradients=gradients, loss=loss, adam=adam,
        densification=densification, mip=mip, eta=eta, structure=structure,
    )


def allocate_viewer(
    device: spy.Device,
    shared_data_module: spy.Module,
    max_splats: int,
    width: int,
    height: int,
    max_sort_pairs: int = DEFAULT_MAX_SORT_PAIRS,
    sh_rest_floats: int = 0,
    _allocator=None,
) -> GaussianBuffers:
    """Allocate only what the forward pass reads (several GB less than :func:`allocate`)."""
    types = _reflect_types(shared_data_module)
    allocator = _allocator if _allocator is not None else _Alloc(device, width, height)
    num_tiles_x, num_tiles_y = _tile_grid(width, height)
    num_tiles = num_tiles_x * num_tiles_y
    max_buckets = max_raster_buckets(max_sort_pairs, num_tiles)

    primitive_values = allocator.typed_buffer("primitive.values", types.gaussian, max_splats)
    active_count = allocator.uint_buffer("primitive.active_count", 1)
    # Never computed here (mip is off in the viewer), but bin_count binds it.
    filter_3d = allocator.typed_buffer("primitive.filter_3d", "float", max_splats)
    primitive = PrimitiveStorage(primitive_values, active_count, filter_3d, types.gaussian_layout)

    projected_2d = allocator.typed_buffer("projected_2d", types.projected, max_splats)
    blend_data = allocator.typed_buffer("blend_data", types.blend, max_splats)
    tile_count = allocator.uint_buffer("tile_count_per_gaussian", max_splats)
    depth_order = _allocate_depth_order(allocator, max_splats)
    sort = _allocate_sort(allocator, max_sort_pairs)
    tile_start = allocator.uint_buffer("tile_start", num_tiles)
    tile_end = allocator.uint_buffer("tile_end", num_tiles)
    bucket_count = allocator.uint_buffer("bucket_count", num_tiles)
    bucket_offsets = allocator.uint_buffer("bucket_offsets", num_tiles)
    bucket_offset_total = allocator.uint_buffer("bucket_offset_total", 1)
    raster_checkpoints = allocator.typed_buffer(
        "raster_checkpoints", "uint2", max_buckets * TILE_AREA)
    output_image = allocator.tex(spy.Format.rgba32_float, "output_image")
    gaussian_count_per_pixel = allocator.tex(spy.Format.r32_uint, "gaussian_count_per_pixel")
    scan = _allocate_scan(allocator, max_splats, with_work=False)
    raster = RasterResources(
        projected_2d=projected_2d,
        blend_data=blend_data,
        tile_count_per_gaussian=tile_count,
        **depth_order,
        tile_start=tile_start,
        tile_end=tile_end,
        bucket_count=bucket_count,
        bucket_offsets=bucket_offsets,
        bucket_offset_total=bucket_offset_total,
        raster_checkpoints=raster_checkpoints,
        output_image=output_image,
        gaussian_count_per_pixel=gaussian_count_per_pixel,
        cluster_visible=_all_clusters_visible(allocator, max_splats),
    )

    sh = None
    if sh_rest_floats > 0:
        feat_uints = (sh_rest_floats + 1) // 2
        sh = SphericalHarmonicsResources(
            rest_floats=sh_rest_floats,
            feat_uints=feat_uints,
            features_rest=allocator.uint_buffer("sh.features_rest", max_splats * feat_uints),
            features_rest_grad=allocator.uint_buffer("sh.features_rest_grad", 1),
            adam_rest=allocator.uint_buffer("sh.adam_rest", 1),
            moment_planes=0,
        )

    active_count.copy_from_numpy(np.zeros(1, dtype=np.uint32))
    sort.stats.copy_from_numpy(np.zeros(2, dtype=np.uint32))
    raster.depth_sort.stats.copy_from_numpy(np.zeros(2, dtype=np.uint32))

    return GaussianBuffers(
        max_splats=max_splats, max_sort_pairs=max_sort_pairs, width=width, height=height,
        num_tiles_x=num_tiles_x, num_tiles_y=num_tiles_y,
        primitive=primitive, scan=scan, sort=sort, raster=raster, sh=sh,
    )


class _AllocationCounter:
    """Runs the real allocation recipe, counting bytes instead of allocating."""

    def __init__(self, width: int, height: int,
                 fixed_dtype: str = FIXED_POINT_INT64.buffer_dtype) -> None:
        self.pixels = width * height
        self.fixed_dtype = fixed_dtype
        self.total = 0

    def typed_buffer(self, label, dtype, *shape):
        from types import SimpleNamespace
        stride = ({"uint": 4, "int": 4, "float": 4, "int64_t": 8, "uint2": 8, "float4": 16}[dtype]
                  if isinstance(dtype, str)
                  else int(dtype.buffer_layout.reflection.stride))
        size = int(np.prod(shape)) * stride
        self.total += size
        return SimpleNamespace(storage=SimpleNamespace(size=size),
                               copy_from_numpy=lambda _data: None)

    def uint_buffer(self, label, *shape):
        return self.typed_buffer(label, "uint", *shape)

    def fixed_buffer(self, label, *shape):
        return self.typed_buffer(label, self.fixed_dtype, *shape)

    def indirect_buffer(self, label, records=1):
        return self.uint_buffer(label, 3 * records)

    def tex(self, fmt, label, usage=None):
        self.total += self.pixels * {
            spy.Format.rgba32_float: 16,
            spy.Format.rgba8_unorm: 4,
            spy.Format.r32_uint: 4,
        }[fmt]


def estimate_allocation_bytes(shared_data_module, *, viewer_only=False, **sizes) -> int:
    """Base buffer/texture bytes; excludes sampler resources and driver overhead."""
    fixed_dtype = fixed_point_format(shared_data_module).buffer_dtype
    counter = _AllocationCounter(sizes["width"], sizes["height"], fixed_dtype)
    allocator = allocate_viewer if viewer_only else allocate
    allocator(device=None, shared_data_module=shared_data_module,
              _allocator=counter, **sizes)
    return counter.total
