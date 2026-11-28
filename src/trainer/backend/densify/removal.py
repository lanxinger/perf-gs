# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Post-growth pruning by the squared RGB change when a single splat is removed."""

from __future__ import annotations

import numpy as np

from trainer.backend.compute.fixed_point import fixed_point_format


def accumulate_removal_cost(pipeline, step: int, hardness: float) -> None:
    raster = pipeline.buffers.raster
    kernel = pipeline.kernels.kernel("densify/removal_cost.slang", "accumulateRemovalCost")
    pipeline.commands.dispatch(kernel, "removal_cost",
        thread_count=[(pipeline.width + 3) // 4, (pipeline.height + 3) // 4, 1],
        camera=pipeline.rasterizer.camera_buffer, blend_data=raster.blend_data.storage,
        tile_start=raster.tile_start.storage, sorted_ids=pipeline.rasterizer.sorted_values.storage,
        processed=raster.gaussian_count_per_pixel, image=raster.output_image,
        scores=pipeline.buffers.densification.blend_weight.storage,
        phase=(13 * step) % 16, hardness=hardness)


def prune_by_removal_cost(pipeline, fraction: float) -> dict[str, int]:
    count = pipeline.primitives.count()
    target = min(count, max(1, int(fraction * count)))
    dtype = fixed_point_format(getattr(pipeline, "slang_session", getattr(pipeline, "device", None))).dtype
    scores = pipeline.buffers.densification.blend_weight.storage.to_numpy().view(dtype).reshape(-1)[:count]
    # Zero denotes no measurement. A small or partially sampled scene can therefore
    # finish above its requested budget.
    observed = np.flatnonzero(scores > 0)
    remove = min(count - target, observed.size)
    if remove:
        lowest = np.argpartition(scores[observed], remove - 1)[:remove]
        mask = np.zeros(pipeline.max_splats, dtype=np.uint32)
        mask[:count] = 1
        mask[observed[lowest]] = 0
        pipeline.simple_densifier.compact_mask(mask)
    return {"pre_count": count, "target_count": target, "final_count": count - remove,
            "unobserved_count": count - observed.size}
