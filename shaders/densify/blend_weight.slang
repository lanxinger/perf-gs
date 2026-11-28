// Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

#ifndef TRAINER_BLEND_WEIGHT_SLANG
#define TRAINER_BLEND_WEIGHT_SLANG

#include "../common/fixed_point.slang"

// True when the per-view average blend weight (alpha * T summed over pixels) is under min_blend_weight;
// splats never seen in the window are kept (no evidence), and min_blend_weight <= 0 disables.
bool isUnderweight(fixed_t blend_weight, uint views, float min_blend_weight)
{
    return min_blend_weight > 0.0 && views > 0u
        && fromFixed(blend_weight) < min_blend_weight * float(views);
}

#endif // TRAINER_BLEND_WEIGHT_SLANG
