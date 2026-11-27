# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Numpy dtypes for shader structs, read from slangpy reflection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import slangpy as spy

_Kind = spy.TypeReflection.Kind
_Scalar = spy.TypeReflection.ScalarType
_NUMPY_SCALARS = {
    _Scalar.float32: np.float32,
    _Scalar.float16: np.float16,
    _Scalar.int32: np.int32,
    _Scalar.uint32: np.uint32,
    _Scalar.int64: np.int64,
    _Scalar.uint64: np.uint64,
}


# Scalar-packed Metal struct fields coalesced back into logical vector fields
# (exact offsets still come from Slang reflection).
_PACKED_VECTOR_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mean", ("_mean_x", "_mean_y", "_mean_z")),
    ("log_scale", ("_log_scale_x", "_log_scale_y", "_log_scale_z")),
    ("rot_quat", ("_rot_w", "_rot_x", "_rot_y", "_rot_z")),
    ("sh0", ("_sh0_r", "_sh0_g", "_sh0_b")),
)


@dataclass(frozen=True)
class StructLayout:
    """A shader struct's byte layout as a structured numpy dtype."""

    name: str
    dtype: np.dtype

    @property
    def float_count(self) -> int:
        return self.dtype.itemsize // 4

    @classmethod
    def of(cls, slang_type) -> StructLayout:
        layout = slang_type.buffer_layout.reflection
        raw_fields = list(layout.fields)
        names, formats, offsets = [], [], []
        idx = 0
        while idx < len(raw_fields):
            matched = False
            for logical, members in _PACKED_VECTOR_GROUPS:
                end = idx + len(members)
                if end <= len(raw_fields) and tuple(f.name for f in raw_fields[idx:end]) == members:
                    group = raw_fields[idx:end]
                    first_type = group[0].type_layout.type
                    scalar = _NUMPY_SCALARS[first_type.scalar_type]
                    elem_size = np.dtype(scalar).itemsize
                    base_offset = group[0].offset
                    assert all(
                        f.type_layout.type.kind == _Kind.scalar
                        and _NUMPY_SCALARS[f.type_layout.type.scalar_type] is scalar
                        and f.offset == base_offset + k * elem_size
                        for k, f in enumerate(group)
                    ), f"packed fields {members} are not contiguous"
                    names.append(logical)
                    formats.append((scalar, (len(members),)))
                    offsets.append(base_offset)
                    idx = end
                    matched = True
                    break
            if matched:
                continue
            field = raw_fields[idx]
            field_type = field.type_layout.type
            scalar = _NUMPY_SCALARS[field_type.scalar_type]
            if field_type.kind == _Kind.scalar:
                formats.append(scalar)
            elif field_type.kind == _Kind.vector:
                formats.append((scalar, (field_type.col_count,)))
            else:
                formats.append((scalar, (field_type.row_count, field_type.col_count)))
            names.append(field.name)
            offsets.append(field.offset)
            idx += 1
        dtype = np.dtype({"names": names, "formats": formats, "offsets": offsets,
                          "itemsize": layout.stride})
        return cls(slang_type.name, dtype)
