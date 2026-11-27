# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""Windows-only DXGI probe of this process's VRAM budget and usage (any graphics API);
exceeding the budget makes WDDM page to system memory rather than fail allocations."""
from __future__ import annotations

import ctypes
import sys
import uuid

_IID_IDXGIFACTORY1 = uuid.UUID("770aae78-f26f-4dba-a829-253c83d1b387")
_IID_IDXGIADAPTER3 = uuid.UUID("645967a4-1392-4310-a798-8053ce3e93fd")
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_DXGI_MEMORY_SEGMENT_GROUP_LOCAL = 0
# COM vtable slots (dxgi.h / dxgi1_4.h declaration order).
_RELEASE = 2
_QUERY_INTERFACE = 0
_ENUM_ADAPTERS1 = 12
_GET_DESC1 = 10
_QUERY_VIDEO_MEMORY_INFO = 14


class _AdapterDesc1(ctypes.Structure):
    _fields_ = [("Description", ctypes.c_wchar * 128),
                ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
                ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", ctypes.c_uint64),
                ("Flags", ctypes.c_uint)]


class _VideoMemoryInfo(ctypes.Structure):
    _fields_ = [("Budget", ctypes.c_uint64), ("CurrentUsage", ctypes.c_uint64),
                ("AvailableForReservation", ctypes.c_uint64),
                ("CurrentReservation", ctypes.c_uint64)]


def _call(obj, slot, *args):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    argtypes = [type(arg) if isinstance(arg, ctypes._SimpleCData) else ctypes.c_void_p
                for arg in args]
    method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])
    return method(obj, *args)


def query_video_memory(adapter_name: str) -> tuple[int, int] | None:
    """``(budget, current_usage)`` bytes for the adapter named ``adapter_name``, else for the
    hardware adapter with the most dedicated memory; uses DXGI on Windows, Metal on macOS,
    and returns None on other platforms.
    """
    if sys.platform == "darwin":
        from trainer.backend.platform.metal import query_metal_video_memory
        return query_metal_video_memory(adapter_name)
    if sys.platform != "win32":
        return None
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        return None
    factory = ctypes.c_void_p()
    iid = ctypes.create_string_buffer(_IID_IDXGIFACTORY1.bytes_le)
    if dxgi.CreateDXGIFactory1(iid, ctypes.byref(factory)) < 0:
        return None
    adapters = []
    try:
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if _call(factory, _ENUM_ADAPTERS1, ctypes.c_uint(index),
                     ctypes.byref(adapter)) < 0:
                break           # DXGI_ERROR_NOT_FOUND ends the enumeration
            adapters.append(adapter)
            index += 1
        best, best_key = None, None
        for adapter in adapters:
            desc = _AdapterDesc1()
            if _call(adapter, _GET_DESC1, ctypes.byref(desc)) < 0:
                continue
            if desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE:
                continue
            key = (desc.Description == adapter_name, desc.DedicatedVideoMemory)
            if best_key is None or key > best_key:
                best, best_key = adapter, key
        if best is None:
            return None
        adapter3 = ctypes.c_void_p()
        iid3 = ctypes.create_string_buffer(_IID_IDXGIADAPTER3.bytes_le)
        if _call(best, _QUERY_INTERFACE, iid3, ctypes.byref(adapter3)) < 0:
            return None
        try:
            info = _VideoMemoryInfo()
            if _call(adapter3, _QUERY_VIDEO_MEMORY_INFO, ctypes.c_uint(0),
                     ctypes.c_int(_DXGI_MEMORY_SEGMENT_GROUP_LOCAL),
                     ctypes.byref(info)) < 0:
                return None
            return int(info.Budget), int(info.CurrentUsage)
        finally:
            _call(adapter3, _RELEASE)
    finally:
        for adapter in adapters:
            _call(adapter, _RELEASE)
        _call(factory, _RELEASE)
