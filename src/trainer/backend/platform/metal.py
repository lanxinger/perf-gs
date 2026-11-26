# Copyright (c) 2026 Electronic Arts Inc. All rights reserved.

"""macOS Metal probe of this process's recommended GPU working-set budget and current usage."""
from __future__ import annotations

import ctypes
import sys


def query_metal_video_memory(adapter_name: str) -> tuple[int, int] | None:
    """``(recommended_max_working_set_bytes, current_allocated_bytes)`` for the Metal device
    named ``adapter_name``, else the device with the largest recommended working set;
    None off macOS or when Metal is unavailable.
    """
    if sys.platform != "darwin":
        return None
    try:
        objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    except OSError:
        return None

    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p

    msg_ptr = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc))
    msg_u64 = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc))
    msg_idx = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64)(
        ("objc_msgSend", objc))
    msg_cstr = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc))
    msg_void = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc))

    metal.MTLCopyAllDevices.restype = ctypes.c_void_p
    devices = metal.MTLCopyAllDevices()
    if not devices:
        return None

    sel_count = objc.sel_registerName(b"count")
    sel_object_at = objc.sel_registerName(b"objectAtIndex:")
    sel_name = objc.sel_registerName(b"name")
    sel_utf8 = objc.sel_registerName(b"UTF8String")
    sel_budget = objc.sel_registerName(b"recommendedMaxWorkingSetSize")
    sel_usage = objc.sel_registerName(b"currentAllocatedSize")
    sel_release = objc.sel_registerName(b"release")

    try:
        count = msg_u64(devices, sel_count)
        best: tuple[int, int] | None = None
        best_key: tuple[bool, int] | None = None
        for index in range(count):
            dev = msg_idx(devices, sel_object_at, index)
            if not dev:
                continue
            budget = int(msg_u64(dev, sel_budget))
            if budget <= 0:
                continue
            usage = int(msg_u64(dev, sel_usage))
            ns_name = msg_ptr(dev, sel_name)
            raw_name = msg_cstr(ns_name, sel_utf8) if ns_name else None
            name = raw_name.decode("utf-8", errors="replace") if raw_name else ""
            key = (name == adapter_name, budget)
            if best_key is None or key > best_key:
                best, best_key = (budget, usage), key
        return best
    finally:
        msg_void(devices, sel_release)
