import ctypes
import ctypes.util
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class DeviceProfile:
    chip: str
    architecture: str
    workers: int = 36
    mlp_workers: int = 32
    head_workers: int = 32
    prefill_chunk_size: int = 2048
    specialize_mixed_experts: bool = True


@lru_cache(maxsize=1)
def _gpu_core_count(default=18):
    try:
        iokit = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit"))
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        iokit.IOServiceMatching.restype = ctypes.c_void_p
        iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint, ctypes.c_void_p]
        iokit.IORegistryEntryCreateCFProperty.argtypes = [
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint32,
        ]
        iokit.IORegistryEntryCreateCFProperty.restype = ctypes.c_void_p
        cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        cf.CFStringCreateWithCString.restype = ctypes.c_void_p
        cf.CFNumberGetValue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        cf.CFRelease.argtypes = [ctypes.c_void_p]

        service = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AGXAccelerator"))
        if not service:
            return default
        key = cf.CFStringCreateWithCString(None, b"gpu-core-count", 0x08000100)
        prop = iokit.IORegistryEntryCreateCFProperty(service, key, None, 0)
        cf.CFRelease(key)
        iokit.IOObjectRelease(service)
        if not prop:
            return default
        value = ctypes.c_int64(0)
        ok = cf.CFNumberGetValue(prop, 4, ctypes.byref(value))
        cf.CFRelease(prop)
        return int(value.value) if ok and value.value > 0 else default
    except Exception:
        return default


def profile_for_device(info):
    chips = {
        ("applegpu_g16s", "Apple M4 Pro"): "M4 Pro",
        ("applegpu_g16s", "Apple M4 Max"): "M4 Max",
        ("applegpu_g17s", "Apple M5 Pro"): "M5 Pro",
        ("applegpu_g17s", "Apple M5 Max"): "M5 Max",
    }
    architecture = info["architecture"]
    identity = (architecture, info["device_name"])
    if identity not in chips:
        raise ValueError(f"Unsupported resident kernel device: {identity}")
    if chips[identity] == "M5 Max":
        return DeviceProfile(
            chip=chips[identity],
            architecture=architecture,
            workers=76,
            mlp_workers=64,
            head_workers=64,
            prefill_chunk_size=4096,
        )
    if chips[identity] == "M5 Pro":
        cores = info.get("gpu_cores") or _gpu_core_count()
        workers = max(32, min(40, (cores * 2) // 4 * 4))
        return DeviceProfile(
            chip=chips[identity],
            architecture=architecture,
            workers=workers,
            mlp_workers=32,
            head_workers=workers,
            prefill_chunk_size=4096,
        )
    return DeviceProfile(
        chip=chips[identity],
        architecture=architecture,
        prefill_chunk_size=2048,
        # The specialized M4 resident shader fails scalar/packet parity.
        specialize_mixed_experts=chips[identity] not in ("M4 Pro", "M4 Max"),
    )
