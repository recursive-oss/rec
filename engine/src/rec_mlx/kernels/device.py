from dataclasses import dataclass


@dataclass(frozen=True)
class DeviceProfile:
    chip: str
    architecture: str
    workers: int = 36
    mlp_workers: int = 32
    head_workers: int = 32
    prefill_chunk_size: int = 2048
    specialize_mixed_experts: bool = True


def profile_for_device(info):
    chips = {
        ("applegpu_g15s", "Apple M3 Max"): "M3 Max",
        ("applegpu_g16s", "Apple M4 Max"): "M4 Max",
        ("applegpu_g16s", "Apple M4 Pro"): "M4 Pro",
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
    return DeviceProfile(
        chip=chips[identity],
        architecture=architecture,
        prefill_chunk_size=4096 if chips[identity] == "M5 Pro" else 2048,
        # The specialized resident shader fails scalar/packet parity on M4 Pro
        # and is unverified on M3 Max and M4 Max.
        specialize_mixed_experts=chips[identity] not in ("M3 Max", "M4 Max", "M4 Pro"),
    )
