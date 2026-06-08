"""Unified device detection - CUDA / MPS (Apple Silicon M1-M5) / ROCm (AMD) / TPU / NPU / CPU.

Auto-detects the best available compute device across all platforms including
Google Cloud TPUs (torch_xla) and Huawei Ascend NPUs (torch_npu).

Apple Silicon chip detection includes per-generation capabilities:
  - M1/M2/M3/M4/M5 chip identification with GPU cores, NE TOPS, memory bandwidth
  - M4+: Improved Neural Engine (38 TOPS), ray tracing, AV1 decode
  - M5+: Enhanced NE, multi-GPU clusters via UltraFusion
"""

from __future__ import annotations

import logging
import platform
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("aitrainer.device")


class AppleChipType(Enum):
    """Apple Silicon chip generation."""
    UNKNOWN = "unknown"
    M1 = "m1"
    M1_PRO = "m1_pro"
    M1_MAX = "m1_max"
    M1_ULTRA = "m1_ultra"
    M2 = "m2"
    M2_PRO = "m2_pro"
    M2_MAX = "m2_max"
    M2_ULTRA = "m2_ultra"
    M3 = "m3"
    M3_PRO = "m3_pro"
    M3_MAX = "m3_max"
    M4 = "m4"
    M4_PRO = "m4_pro"
    M4_MAX = "m4_max"
    M5 = "m5"
    M5_PRO = "m5_pro"
    M5_MAX = "m5_max"


# M-series chip capabilities: (gpu_cores_min, gpu_cores_max, ne_tops, memory_bw_gb_s)
_M4_CAPABILITIES: Dict[AppleChipType, Tuple[int, int, int, int]] = {
    AppleChipType.M1: (7, 8, 11, 70),
    AppleChipType.M1_PRO: (14, 16, 11, 200),
    AppleChipType.M1_MAX: (24, 32, 11, 400),
    AppleChipType.M1_ULTRA: (48, 64, 22, 800),
    AppleChipType.M2: (8, 10, 15, 100),
    AppleChipType.M2_PRO: (16, 19, 15, 200),
    AppleChipType.M2_MAX: (30, 38, 15, 400),
    AppleChipType.M2_ULTRA: (60, 76, 30, 800),
    AppleChipType.M3: (8, 10, 18, 100),
    AppleChipType.M3_PRO: (14, 18, 18, 150),
    AppleChipType.M3_MAX: (30, 40, 18, 400),
    AppleChipType.M4: (8, 10, 38, 120),
    AppleChipType.M4_PRO: (16, 20, 38, 273),
    AppleChipType.M4_MAX: (32, 40, 38, 546),
    AppleChipType.M5: (8, 12, 45, 150),
    AppleChipType.M5_PRO: (16, 24, 45, 300),
    AppleChipType.M5_MAX: (32, 48, 45, 600),
}


@dataclass
class AppleChipInfo:
    chip_type: AppleChipType = AppleChipType.UNKNOWN
    chip_name: str = "Unknown Apple Silicon"
    gpu_cores: int = 0
    neural_engine_tops: int = 0
    memory_bandwidth_gb_s: int = 0
    has_ray_tracing: bool = False
    has_av1_decode: bool = False
    has_ultra_fusion: bool = False
    is_m4_or_newer: bool = False
    is_m5_or_newer: bool = False


def _parse_apple_chip(brand_string: str) -> AppleChipInfo:
    """Parse Apple Silicon chip name into detailed chip info."""
    info = AppleChipInfo()

    # Normalize for matching
    s = brand_string.lower().replace("apple ", "")

    # Detect chip generation
    chip_mapping = {
        "m1 ultra": AppleChipType.M1_ULTRA,
        "m1 max": AppleChipType.M1_MAX,
        "m1 pro": AppleChipType.M1_PRO,
        "m1": AppleChipType.M1,
        "m2 ultra": AppleChipType.M2_ULTRA,
        "m2 max": AppleChipType.M2_MAX,
        "m2 pro": AppleChipType.M2_PRO,
        "m2": AppleChipType.M2,
        "m3 max": AppleChipType.M3_MAX,
        "m3 pro": AppleChipType.M3_PRO,
        "m3": AppleChipType.M3,
        "m4 max": AppleChipType.M4_MAX,
        "m4 pro": AppleChipType.M4_PRO,
        "m4": AppleChipType.M4,
        "m5 max": AppleChipType.M5_MAX,
        "m5 pro": AppleChipType.M5_PRO,
        "m5": AppleChipType.M5,
    }

    for key, chip_type in chip_mapping.items():
        if key in s:
            info.chip_type = chip_type
            break

    if info.chip_type == AppleChipType.UNKNOWN:
        # Fallback: detect generation number
        gen_match = re.search(r'm(\d+)', s)
        if gen_match:
            gen = int(gen_match.group(1))
            if gen >= 5:
                info.chip_type = AppleChipType.M5
            elif gen >= 4:
                info.chip_type = AppleChipType.M4
            elif gen >= 3:
                info.chip_type = AppleChipType.M3
            elif gen >= 2:
                info.chip_type = AppleChipType.M2
            else:
                info.chip_type = AppleChipType.M1

    info.chip_name = brand_string.replace("Apple ", "").strip() or "Apple Silicon"

    # Fill capabilities
    caps = _M4_CAPABILITIES.get(info.chip_type)
    if caps:
        info.gpu_cores = caps[1]  # max GPU cores
        info.neural_engine_tops = caps[2]
        info.memory_bandwidth_gb_s = caps[3]

    # Feature flags per generation
    chip_gen_match = re.match(r'm(\d+)', info.chip_type.value)
    if chip_gen_match:
        gen = int(chip_gen_match.group(1))
        info.has_ray_tracing = gen >= 3  # M3+ has hardware ray tracing
        info.has_av1_decode = gen >= 4    # M4+ has AV1 decode
        info.has_ultra_fusion = gen >= 1 and ("ultra" in info.chip_type.value)
        info.is_m4_or_newer = gen >= 4
        info.is_m5_or_newer = gen >= 5

    return info


@dataclass
class DeviceInfo:
    name: str
    device_type: str  # cuda | mps | rocm | tpu | npu | cpu
    available: bool
    device_count: int = 0
    device_name: str = ""
    memory_gb: float = 0.0
    compute_capability: str = ""
    extra: Dict[str, Any] = None

    # Apple Silicon specific (populated when device_type == "mps")
    apple_chip_info: Optional[AppleChipInfo] = None

    # NVIDIA GPU specific (populated when device_type == "cuda")
    nvidia_gpu_info: Optional[NvidiaGpuInfo] = None

    def __post_init__(self):
        if self.extra is None:
            self.extra = {}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _clean_cpu_name(raw: str) -> str:
    """Clean up verbose CPU name string."""
    # Try platform-specific readable name first
    try:
        import subprocess
        if platform.system() == "Darwin":
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                    capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                return result.stdout.strip()
    except Exception:
        pass

    # General cleanup
    cleaned = re.sub(r'\s+(Family|Model|Stepping)\s+\d+', '', raw)
    cleaned = re.sub(r',\s*\w+', '', cleaned)
    parts = cleaned.split()
    if parts and parts[0].upper() in ("AMD64", "X86_64", "I386", "ARM64"):
        return platform.system() + " x86" if "64" in parts[0] else platform.system()
    return cleaned.strip() or "CPU"


# NVIDIA GPU Generation Detection (RTX 30/40/50, Professional, Data Center)

class NvidiaGpuGeneration(Enum):
    """NVIDIA GPU architecture generation."""
    UNKNOWN = "unknown"
    TURING = "turing"           # RTX 20 series (compute 7.5)
    AMPERE = "ampere"           # RTX 30 series (compute 8.0/8.6)
    ADA_LOVELACE = "ada"        # RTX 40 series (compute 8.9)
    BLACKWELL = "blackwell"     # RTX 50 series (compute 10.0)
    HOPPER = "hopper"           # H100/H200 (compute 9.0)
    DATA_CENTER = "data_center" # A100, B100, B200
    PROFESSIONAL = "professional"  # RTX A-series, Quadro


@dataclass
class NvidiaGpuInfo:
    gpu_name: str = ""
    generation: NvidiaGpuGeneration = NvidiaGpuGeneration.UNKNOWN
    compute_capability: str = ""
    memory_gb: float = 0.0
    cuda_cores: int = 0
    tensor_cores: int = 0
    supports_flash_attention: bool = False
    supports_fp8: bool = False
    supports_fp16: bool = True
    is_professional: bool = False
    is_data_center: bool = False


# GPU name patterns → generation mapping: (pattern, generation, is_professional, supports_fp8)
_NVIDIA_GPU_PATTERNS: List[Tuple[str, NvidiaGpuGeneration, bool, bool]] = [
    # RTX 50 series "Blackwell"
    (r"(?i)rtx 5090", NvidiaGpuGeneration.BLACKWELL, False, True),
    (r"(?i)rtx 5080", NvidiaGpuGeneration.BLACKWELL, False, True),
    (r"(?i)rtx 5070", NvidiaGpuGeneration.BLACKWELL, False, True),
    (r"(?i)rtx 50\d{2}", NvidiaGpuGeneration.BLACKWELL, False, True),
    # RTX 40 series "Ada Lovelace"
    (r"(?i)rtx 4090", NvidiaGpuGeneration.ADA_LOVELACE, False, True),
    (r"(?i)rtx 4080", NvidiaGpuGeneration.ADA_LOVELACE, False, True),
    (r"(?i)rtx 4070", NvidiaGpuGeneration.ADA_LOVELACE, False, True),
    (r"(?i)rtx 40\d{2}", NvidiaGpuGeneration.ADA_LOVELACE, False, True),
    # RTX 30 series "Ampere"
    (r"(?i)rtx 3090", NvidiaGpuGeneration.AMPERE, False, False),
    (r"(?i)rtx 3080", NvidiaGpuGeneration.AMPERE, False, False),
    (r"(?i)rtx 3070", NvidiaGpuGeneration.AMPERE, False, False),
    (r"(?i)rtx 30\d{2}", NvidiaGpuGeneration.AMPERE, False, False),
    # RTX 20 series "Turing"
    (r"(?i)rtx 2080", NvidiaGpuGeneration.TURING, False, False),
    (r"(?i)rtx 2070", NvidiaGpuGeneration.TURING, False, False),
    # Professional Ada
    (r"(?i)rtx a6000", NvidiaGpuGeneration.PROFESSIONAL, True, True),
    (r"(?i)rtx a5000", NvidiaGpuGeneration.PROFESSIONAL, True, True),
    (r"(?i)rtx a4000", NvidiaGpuGeneration.PROFESSIONAL, True, True),
    (r"(?i)rtx a2000", NvidiaGpuGeneration.PROFESSIONAL, True, True),
    (r"(?i)quadro", NvidiaGpuGeneration.PROFESSIONAL, True, False),
    # Data Center
    (r"(?i)h100|h200", NvidiaGpuGeneration.HOPPER, False, True),
    (r"(?i)a100|a10|a16", NvidiaGpuGeneration.DATA_CENTER, False, False),
    (r"(?i)b100|b200", NvidiaGpuGeneration.DATA_CENTER, False, True),
]

# GPU core counts (approximate)
_NVIDIA_CUDA_CORES: Dict[str, int] = {
    "RTX 5090": 21760, "RTX 5080": 10752, "RTX 5070": 6144,
    "RTX 4090": 16384, "RTX 4080": 9728, "RTX 4070": 5888, "RTX 4060": 3072,
    "RTX 3090": 10496, "RTX 3080": 8704, "RTX 3070": 5888, "RTX 3060": 3584,
    "RTX A6000": 10752, "RTX A5000": 8192, "RTX A4000": 6144,
    "H100": 18432, "A100": 6912,
}


def _detect_nvidia_gpu() -> NvidiaGpuInfo:
    info = NvidiaGpuInfo()
    try:
        import torch
        if not torch.cuda.is_available():
            return info

        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = props.total_memory / (1024 ** 3)
        cap = f"{props.major}.{props.minor}"

        info.gpu_name = name
        info.memory_gb = round(mem, 1)
        info.compute_capability = cap
        info.supports_fp16 = True

        # Match against known patterns
        for pattern, generation, is_pro, supports_fp8 in _NVIDIA_GPU_PATTERNS:
            if re.search(pattern, name):
                info.generation = generation
                info.is_professional = is_pro
                info.is_data_center = generation in (NvidiaGpuGeneration.DATA_CENTER, NvidiaGpuGeneration.HOPPER)
                info.supports_fp8 = supports_fp8
                break

        # Estimate CUDA cores
        for gpu_name_part, cores in _NVIDIA_CUDA_CORES.items():
            if gpu_name_part.lower() in name.lower():
                info.cuda_cores = cores
                break
        if info.cuda_cores == 0:
            info.cuda_cores = getattr(props, 'multi_processor_count', 0) * 128

        # Flash Attention: compute 7.5+
        info.supports_flash_attention = props.major > 7 or (props.major == 7 and props.minor >= 5)

        logger.info(
            f"Detected NVIDIA GPU: {name} | "
            f"Gen={info.generation.value} | "
            f"Compute={cap} | "
            f"VRAM={info.memory_gb}GB | "
            f"FlashAttn={'Y' if info.supports_flash_attention else 'N'} | "
            f"FP8={'Y' if info.supports_fp8 else 'N'}"
        )

    except Exception as e:
        logger.debug(f"NVIDIA GPU detection: {e}")

    return info


def get_nvidia_gpu_info() -> NvidiaGpuInfo:
    return _detect_nvidia_gpu()


def supports_flash_attention() -> bool:
    """Check if current NVIDIA GPU supports Flash Attention (compute 7.5+)."""
    return _detect_nvidia_gpu().supports_flash_attention


def supports_fp8_training() -> bool:
    """Check if GPU supports FP8 training (H100, RTX 40/50 series)."""
    return _detect_nvidia_gpu().supports_fp8


def nvidia_gpu_generation() -> str:
    return _detect_nvidia_gpu().generation.value


def is_professional_gpu() -> bool:
    return _detect_nvidia_gpu().is_professional


def detect_device() -> DeviceInfo:
    """Detect the best available compute device in priority order."""

    # 1. Try CUDA (NVIDIA) with generation detection
    try:
        import torch
        if torch.cuda.is_available() and not _is_rocm():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            mem = props.total_memory / (1024 ** 3)
            cap = f"{props.major}.{props.minor}"
            nvidia_info = _detect_nvidia_gpu()
            return DeviceInfo(
                name=f"CUDA ({name})",
                device_type="cuda",
                available=True,
                device_count=count,
                device_name=name,
                memory_gb=round(mem, 1),
                compute_capability=cap,
                nvidia_gpu_info=nvidia_info,
            )
    except Exception as e:
        logger.debug(f"CUDA detection: {e}")

    # 2. Try TPU via torch_xla
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        n_devices = xm.xrt_world_size() if hasattr(xm, 'xrt_world_size') else 1
        dev_name = str(dev)
        return DeviceInfo(
            name=f"TPU ({dev_name})",
            device_type="tpu",
            available=True,
            device_count=n_devices,
            device_name=dev_name,
            memory_gb=_get_tpu_memory(),
            compute_capability="tpu",
            extra={"xla_version": getattr(torch_xla, "__version__", "unknown")},
        )
    except ImportError:
        logger.debug("TPU (torch_xla) not installed")
    except Exception as e:
        logger.debug(f"TPU detection: {e}")

    # 3. Try NPU via torch_npu (Huawei Ascend)
    try:
        import torch_npu
        if torch_npu.npu.is_available():
            count = torch_npu.npu.device_count()
            name = torch_npu.npu.get_device_name(0) if count > 0 else "Ascend NPU"
            mem = _get_npu_memory()
            return DeviceInfo(
                name=f"NPU ({name})",
                device_type="npu",
                available=True,
                device_count=count,
                device_name=name,
                memory_gb=mem,
                compute_capability="npu",
                extra={"npu_version": getattr(torch_npu, "__version__", "unknown")},
            )
    except ImportError:
        logger.debug("NPU (torch_npu) not installed")
    except Exception as e:
        logger.debug(f"NPU detection: {e}")

    # 3b. Try Intel XPU via intel_extension_for_pytorch
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        if torch.xpu.is_available():
            count = torch.xpu.device_count()
            name = torch.xpu.get_device_name(0) if count > 0 else "Intel XPU"
            return DeviceInfo(
                name=f"XPU ({name})",
                device_type="xpu",
                available=True,
                device_count=count,
                device_name=name,
                memory_gb=0.0,
                compute_capability="xpu",
            )
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"XPU detection: {e}")

    # 4. Try MPS (Apple Silicon) with M1-M5 chip detection
    try:
        import torch
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            system = platform.platform()
            chip_name = _get_mac_chip_name()
            chip_info = _parse_apple_chip(chip_name)

            return DeviceInfo(
                name=f"MPS ({chip_info.chip_name})",
                device_type="mps",
                available=True,
                device_count=1,
                device_name=chip_info.chip_name,
                memory_gb=_get_mac_memory(),
                compute_capability="mps",
                apple_chip_info=chip_info,
            )
    except Exception as e:
        logger.debug(f"MPS detection: {e}")

    # 5. Try ROCm (AMD)
    try:
        import torch
        if _is_rocm() and torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            return DeviceInfo(
                name=f"ROCm ({name})",
                device_type="rocm",
                available=True,
                device_count=count,
                device_name=name,
                memory_gb=0.0,
                compute_capability="rocm",
            )
    except Exception as e:
        logger.debug(f"ROCm detection: {e}")

    # 6. Fallback to CPU
    cpu_raw = platform.processor() or platform.machine() or "CPU"
    cpu_name = _clean_cpu_name(cpu_raw)
    return DeviceInfo(
        name=f"CPU ({cpu_name})",
        device_type="cpu",
        available=True,
        device_count=1,
        device_name=cpu_name,
        memory_gb=0.0,
        compute_capability="cpu",
    )


def _is_rocm() -> bool:
    try:
        import torch
        return hasattr(torch.version, 'hip') and torch.version.hip is not None
    except Exception:
        return False


def _get_tpu_memory() -> float:
    try:
        import torch_xla.core.xla_model as xm
        mem_info = xm.get_memory_info(xm.xla_device())
        if mem_info:
            return round(mem_info.get("bytes", 0) / (1024 ** 3), 1)
    except Exception:
        pass
    return 8.0  # Default TPU v2/v3 core has ~8GB HBM


def _get_npu_memory() -> float:
    try:
        import torch_npu
        import torch
        if torch_npu.npu.is_available():
            mem = torch_npu.npu.get_device_properties(0).total_memory
            return round(mem / (1024 ** 3), 1)
    except Exception:
        pass
    return 0.0


def _get_mac_chip_name() -> str:
    try:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    try:
        import platform as pl
        return pl.machine()
    except Exception:
        return "Apple Silicon"


def get_mac_chip_info() -> AppleChipInfo:
    chip_name = _get_mac_chip_name()
    return _parse_apple_chip(chip_name)


def is_m4_or_newer() -> bool:
    return get_mac_chip_info().is_m4_or_newer


def is_m5_or_newer() -> bool:
    return get_mac_chip_info().is_m5_or_newer


def apple_supports_ray_tracing() -> bool:
    return get_mac_chip_info().has_ray_tracing


def _get_mac_memory() -> float:
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return int(result.stdout.strip()) / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def get_torch_device(device_type: Optional[str] = None) -> Any:
    try:
        import torch
    except ImportError:
        import types
        return types.SimpleNamespace(device=lambda: 'cpu')

    if device_type == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            return xm.xla_device()
        except ImportError:
            logger.warning("torch_xla not installed, falling back to CPU")
            return torch.device("cpu")

    if device_type == "npu":
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                return torch.device(f"npu:0")
            return torch.device("cpu")
        except ImportError:
            logger.warning("torch_npu not installed, falling back to CPU")
            return torch.device("cpu")

    if device_type == "xpu":
        try:
            import intel_extension_for_pytorch as ipex
            if torch.xpu.is_available():
                return torch.device("xpu:0")
            return torch.device("cpu")
        except ImportError:
            return torch.device("cpu")

    if device_type == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_type == "mps":
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif device_type == "rocm":
        if torch.cuda.is_available() and _is_rocm():
            return torch.device("cuda")
        return torch.device("cpu")
    elif device_type == "cpu":
        return torch.device("cpu")
    else:
        info = detect_device()
        if info.device_type == "tpu":
            try:
                import torch_xla.core.xla_model as xm
                return xm.xla_device()
            except Exception:
                return torch.device("cpu")
        elif info.device_type == "npu":
            return torch.device("npu:0")
        elif info.device_type == "xpu":
            return torch.device("xpu:0")
        elif info.device_type == "cuda":
            return torch.device("cuda")
        elif info.device_type == "mps":
            return torch.device("mps")
        elif info.device_type == "rocm":
            return torch.device("cuda")
        else:
            return torch.device("cpu")


def get_device_summary() -> Dict[str, Any]:
    info = detect_device()
    all_devices: Dict[str, bool] = {"cpu": True}

    try:
        import torch
        all_devices["cuda"] = torch.cuda.is_available() and not _is_rocm()
        all_devices["rocm"] = _is_rocm() and torch.cuda.is_available()
        if hasattr(torch.backends, 'mps'):
            all_devices["mps"] = torch.backends.mps.is_available()
        else:
            all_devices["mps"] = False
    except Exception:
        pass

    # Check TPU
    try:
        import torch_xla
        all_devices["tpu"] = True
    except ImportError:
        all_devices["tpu"] = False

    # Check NPU
    try:
        import torch_npu
        all_devices["npu"] = torch_npu.npu.is_available()
    except ImportError:
        all_devices["npu"] = False

    # Check Intel XPU
    try:
        import intel_extension_for_pytorch as ipex
        import torch
        all_devices["xpu"] = torch.xpu.is_available()
    except ImportError:
        all_devices["xpu"] = False

    available = {k: v for k, v in all_devices.items() if v}

    result = {
        "primary": info.name,
        "type": info.device_type,
        "available_devices": available,
        "count": len(available),
        "gpu_count": info.device_count,
        "gpu_name": info.device_name,
        "memory_gb": info.memory_gb,
        "compute_capability": info.compute_capability,
        "extra": info.extra,
    }

    # Add Apple Silicon chip details when applicable
    if info.apple_chip_info:
        chip = info.apple_chip_info
        result["apple_chip"] = {
            "chip_name": chip.chip_name,
            "chip_type": chip.chip_type.value,
            "gpu_cores": chip.gpu_cores,
            "neural_engine_tops": chip.neural_engine_tops,
            "memory_bandwidth_gb_s": chip.memory_bandwidth_gb_s,
            "has_ray_tracing": chip.has_ray_tracing,
            "has_av1_decode": chip.has_av1_decode,
            "is_m4_or_newer": chip.is_m4_or_newer,
            "is_m5_or_newer": chip.is_m5_or_newer,
        }

    # Add NVIDIA GPU details when applicable
    if info.nvidia_gpu_info:
        nv = info.nvidia_gpu_info
        result["nvidia_gpu"] = {
            "gpu_name": nv.gpu_name,
            "generation": nv.generation.value,
            "compute_capability": nv.compute_capability,
            "cuda_cores": nv.cuda_cores,
            "memory_gb": nv.memory_gb,
            "supports_flash_attention": nv.supports_flash_attention,
            "supports_fp8": nv.supports_fp8,
            "is_professional": nv.is_professional,
            "is_data_center": nv.is_data_center,
        }

    return result


def get_device_for_strategy(strategy: str = "auto") -> Tuple[Any, str]:

    Args:
        strategy: 'auto', 'gpu', 'tpu', 'npu', 'cpu'

    Returns:
        (device, device_type_string)
    """
    strategy_map = {
        "tpu": "tpu",
        "npu": "npu",
        "gpu": None,  # auto-detect best GPU
        "cpu": "cpu",
        "auto": None,
    }
    mapped = strategy_map.get(strategy, None)
    if strategy == "gpu":
        info = detect_device()
        if info.device_type in ("cuda", "rocm"):
            return get_torch_device(info.device_type), info.device_type
        return torch.device("cpu"), "cpu"

    device = get_torch_device(mapped)
    if mapped:
        return device, mapped
    info = detect_device()
    return device, info.device_type


def is_tpu_available() -> bool:
    """Check if TPU is available."""
    try:
        import torch_xla
        return True
    except ImportError:
        return False


def is_npu_available() -> bool:
    try:
        import torch_npu
        return torch_npu.npu.is_available()
    except ImportError:
        return False


def is_xpu_available() -> bool:
    try:
        import intel_extension_for_pytorch
        import torch
        return torch.xpu.is_available()
    except ImportError:
        return False


# ─── TPU/NPU Training Utilities ────────────────────────────────────────

def tpu_train_step(model, batch, optimizer, device):
    import torch_xla.core.xla_model as xm
    import torch_xla.core.xla_model as xm

    outputs = model(**batch)
    loss = outputs["loss"] if isinstance(outputs, dict) else outputs
    loss.backward()
    xm.optimizer_step(optimizer, barrier=True)
    xm.mark_step()
    return loss


def npu_train_step(model, batch, optimizer, device):
    import torch
    if isinstance(batch, dict):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
    else:
        batch = tuple(b.to(device) if isinstance(b, torch.Tensor) else b for b in batch)
        outputs = model(*batch)

    loss = outputs["loss"] if isinstance(outputs, dict) else outputs
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss


# ─── Device-aware mixed precision ─────────────────────────────────────

def get_autocast_context(device_type: str):
    """Get the appropriate autocast context for mixed precision training.

    Platform-specific behavior:
    - CUDA/ROCM: torch.cuda.amp.autocast (fp16/bf16)
    - MPS (Apple Silicon): torch.mps autocast if available, fp16 native
    - TPU: bfloat16 native
    - NPU: Ascend AMP
    - CPU: no-op
    """
    import torch

    if device_type == "tpu":
        # TPU uses bfloat16 natively
        return torch.no_grad()  # placeholder
    elif device_type == "npu":
        # Ascend NPU supports amp
        return torch.cuda.amp.autocast() if hasattr(torch.cuda, 'amp') else torch.no_grad()
    elif device_type in ("cuda", "rocm"):
        return torch.cuda.amp.autocast() if hasattr(torch.cuda, 'amp') else torch.no_grad()
    elif device_type == "mps":
        # MPS supports fp16 natively; use autocast for bf16 on M3+
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'autocast'):
            return torch.mps.autocast(dtype=torch.float16)  # available in newer PyTorch
        # Fallback: MPS handles fp16/bf32 natively
        return torch.autocast(device_type="mps", dtype=torch.float16,
                              enabled=True) if hasattr(torch, 'autocast') else torch.no_grad()
    else:
        return torch.no_grad()


# Global singleton (lazy init)
DEVICE_INFO = detect_device()
try:
    PRIMARY_DEVICE, PRIMARY_DEVICE_TYPE = get_device_for_strategy("auto")
except Exception:
    import types as _t
    PRIMARY_DEVICE = _t.SimpleNamespace(device=lambda: "cpu")
    PRIMARY_DEVICE_TYPE = "cpu"
