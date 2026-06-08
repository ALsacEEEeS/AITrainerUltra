"""Universal open-source model loader.

Auto-detects model format and loads from:
- HuggingFace Hub (model ID or local cache)
- SafeTensors (.safetensors)
- PyTorch (.bin, .pt, .pth)
- GGUF (llama.cpp format via llama-cpp-python)
- ONNX (.onnx)
- LoRA adapters (PEFT)
- Custom local paths
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("aitrainer.model_loader")


class ModelFormat:
    """Detected model file format."""
    HUGGINGFACE = "huggingface"
    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    GGUF = "gguf"
    ONNX = "onnx"
    PEFT_LORA = "peft_lora"
    TENSORFLOW = "tensorflow"
    MLX = "mlx"  # Apple MLX format for Apple Silicon
    UNKNOWN = "unknown"


MODEL_FORMAT_EXTENSIONS = {
    ".safetensors": ModelFormat.SAFETENSORS,
    ".bin": ModelFormat.PYTORCH,
    ".pt": ModelFormat.PYTORCH,
    ".pth": ModelFormat.PYTORCH,
    ".gguf": ModelFormat.GGUF,
    ".ggml": ModelFormat.GGUF,
    ".onnx": ModelFormat.ONNX,
    ".h5": ModelFormat.TENSORFLOW,
    ".tf": ModelFormat.TENSORFLOW,
    ".mlx": ModelFormat.MLX,        # Apple MLX format (Apple Silicon)
    ".npz": ModelFormat.MLX,        # MLX saved weights
}

# Common HuggingFace config files that indicate a model directory
HF_CONFIG_FILES = [
    "config.json",
    "model_index.json",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
]

# MLX config files that indicate an MLX model directory
MLX_CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "weights.safetensors",
    "model.safetensors",
]

# Architecture name patterns for auto-detection
ARCH_PATTERNS = {
    "llama": r"(?i)(llama|llm|mistral|mixtral|gemma|qwen|deepseek)",
    "gpt": r"(?i)(gpt2|gpt_neo|gptj|gpt_neox|opt|bloom)",
    "bert": r"(?i)(bert|roberta|albert|distilbert|electra)",
    "t5": r"(?i)(t5|flan|mt5|bart|pegasus)",
    "vit": r"(?i)(vit|deit|beit|swin|convnext)",
    "clip": r"(?i)(clip|blip)",
    "whisper": r"(?i)(whisper)",
    "stable-diffusion": r"(?i)(diffusion|unet|vae|sd\d)",
}


def detect_model_format(path: str) -> Tuple[str, str]:
    """Detect model format and architecture from a path.

    Args:
        path: HuggingFace model ID or local file/directory path.

    Returns:
        (format, architecture_hint)
    """
    path_str = str(path)

    # Check if it's a HuggingFace model ID (no file extension, no local path)
    if "/" in path_str and not os.path.exists(path_str):
        parts = path_str.split("/")
        if len(parts) == 2 and re.match(r'^[\w.-]+$', parts[0]) and re.match(r'^[\w.-]+$', parts[1]):
            arch = _detect_hf_architecture(path_str)
            return ModelFormat.HUGGINGFACE, arch

    p = Path(path_str)

    if not p.exists():
        # Not a local path, try as HuggingFace ID
        if "/" in path_str:
            arch = _detect_hf_architecture(path_str)
            return ModelFormat.HUGGINGFACE, arch
        return ModelFormat.UNKNOWN, ""

    if p.is_dir():
        # Directory - check for config files
        return _detect_directory_format(p)

    if p.is_file():
        ext = p.suffix.lower()
        fmt = MODEL_FORMAT_EXTENSIONS.get(ext, ModelFormat.UNKNOWN)
        arch = _detect_architecture_from_name(p.name)
        return fmt, arch

    return ModelFormat.UNKNOWN, ""


def _detect_directory_format(path: Path) -> Tuple[str, str]:
    files = [f.name for f in path.iterdir() if f.is_file()]

    # Check for HuggingFace config
    if any(cf in files for cf in HF_CONFIG_FILES):
        # Could also be an MLX model — check for MLX weight files
        mlx_files = [f for f in files if f.endswith(".mlx") or f.endswith(".npz")]
        if mlx_files:
            return ModelFormat.MLX, _detect_architecture_from_name(mlx_files[0])
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                arch = config.get("model_type", "") or config.get("_model_type", "")
                return ModelFormat.HUGGINGFACE, arch
            except Exception:
                pass
        return ModelFormat.HUGGINGFACE, ""

    # Check for adapter config (LoRA)
    if "adapter_config.json" in files:
        return ModelFormat.PEFT_LORA, "peft"

    # Count file types
    safetensors = [f for f in files if f.endswith(".safetensors")]
    pytorch = [f for f in files if f.endswith((".bin", ".pt", ".pth"))]
    gguf = [f for f in files if f.endswith(".gguf")]
    onnx = [f for f in files if f.endswith(".onnx")]

    if safetensors:
        return ModelFormat.SAFETENSORS, _detect_architecture_from_name(safetensors[0])
    if gguf:
        return ModelFormat.GGUF, _detect_architecture_from_name(gguf[0])
    if pytorch:
        return ModelFormat.PYTORCH, _detect_architecture_from_name(pytorch[0])
    if onnx:
        return ModelFormat.ONNX, _detect_architecture_from_name(onnx[0])

    return ModelFormat.UNKNOWN, ""


def _detect_hf_architecture(model_id: str) -> str:
    name = model_id.lower()
    for arch, pattern in ARCH_PATTERNS.items():
        if re.search(pattern, name):
            return arch
    return ""


def _detect_architecture_from_name(filename: str) -> str:
    name = filename.lower()
    for arch, pattern in ARCH_PATTERNS.items():
        if re.search(pattern, name):
            return arch
    return ""


def get_model_info(path: str) -> Dict[str, Any]:
    fmt, arch = detect_model_format(path)
    p = Path(path)

    info = {
        "path": path,
        "format": fmt,
        "architecture": arch,
        "exists": p.exists(),
        "is_dir": p.is_dir() if p.exists() else False,
        "size_bytes": 0,
        "size_readable": "",
        "num_files": 0,
        "file_types": {},
    }

    if p.exists():
        if p.is_dir():
            total = 0
            for f in p.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
                    ext = f.suffix.lower()
                    info["file_types"][ext] = info["file_types"].get(ext, 0) + 1
                    info["num_files"] += 1
            info["size_bytes"] = total
        else:
            info["size_bytes"] = p.stat().st_size
            info["num_files"] = 1

        info["size_readable"] = _format_bytes(info["size_bytes"])

    return info


def _format_bytes(b: int) -> str:
    if b < 1024:
        return f"{b}B"
    elif b < 1024 ** 2:
        return f"{b / 1024:.1f}KB"
    elif b < 1024 ** 3:
        return f"{b / 1024 ** 2:.1f}MB"
    else:
        return f"{b / 1024 ** 3:.2f}GB"


async def load_model_from_path(
    path: str,
    device: str = "auto",
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Load a model from any supported format.

    Args:
        path: HuggingFace ID, local path, or file path.
        device: 'auto', 'cpu', 'cuda', 'mps', etc.
        load_in_8bit: Enable 8-bit quantization.
        load_in_4bit: Enable 4-bit quantization.
        use_cache: Use HuggingFace cache.

    Returns:
        Dict with 'model', 'tokenizer', 'format', 'architecture'.
    """
    fmt, arch = detect_model_format(path)

    if fmt == ModelFormat.HUGGINGFACE:
        return await _load_huggingface(
            path, device, load_in_8bit, load_in_4bit, use_cache,
        )
    elif fmt == ModelFormat.SAFETENSORS:
        return await _load_safetensors(path, device)
    elif fmt == ModelFormat.PYTORCH:
        return await _load_pytorch(path, device, arch)
    elif fmt == ModelFormat.GGUF:
        return await _load_gguf(path, device)
    elif fmt == ModelFormat.ONNX:
        return _load_onnx(path, device)
    elif fmt == ModelFormat.PEFT_LORA:
        return await _load_peft_lora(path, device)
    elif fmt == ModelFormat.MLX:
        return _load_mlx(path, device)
    else:
        # Try as HuggingFace ID anyway
        try:
            return await _load_huggingface(
                path, device, load_in_8bit, load_in_4bit, use_cache,
            )
        except Exception as e:
            raise ValueError(
                f"Could not load model from '{path}'. "
                f"Detected format: {fmt}. Error: {e}"
            )


async def _load_huggingface(
    path: str, device: str,
    load_in_8bit: bool, load_in_4bit: bool,
    use_cache: bool,
) -> Dict[str, Any]:
    """Load from HuggingFace Hub or local HF directory."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoModelForImageClassification,
        AutoModel,
        AutoTokenizer,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    model_name = path
    kwargs = {
        "device_map": "auto" if device == "auto" else device,
        "torch_dtype": "auto",
    }

    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
        )
    elif load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    if not use_cache:
        kwargs["cache_dir"] = None

    # Try loading tokenizer/model
    tokenizer = None
    processor = None
    model = None

    # Try different model classes with detailed error tracking
    model_classes = [
        ("AutoModelForCausalLM", AutoModelForCausalLM),
        ("AutoModelForSequenceClassification", AutoModelForSequenceClassification),
        ("AutoModelForImageClassification", AutoModelForImageClassification),
        ("AutoModel", AutoModel),
    ]

    last_error = ""
    for class_name, model_class in model_classes:
        try:
            logger.info(f"Trying to load '{model_name}' with {class_name}")
            model = model_class.from_pretrained(model_name, **kwargs)
            logger.info(f"Successfully loaded with {class_name}")
            break
        except ImportError as e:
            last_error = f"缺少依赖: {e}"
            logger.debug(f"{class_name} failed: {e}")
            continue
        except OSError as e:
            last_error = f"模型 '{model_name}' 不存在或无法访问: {e}"
            logger.warning(last_error)
            continue
        except Exception as e:
            last_error = f"{class_name}: {e}"
            logger.debug(f"{class_name} failed: {e}")
            continue

    if model is None:
        raise ValueError(
            f"无法加载模型 '{model_name}'. "
            f"最后错误: {last_error}. "
            f"请确认模型ID正确, 或检查网络连接和磁盘空间"
        )

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception:
        pass

    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception:
        pass

    arch = getattr(model.config, "model_type", "") if hasattr(model, "config") else ""

    return {
        "model": model,
        "tokenizer": tokenizer,
        "processor": processor,
        "format": ModelFormat.HUGGINGFACE,
        "architecture": arch,
        "model_name": model_name,
        "model_class": class_name if model else "unknown",
        "params": sum(p.numel() for p in model.parameters()) if model else 0,
    }


async def _load_safetensors(path: str, device: str) -> Dict[str, Any]:
    """Load from SafeTensors files."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    p = Path(path)
    if p.is_dir():
        # Find all .safetensors files and try to load as HuggingFace
        return await _load_huggingface(str(p), device, False, False, True)

    tensors = load_file(path)
    # Return raw tensors - caller needs to know architecture
    return {
        "tensors": tensors,
        "format": ModelFormat.SAFETENSORS,
        "architecture": _detect_architecture_from_name(path),
        "num_tensors": len(tensors),
    }


async def _load_pytorch(path: str, device: str, arch: str) -> Dict[str, Any]:
    """Load from PyTorch checkpoint."""
    import torch

    p = Path(path)
    if p.is_dir():
        return await _load_huggingface(str(p), device, False, False, True)

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    return {
        "checkpoint": checkpoint,
        "format": ModelFormat.PYTORCH,
        "architecture": arch,
        "is_state_dict": isinstance(checkpoint, dict) and any(
            k.endswith((".weight", ".bias", "gamma", "beta"))
            for k in list(checkpoint.keys())[:10]
        ),
    }


async def _load_gguf(path: str, device: str) -> Dict[str, Any]:
    """Load GGUF model (requires llama-cpp-python or ctransformers)."""
    result = {
        "model": None,
        "format": ModelFormat.GGUF,
        "architecture": _detect_architecture_from_name(path),
        "path": path,
        "backend": None,
    }

    # Try llama-cpp-python
    try:
        from llama_cpp import Llama
        model = Llama(
            model_path=path,
            n_ctx=4096,
            n_gpu_layers=-1 if device != "cpu" else 0,
            verbose=False,
        )
        result["model"] = model
        result["backend"] = "llama-cpp"
        return result
    except ImportError:
        pass

    # Try ctransformers
    try:
        from ctransformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(path)
        result["model"] = model
        result["backend"] = "ctransformers"
        return result
    except ImportError:
        pass

    result["error"] = "GGUF loading requires llama-cpp-python or ctransformers"
    return result


def _load_onnx(path: str, device: str) -> Dict[str, Any]:
    """Load ONNX model."""
    import onnxruntime as ort

    p = Path(path)
    if p.is_dir():
        onnx_files = list(p.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No .onnx files found in {path}")
        path = str(onnx_files[0])

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
        if device != "cpu" else ["CPUExecutionProvider"]

    session = ort.InferenceSession(path, providers=providers)

    return {
        "session": session,
        "format": ModelFormat.ONNX,
        "architecture": _detect_architecture_from_name(path),
        "inputs": [{"name": i.name, "shape": i.shape, "type": i.type} for i in session.get_inputs()],
        "outputs": [{"name": o.name, "shape": o.shape, "type": o.type} for o in session.get_outputs()],
    }


async def _load_peft_lora(path: str, device: str) -> Dict[str, Any]:
    import torch
    from peft import PeftModel, PeftConfig

    config = PeftConfig.from_pretrained(path)
    base_model_path = config.base_model_name_or_path

    from transformers import AutoModelForCausalLM
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float32,
    )
    model = PeftModel.from_pretrained(base_model, path)

    return {
        "model": model,
        "format": ModelFormat.PEFT_LORA,
        "architecture": config.task_type or "peft",
        "base_model": base_model_path,
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "target_modules": config.target_modules,
    }


def _load_mlx(path: str, device: str) -> Dict[str, Any]:
    """Load MLX format model optimized for Apple Silicon.

    MLX models leverage Apple's unified memory architecture on M-series chips.
    Requires mlx (Apple ML framework) which only runs on macOS.
    Falls back to PyTorch loading when MLX is unavailable.
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn

        p = Path(path)
        if p.is_dir():
            # Load MLX model directory
            try:
                from mlx_lm import load as mlx_load
                model, tokenizer = mlx_load(path)
                return {
                    "model": model,
                    "tokenizer": tokenizer,
                    "format": ModelFormat.MLX,
                    "architecture": _detect_architecture_from_name(path),
                    "backend": "mlx-lm",
                }
            except ImportError:
                pass

            # Fallback: load individual weight files
            weights = {}
            for f in p.glob("*.safetensors"):
                try:
                    from safetensors import safe_open
                    with safe_open(str(f), framework="mlx") as sf:
                        for key in sf.keys():
                            weights[key] = sf.get_tensor(key)
                except Exception:
                    pass
            for f in p.glob("*.npz"):
                data = mx.load(str(f))
                if isinstance(data, dict):
                    weights.update(data)

            return {
                "weights": weights,
                "format": ModelFormat.MLX,
                "architecture": _detect_architecture_from_name(path),
                "num_tensors": len(weights),
                "backend": "mlx-weights",
            }

        # Single file
        if p.suffix == ".npz":
            weights = mx.load(path)
            return {
                "weights": weights if isinstance(weights, dict) else {"data": weights},
                "format": ModelFormat.MLX,
                "architecture": _detect_architecture_from_name(path),
                "backend": "mlx-npz",
            }
        if p.suffix == ".mlx":
            weights = mx.load(path)
            return {
                "weights": weights if isinstance(weights, dict) else {"data": weights},
                "format": ModelFormat.MLX,
                "architecture": _detect_architecture_from_name(path),
                "backend": "mlx",
            }

        # Unknown MLX format — try as HuggingFace
        return {
            "format": ModelFormat.MLX,
            "error": "Unknown MLX format",
            "architecture": _detect_architecture_from_name(path),
        }

    except ImportError:
        logger.warning("mlx not installed; falling back to PyTorch for MLX model")
        # Fallback: try loading as HuggingFace format
        import asyncio
        return asyncio.run(_load_huggingface(path, device, False, False, True))


def list_model_files(path: str) -> List[Dict[str, Any]]:
    """List all model files in a directory with format info."""
    p = Path(path)
    if not p.exists():
        return []

    if p.is_file():
        fmt, arch = detect_model_format(str(p))
        return [{
            "path": str(p),
            "name": p.name,
            "format": fmt,
            "architecture": arch,
            "size": p.stat().st_size,
            "size_readable": _format_bytes(p.stat().st_size),
        }]

    results = []
    for f in sorted(p.iterdir()):
        if f.is_file():
            fmt, arch = detect_model_format(str(f))
            results.append({
                "path": str(f),
                "name": f.name,
                "format": fmt,
                "architecture": arch,
                "size": f.stat().st_size,
                "size_readable": _format_bytes(f.stat().st_size),
            })
        elif f.is_dir():
            # Check if it's a model directory (has config.json)
            if (f / "config.json").exists():
                total_size = sum(
                    subf.stat().st_size for subf in f.rglob("*") if subf.is_file()
                )
                fmt, arch = detect_model_format(str(f))
                results.append({
                    "path": str(f),
                    "name": f.name + "/",
                    "format": fmt,
                    "architecture": arch,
                    "size": total_size,
                    "size_readable": _format_bytes(total_size),
                    "is_directory": True,
                })

    return results


def get_model_metadata(path: str) -> Dict[str, Any]:
    """Extract model metadata (config.json) without loading the model."""
    p = Path(path)
    if p.is_dir():
        config_path = p / "config.json"
    elif p.is_file():
        config_path = p.parent / "config.json"
    else:
        try:
            from huggingface_hub import model_info
            info = model_info(path)
            return {
                "model_id": path,
                "pipeline_tag": info.pipeline_tag,
                "tags": info.tags,
                "downloads": info.downloads,
                "likes": info.likes,
                "library": info.library_name,
            }
        except Exception:
            return {"error": "Could not fetch metadata"}

    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            return {
                "model_type": config.get("model_type", ""),
                "architecture": config.get("architectures", []),
                "hidden_size": config.get("hidden_size", config.get("d_model", "")),
                "num_layers": config.get("num_hidden_layers",
                             config.get("num_layers", config.get("n_layer", ""))),
                "num_heads": config.get("num_attention_heads",
                            config.get("n_head", "")),
                "vocab_size": config.get("vocab_size", ""),
                "max_position": config.get("max_position_embeddings",
                              config.get("n_positions", "")),
            }
        except Exception as e:
            return {"error": str(e)}

    return {"error": "No config.json found"}


# Model format compatibility table
MODEL_FORMAT_TABLE = [
    {"format": "HuggingFace Hub", "ext": "model ID", "lib": "transformers", "desc": "从HuggingFace加载在线模型"},
    {"format": "SafeTensors", "ext": ".safetensors", "lib": "safetensors", "desc": "安全的序列化格式，速度快"},
    {"format": "PyTorch", "ext": ".bin / .pt / .pth", "lib": "torch", "desc": "PyTorch原生格式"},
    {"format": "GGUF", "ext": ".gguf", "lib": "llama-cpp-python", "desc": "llama.cpp量化格式，CPU友好"},
    {"format": "ONNX", "ext": ".onnx", "lib": "onnxruntime", "desc": "开放神经网络交换格式"},
    {"format": "PEFT LoRA", "ext": "adapter_config.json", "lib": "peft", "desc": "LoRA/QLoRA适配器"},
    {"format": "TensorFlow", "ext": ".h5 / .tf", "lib": "tensorflow", "desc": "TensorFlow SavedModel"},
    {"format": "MLX (Apple Silicon)", "ext": ".mlx / .npz", "lib": "mlx", "desc": "Apple MLX格式，M系列芯片优化"},
]
