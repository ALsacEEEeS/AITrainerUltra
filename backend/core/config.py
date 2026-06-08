"""Configuration management for training jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HyperParameters:
    """Training hyperparameters shared across model types.

    无硬性限制 — 所有参数均可根据硬件能力自由调整。
    M4/M5 统一内存架构支持更大的 batch_size 和 seq_length。
    """
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    gradient_accumulation_steps: int = 1
    fp16: bool = False
    bf16: bool = False


@dataclass
class LoraConfigData:
    """LoRA-specific hyperparameters."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class QLoraConfigData:
    """QLoRA-specific quantization parameters."""
    bits: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "float16"


@dataclass
class DatasetConfig:
    """Dataset configuration.

    max_seq_length 默认2048 (兼容M4/M5的128K上下文).
    设为 -1 可禁用截断, 使用数据集原始长度.
    """
    path: str = ""
    name: str = ""
    split: str = "train"
    text_column: str = "text"
    max_samples: Optional[int] = None  # None=全部, 内存不足可设500-5000
    max_seq_length: int = 2048  # M4支持32K, M5支持128K. 设为-1则禁用截断


@dataclass
class TrainingConfig:
    """Top-level training configuration."""
    model_type: str = "llm"
    model_name: str = ""
    output_dir: str = "./output"
    task: str = "text-generation"
    device_strategy: str = "auto"  # auto | cuda | mps | rocm | tpu | npu | cpu

    hyperparameters: HyperParameters = field(default_factory=HyperParameters)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lora: Optional[LoraConfigData] = None
    qlora: Optional[QLoraConfigData] = None
    scratch_config: Optional[Dict[str, Any]] = None  # From-scratch architecture params

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "output_dir": self.output_dir,
            "task": self.task,
            "hyperparameters": asdict(self.hyperparameters),
            "dataset": asdict(self.dataset),
            "lora": asdict(self.lora) if self.lora else None,
            "qlora": asdict(self.qlora) if self.qlora else None,
            "scratch_config": self.scratch_config,
        }

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        """Load configuration from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = cls(
            model_type=data.get("model_type", "llm"),
            model_name=data.get("model_name", ""),
            output_dir=data.get("output_dir", "./output"),
            task=data.get("task", "text-generation"),
            hyperparameters=HyperParameters(**data.get("hyperparameters", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
        )
        if data.get("lora"):
            cfg.lora = LoraConfigData(**data["lora"])
        if data.get("qlora"):
            cfg.qlora = QLoraConfigData(**data["qlora"])
        if data.get("scratch_config"):
            cfg.scratch_config = data["scratch_config"]
        return cfg


# Predefined model presets
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    "tiny-llm": {
        "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "model_type": "llm",
        "task": "text-generation",
    },
    "llama-3-8b": {
        "model_name": "meta-llama/Meta-Llama-3-8B",
        "model_type": "llm",
        "task": "text-generation",
    },
    "mistral-7b": {
        "model_name": "mistralai/Mistral-7B-v0.1",
        "model_type": "llm",
        "task": "text-generation",
    },
    "sd-lcm": {
        "model_name": "SimianLuo/LCM_Dreamshaper_v7",
        "model_type": "lcm",
        "task": "image-generation",
    },
    "resnet-50": {
        "model_name": "microsoft/resnet-50",
        "model_type": "cnn",
        "task": "image-classification",
    },
    # === 🆕 New presets ===
    "gpt2": {
        "model_name": "gpt2",
        "model_type": "gpt",
        "task": "text-generation",
    },
    "gpt2-medium": {
        "model_name": "gpt2-medium",
        "model_type": "gpt",
        "task": "text-generation",
    },
    "bert-base": {
        "model_name": "bert-base-uncased",
        "model_type": "bert",
        "task": "sequence-classification",
    },
    "bert-ner": {
        "model_name": "dslim/bert-base-NER",
        "model_type": "bert",
        "task": "token-classification",
    },
    "clip-vit": {
        "model_name": "openai/clip-vit-base-patch32",
        "model_type": "clip",
        "task": "multimodal",
    },
    "blip-vqa": {
        "model_name": "Salesforce/blip-vqa-base",
        "model_type": "blip",
        "task": "multimodal",
    },
    "llava": {
        "model_name": "llava-hf/llava-1.5-7b-hf",
        "model_type": "multimodal",
        "task": "multimodal",
    },
    "rnn-lm": {
        "model_name": "",
        "model_type": "rnn",
        "task": "text-generation",
    },
    "lstm-classifier": {
        "model_name": "",
        "model_type": "lstm",
        "task": "sequence-classification",
    },
    # === 🆕 From-scratch presets ===
    "scratch-transformer": {
        "model_name": "",
        "model_type": "scratch-transformer",
        "task": "language-modeling",
    },
    "scratch-encoder": {
        "model_name": "",
        "model_type": "scratch-transformer",
        "task": "encoder-only",
    },
    "scratch-cnn": {
        "model_name": "",
        "model_type": "scratch-cnn",
        "task": "image-classification",
    },
    "scratch-lstm": {
        "model_name": "",
        "model_type": "scratch-lstm",
        "task": "sequence-classification",
    },
    # === 🆕 MoE presets ===
    "mixtral-8x7b": {
        "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "model_type": "moe-finetune",
        "task": "text-generation",
    },
    "deepseek-moe": {
        "model_name": "deepseek-ai/deepseek-moe-16b-base",
        "model_type": "moe-finetune",
        "task": "text-generation",
    },
    "qwen2-moe": {
        "model_name": "Qwen/Qwen2-57B-A14B-Instruct",
        "model_type": "moe-finetune",
        "task": "text-generation",
    },
    "moe-scratch": {
        "model_name": "",
        "model_type": "moe-from-scratch",
        "task": "language-modeling",
    },
    # === 🆕 New Model Type Presets ===
    "whisper-tiny": {
        "model_name": "openai/whisper-tiny",
        "model_type": "whisper",
        "task": "speech-recognition",
    },
    "whisper-small": {
        "model_name": "openai/whisper-small",
        "model_type": "whisper",
        "task": "speech-recognition",
    },
    "sd-15": {
        "model_name": "runwayml/stable-diffusion-v1-5",
        "model_type": "diffusion",
        "task": "image-generation",
    },
    "sdxl": {
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "model_type": "diffusion",
        "task": "image-generation",
    },
    "flux-dev": {
        "model_name": "black-forest-labs/FLUX.1-dev",
        "model_type": "flux",
        "task": "image-generation",
    },
    "t5-small": {
        "model_name": "t5-small",
        "model_type": "t5",
        "task": "text2text-generation",
    },
    "t5-base": {
        "model_name": "t5-base",
        "model_type": "t5",
        "task": "text2text-generation",
    },
    "phi-2": {
        "model_name": "microsoft/phi-2",
        "model_type": "phi",
        "task": "text-generation",
    },
    "phi-3-mini": {
        "model_name": "microsoft/Phi-3-mini-4k-instruct",
        "model_type": "phi",
        "task": "text-generation",
    },
    "phi-4": {
        "model_name": "microsoft/Phi-4-mini-instruct",
        "model_type": "phi",
        "task": "text-generation",
    },
    "detr-resnet-50": {
        "model_name": "facebook/detr-resnet-50",
        "model_type": "detr",
        "task": "object-detection",
    },
    "bge-small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "model_type": "embedding",
        "task": "feature-extraction",
    },
    "bge-large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "model_type": "embedding",
        "task": "feature-extraction",
    },
    "sam-base": {
        "model_name": "facebook/sam-vit-base",
        "model_type": "sam",
        "task": "image-segmentation",
    },
    # === 🎬 Video Generation Models ===
    "svd-img2vid": {
        "model_name": "stabilityai/stable-video-diffusion-img2vid",
        "model_type": "video-diffusion",
        "task": "video-generation",
    },
    "svd-img2vid-xt": {
        "model_name": "stabilityai/stable-video-diffusion-img2vid-xt",
        "model_type": "video-diffusion",
        "task": "video-generation",
    },
    "i2vgen-xl": {
        "model_name": "ali-vilab/i2vgen-xl",
        "model_type": "i2vgen-xl",
        "task": "video-generation",
    },
    "film-frame-interp": {
        "model_name": "google/film-base",
        "model_type": "frame-interpolation",
        "task": "video-generation",
    },
}
