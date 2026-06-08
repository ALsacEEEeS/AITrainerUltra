"""Training recipes — pre-configured hyperparameter presets for common scenarios.

Each recipe is a complete TrainingConfig tuned for a specific model + task + hardware combo.
All recipes include Apple Silicon (M1-M5) optimization recommendations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.core.config import TrainingConfig, HyperParameters, DatasetConfig


@dataclass
class Recipe:
    """A complete training recipe."""
    name: str
    description: str
    config: TrainingConfig
    hardware_recommendation: str = "CPU (8GB+)"
    estimated_time: str = "~5 min"
    tags: List[str] = field(default_factory=list)
    apple_silicon_note: str = ""  # Mac-specific optimization tip


# ─── LLM Recipes ───────────────────────────────────────────────────────

RECIPE_LLM_FULL = Recipe(
    name="LLM Full Fine-tune (Small)",
    description="Full fine-tune a small LLM on text data. Requires ~16GB VRAM.",
    config=TrainingConfig(
        model_type="llm",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        output_dir="./output/llm-full",
        hyperparameters=HyperParameters(
            learning_rate=5e-5, batch_size=4, num_epochs=3,
            warmup_steps=100, weight_decay=0.01, max_grad_norm=1.0,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 16GB+ / Apple M2+ (16GB+)",
    estimated_time="~30 min (on GPU)",
    tags=["llm", "full-finetune", "beginner"],
    apple_silicon_note="M2/M3推荐batch_size=4, M4/M5可提升到batch_size=8。启用fp16可获2x加速。",
)

RECIPE_LLM_LORA = Recipe(
    name="LLM LoRA Fine-tune",
    description="LoRA fine-tune on any LLM. VRAM efficient.",
    config=TrainingConfig(
        model_type="lora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        output_dir="./output/llm-lora",
        hyperparameters=HyperParameters(
            learning_rate=2e-4, batch_size=8, num_epochs=3,
            warmup_steps=50, weight_decay=0.01,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 8GB+ / Apple M1+ (8GB+)",
    estimated_time="~15 min (on GPU)",
    tags=["llm", "lora", "efficient"],
    apple_silicon_note="M1 8GB可运行, M4/M5建议batch_size=16。统一内存架构对LoRA特别友好。",
)

RECIPE_LLM_QLORA = Recipe(
    name="LLM QLoRA 4-bit",
    description="4-bit QLoRA on large LLMs. Runs on 8GB VRAM.",
    config=TrainingConfig(
        model_type="qlora",
        model_name="mistralai/Mistral-7B-v0.1",
        task="text-generation",
        output_dir="./output/llm-qlora",
        hyperparameters=HyperParameters(
            learning_rate=1e-4, batch_size=2, num_epochs=3,
            warmup_steps=50, weight_decay=0.01,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 8GB+ / Apple M1+ (8GB+)",
    estimated_time="~2 hours (on 8GB GPU)",
    tags=["llm", "qlora", "4-bit"],
    apple_silicon_note="M1 8GB可运行7B模型! 使用bitsandbytes-mlx-wheels获得原生MPS QLoRA支持。",
)


# ─── GPT Recipes ───────────────────────────────────────────────────────

RECIPE_GPT_FINETUNE = Recipe(
    name="GPT Fine-tune",
    description="Fine-tune GPT-2 on text generation.",
    config=TrainingConfig(
        model_type="gpt",
        model_name="gpt2",
        task="text-generation",
        output_dir="./output/gpt-finetune",
        hyperparameters=HyperParameters(
            learning_rate=5e-5, batch_size=4, num_epochs=3,
            warmup_steps=100, weight_decay=0.01,
        ),
    ),
    hardware_recommendation="CPU / Any GPU / Apple M1+",
    estimated_time="~10 min (on GPU)",
    tags=["gpt", "finetune"],
    apple_silicon_note="M1及以上均可流畅运行GPT-2 fine-tune。启用MPS加速。",
)


# ─── BERT Recipes ──────────────────────────────────────────────────────

RECIPE_BERT_CLASSIFY = Recipe(
    name="BERT Text Classification",
    description="Fine-tune BERT on text classification (IMDB).",
    config=TrainingConfig(
        model_type="bert",
        model_name="bert-base-uncased",
        task="sequence-classification",
        output_dir="./output/bert-classifier",
        hyperparameters=HyperParameters(
            learning_rate=2e-5, batch_size=16, num_epochs=3,
            warmup_steps=100, weight_decay=0.01,
        ),
    ),
    hardware_recommendation="CPU / Any GPU / Apple M1+",
    estimated_time="~15 min (on GPU)",
    tags=["bert", "classification", "nlp"],
    apple_silicon_note="M1 8GB即可运行, M4/M5支持batch_size=32。MPS加速效果显著。",
)


# ─── MoE Recipes ───────────────────────────────────────────────────────

RECIPE_MOE_SCRATCH = Recipe(
    name="MoE From Scratch",
    description="Train an 8-expert MoE transformer from scratch.",
    config=TrainingConfig(
        model_type="moe-from-scratch",
        task="language-modeling",
        output_dir="./output/moe-scratch",
        hyperparameters=HyperParameters(
            learning_rate=3e-4, batch_size=16, num_epochs=5,
            weight_decay=0.1, max_grad_norm=1.0,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 12GB+ / Apple M2+ (16GB+)",
    estimated_time="~30 min (on GPU)",
    tags=["moe", "scratch", "transformer"],
    apple_silicon_note="M2/M3推荐, M4/M5的GPU核心数多适合MoE并行训练。建议启用bf16。",
)

RECIPE_MOE_FINETUNE = Recipe(
    name="MoE Fine-tune (Mixtral)",
    description="Fine-tune Mixtral 8x7B Instruct.",
    config=TrainingConfig(
        model_type="moe-finetune",
        model_name="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        output_dir="./output/moe-finetune",
        hyperparameters=HyperParameters(
            learning_rate=2e-5, batch_size=1, num_epochs=2,
            warmup_steps=50, weight_decay=0.01,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 48GB+ / 8-bit mode / Apple M3+ (36GB+)",
    estimated_time="~several hours",
    tags=["moe", "mixtral", "8x7b"],
    apple_silicon_note="M3 36GB+可运行8-bit MoE。M4/M5 Ultra建议使用MLX框架加载。",
)


# ─── CNN Recipes ───────────────────────────────────────────────────────

RECIPE_CNN_TRANSFER = Recipe(
    name="CNN Transfer Learning",
    description="ResNet50 transfer learning on CIFAR-10.",
    config=TrainingConfig(
        model_type="cnn",
        model_name="microsoft/resnet-50",
        task="image-classification",
        output_dir="./output/cnn-transfer",
        hyperparameters=HyperParameters(
            learning_rate=1e-4, batch_size=32, num_epochs=5,
            weight_decay=0.01,
        ),
    ),
    hardware_recommendation="CPU / Any GPU / Apple M1+",
    estimated_time="~5 min (on GPU)",
    tags=["cnn", "transfer-learning", "vision"],
    apple_silicon_note="所有M系列均可流畅运行。M4/M5 ANE可加速图像预处理。",
)


# ─── Scratch Recipes ───────────────────────────────────────────────────

RECIPE_SCRATCH_TRANSFORMER = Recipe(
    name="Transformer From Scratch",
    description="Train a decoder-only transformer from random init.",
    config=TrainingConfig(
        model_type="scratch-transformer",
        task="language-modeling",
        output_dir="./output/scratch-transformer",
        hyperparameters=HyperParameters(
            learning_rate=3e-4, batch_size=32, num_epochs=10,
            weight_decay=0.1, max_grad_norm=1.0,
        ),
    ),
    hardware_recommendation="CPU (OK) / Any GPU / Apple M1+",
    estimated_time="~5 min (CPU)",
    tags=["scratch", "transformer", "beginner"],
    apple_silicon_note="M1及以上均可运行小规模scratch训练。M4/M5建议提升d_model=512。",
)


# ─── Multimodal Recipes ───────────────────────────────────────────────

RECIPE_CLIP_ZEROSHOT = Recipe(
    name="CLIP Zero-shot Classify",
    description="Load CLIP for zero-shot image classification.",
    config=TrainingConfig(
        model_type="clip",
        model_name="openai/clip-vit-base-patch32",
        task="multimodal",
        output_dir="./output/clip",
        hyperparameters=HyperParameters(
            learning_rate=5e-5, batch_size=8, num_epochs=1,
        ),
    ),
    hardware_recommendation="CPU / Any GPU (lightweight) / Apple M1+",
    estimated_time="~2 min",
    tags=["clip", "multimodal", "zero-shot"],
    apple_silicon_note="M1 8GB即可运行CLIP。M4/M5 ANE可加速图像编码。",
)


# ─── Advanced LLM Recipes ──────────────────────────────────────────────

RECIPE_LLM_DPO = Recipe(
    name="LLM DPO Training",
    description="Direct Preference Optimization — 用偏好数据对齐LLM。需要首选/非首选对。",
    config=TrainingConfig(
        model_type="llm",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        output_dir="./output/llm-dpo",
        hyperparameters=HyperParameters(
            learning_rate=1e-6, batch_size=4, num_epochs=1,
            warmup_steps=10, weight_decay=0.01, max_grad_norm=0.5,
            fp16=True,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 24GB+ / Apple M3+ (24GB+)",
    estimated_time="~1 hour (on GPU)",
    tags=["llm", "dpo", "alignment", "rlhf"],
    apple_silicon_note="M3 24GB+可运行1B模型DPO。M4/M5建议使用MLX实现。",
)

RECIPE_LLM_FULL_LARGE = Recipe(
    name="LLM Full Fine-tune (Large)",
    description="全参数微调大模型 7B+。需要48GB+显存或使用8-bit。",
    config=TrainingConfig(
        model_type="llm",
        model_name="mistralai/Mistral-7B-v0.1",
        task="text-generation",
        output_dir="./output/llm-full-large",
        hyperparameters=HyperParameters(
            learning_rate=2e-5, batch_size=1, num_epochs=3,
            warmup_steps=50, weight_decay=0.01, gradient_accumulation_steps=8,
            max_grad_norm=1.0, fp16=True,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 48GB+ / 8-bit模式 24GB+ / Apple M3+ (36GB+)",
    estimated_time="~3 hours (on 48GB GPU)",
    tags=["llm", "full-finetune", "large", "advanced"],
    apple_silicon_note="M3 Max 48GB+可运行7B full fine-tune。M4/M5建议使用QLoRA替代。",
)

RECIPE_LORA_LARGE = Recipe(
    name="LoRA Fine-tune (Large, r=64)",
    description="高秩LoRA微调大模型。r=64提供更强适配能力。",
    config=TrainingConfig(
        model_type="lora",
        model_name="meta-llama/Meta-Llama-3-8B",
        task="text-generation",
        output_dir="./output/lora-large",
        hyperparameters=HyperParameters(
            learning_rate=5e-4, batch_size=4, num_epochs=3,
            warmup_steps=100, weight_decay=0.01, fp16=True,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 16GB+ / Apple M2+ (16GB+)",
    estimated_time="~45 min (on 24GB GPU)",
    tags=["lora", "large", "efficient", "llm"],
    apple_silicon_note="M2 Pro 16GB+可运行7B LoRA。M4/M5推荐使用mlx-lm框架。",
)

# ─── Optimization-Focused Recipes ─────────────────────────────────────

RECIPE_FLASH_LLM = Recipe(
    name="Flash Attention LLM",
    description="启用Flash Attention的LLM微调。适合长序列(8K+)。",
    config=TrainingConfig(
        model_type="llm",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        output_dir="./output/flash-llm",
        hyperparameters=HyperParameters(
            learning_rate=5e-5, batch_size=4, num_epochs=3,
            warmup_steps=50, weight_decay=0.01, fp16=True,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 16GB+ (Turing+) / Apple M3+ (24GB+,使用SDPA)",
    estimated_time="~20 min (on GPU with Flash Attn)",
    tags=["llm", "flash-attention", "efficient", "long-context"],
    apple_silicon_note="M3/M4/M5支持PyTorch SDPA (iOS 17+), 虽然不是NVIDIA FlashAttn但同样有内存优化效果。",
)

RECIPE_QLORA_2BIT = Recipe(
    name="QLoRA 2-bit Extreme",
    description="2-bit QLoRA — 极限压缩。8GB显存可跑70B模型。",
    config=TrainingConfig(
        model_type="qlora",
        model_name="mistralai/Mistral-7B-v0.1",
        task="text-generation",
        output_dir="./output/qlora-2bit",
        hyperparameters=HyperParameters(
            learning_rate=5e-5, batch_size=2, num_epochs=3,
            warmup_steps=50, weight_decay=0.01,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 8GB+ / Apple M1+ (8GB+, 使用MLX量化)",
    estimated_time="~2 hours (on 8GB GPU)",
    tags=["qlora", "2-bit", "extreme", "low-vram"],
    apple_silicon_note="M1 8GB使用MLX 4-bit量化可运行13B模型。统一内存优势明显。",
)

# ─── New Task Recipes ──────────────────────────────────────────────────

RECIPE_GPT2_MEDIUM = Recipe(
    name="GPT-2 Medium Fine-tune",
    description="Fine-tune GPT-2 Medium (355M) on text generation.",
    config=TrainingConfig(
        model_type="gpt",
        model_name="gpt2-medium",
        task="text-generation",
        output_dir="./output/gpt2-medium",
        hyperparameters=HyperParameters(
            learning_rate=3e-5, batch_size=2, num_epochs=3,
            warmup_steps=100, weight_decay=0.01, fp16=True,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 8GB+ / Apple M1+ (8GB+)",
    estimated_time="~20 min (on GPU)",
    tags=["gpt2", "medium", "finetune"],
    apple_silicon_note="M1 8GB即可运行GPT-2 Medium。M4/M5支持batch_size=4+。",
)

RECIPE_VIT_FINETUNE = Recipe(
    name="Vision Transformer Fine-tune",
    description="Fine-tune ViT on image classification (CIFAR-100).",
    config=TrainingConfig(
        model_type="cnn",
        model_name="google/vit-base-patch16-224",
        task="image-classification",
        output_dir="./output/vit-finetune",
        hyperparameters=HyperParameters(
            learning_rate=2e-5, batch_size=32, num_epochs=5,
            warmup_steps=100, weight_decay=0.01,
        ),
    ),
    hardware_recommendation="CPU / Any GPU / Apple M1+ (lightweight)",
    estimated_time="~10 min (on GPU)",
    tags=["vit", "vision", "finetune", "image"],
    apple_silicon_note="M1及以上均可运行。M4/M5 ANE可加速ViT推理。",
)

RECIPE_SD_LORA = Recipe(
    name="Stable Diffusion LoRA",
    description="LoRA fine-tune for Stable Diffusion (text-to-image). 需要diffusers。",
    config=TrainingConfig(
        model_type="diffusion",
        model_name="runwayml/stable-diffusion-v1-5",
        task="image-generation",
        output_dir="./output/sd-lora",
        hyperparameters=HyperParameters(
            learning_rate=1e-4, batch_size=1, num_epochs=10,
            warmup_steps=50, weight_decay=0.01,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 12GB+ / Apple M2+ (16GB+)",
    estimated_time="~30 min (on 24GB GPU)",
    tags=["sd", "lora", "image-generation", "diffusion"],
    apple_silicon_note="M2 16GB可运行SD LoRA。M4/M5支持更大的UNet batch。使用mps或diffusers。",
)

RECIPE_MOE_SPARSE = Recipe(
    name="MoE Sparse Training",
    description="从零训练MoE Transformer。8专家top-2路由。",
    config=TrainingConfig(
        model_type="moe-from-scratch",
        task="language-modeling",
        output_dir="./output/moe-sparse",
        hyperparameters=HyperParameters(
            learning_rate=3e-4, batch_size=32, num_epochs=10,
            weight_decay=0.1, max_grad_norm=1.0, warmup_steps=200,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 16GB+ / Apple M3+ (24GB+)",
    estimated_time="~1 hour (on GPU)",
    tags=["moe", "scratch", "sparse", "transformer"],
    apple_silicon_note="M3/M4 GPU核心数越多性能越好。M4 Pro 20核GPU推荐。",
)

# ─── Apple Silicon Exclusive Recipes ────────────────────────────────────

RECIPE_MPS_LLM = Recipe(
    name="MPS LLM Fine-tune (Apple Optimized)",
    description="专为Apple Silicon优化的LLM微调。使用MPS后端和统一内存。",
    config=TrainingConfig(
        model_type="llm",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        output_dir="./output/mps-llm",
        hyperparameters=HyperParameters(
            learning_rate=3e-5, batch_size=8, num_epochs=3,
            warmup_steps=100, weight_decay=0.01, fp16=True,
            gradient_accumulation_steps=2,
        ),
    ),
    hardware_recommendation="Apple M1+ (推荐 M4/M5)",
    estimated_time="~20 min (M4/M5)",
    tags=["llm", "mps", "apple", "optimized"],
    apple_silicon_note="专为M系列优化! 使用MPS后端+fp16混合精度。M4/M5支持batch_size=16, 128K上下文。",
)

RECIPE_M4_MULTIMODAL = Recipe(
    name="M4/M5 Multimodal Vision-Language",
    description="Apple Silicon M4/M5优化的多模态VLM训练。利用统一内存处理图像+文本。",
    config=TrainingConfig(
        model_type="multimodal",
        model_name="llava-hf/llava-1.5-7b-hf",
        task="multimodal",
        output_dir="./output/m4-vlm",
        hyperparameters=HyperParameters(
            learning_rate=2e-5, batch_size=2, num_epochs=2,
            warmup_steps=50, weight_decay=0.01, bf16=True,
        ),
    ),
    hardware_recommendation="Apple M4/M5 (16GB+)",
    estimated_time="~2 hours (M4 Max)",
    tags=["multimodal", "vlm", "m4", "m5", "apple"],
    apple_silicon_note="M4/M5的Neural Engine可加速图像编码。bf16原生支持! 统一内存免去显存拷贝。",
)

RECIPE_MPS_LORA_VL = Recipe(
    name="MPS LoRA + Vision-Language",
    description="Apple Silicon LoRA多模态。极低显存占用。",
    config=TrainingConfig(
        model_type="lora",
        model_name="llava-hf/llava-1.5-7b-hf",
        task="multimodal",
        output_dir="./output/mps-lora-vl",
        hyperparameters=HyperParameters(
            learning_rate=2e-4, batch_size=4, num_epochs=3,
            warmup_steps=50, weight_decay=0.01, fp16=True,
        ),
    ),
    hardware_recommendation="Apple M1+ (8GB+, 推荐 M4/M5)",
    estimated_time="~1 hour (M4)",
    tags=["lora", "multimodal", "mps", "apple", "efficient"],
    apple_silicon_note="M1 8GB也可运行! 统一内存可同时加载视觉和语言模型。M4/M5推荐r=32。",
)


# ─── Video Generation Recipes ──────────────────────────────────────────

RECIPE_SVD_IMG2VID = Recipe(
    name="Stable Video Diffusion (img2vid)",
    description="将静态图像转换为短视频。支持帧数和分辨率配置。",
    config=TrainingConfig(
        model_type="video-diffusion",
        model_name="stabilityai/stable-video-diffusion-img2vid",
        task="video-generation",
        output_dir="./output/svd",
        hyperparameters=HyperParameters(
            learning_rate=1e-5, batch_size=1, num_epochs=1,
            fp16=True,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 24GB+ / Apple M3+ (36GB+)",
    estimated_time="~2 min per video",
    tags=["video", "diffusion", "generation", "svd"],
    apple_silicon_note="M3 Max 36GB+可运行。M4/M5 Ultra推荐使用MLX优化。",
)

RECIPE_I2VGEN = Recipe(
    name="I2VGen-XL 文生视频",
    description="阿里I2VGen-XL — 从文字描述或图像生成短视频。",
    config=TrainingConfig(
        model_type="i2vgen-xl",
        model_name="ali-vilab/i2vgen-xl",
        task="video-generation",
        output_dir="./output/i2vgen",
        hyperparameters=HyperParameters(
            learning_rate=5e-6, batch_size=1, num_epochs=1,
            fp16=True,
        ),
    ),
    hardware_recommendation="NVIDIA GPU 32GB+ / A100 / H100",
    estimated_time="~5 min per video",
    tags=["video", "generation", "i2vgen", "chinese"],
    apple_silicon_note="需要大显存。M4 Ultra 64GB+建议使用MLX。",
)

RECIPE_FRAME_INTERP = Recipe(
    name="Frame Interpolation (插帧)",
    description="视频帧插值 — 提升帧率, 生成慢动作。支持FILM和RIFE。",
    config=TrainingConfig(
        model_type="frame-interpolation",
        model_name="google/film-base",
        task="video-generation",
        output_dir="./output/frame-interp",
        hyperparameters=HyperParameters(
            learning_rate=1e-4, batch_size=4, num_epochs=5,
        ),
    ),
    hardware_recommendation="CPU / Any GPU / Apple M1+",
    estimated_time="~1 min per video",
    tags=["video", "interpolation", "frame", "film"],
    apple_silicon_note="M1及以上均可运行帧插值。ANE可加速光流计算。",
)


# ─── All Recipes ───────────────────────────────────────────────────────

ALL_RECIPES: List[Recipe] = [
    RECIPE_LLM_FULL, RECIPE_LLM_FULL_LARGE, RECIPE_LLM_LORA, RECIPE_LLM_QLORA,
    RECIPE_LLM_DPO, RECIPE_LORA_LARGE,
    RECIPE_GPT_FINETUNE, RECIPE_GPT2_MEDIUM,
    RECIPE_BERT_CLASSIFY,
    RECIPE_MOE_SCRATCH, RECIPE_MOE_SPARSE, RECIPE_MOE_FINETUNE,
    RECIPE_CNN_TRANSFER, RECIPE_VIT_FINETUNE,
    RECIPE_SCRATCH_TRANSFORMER,
    RECIPE_CLIP_ZEROSHOT, RECIPE_SD_LORA,
    RECIPE_FLASH_LLM, RECIPE_QLORA_2BIT,
    RECIPE_MPS_LLM, RECIPE_M4_MULTIMODAL, RECIPE_MPS_LORA_VL,
    RECIPE_SVD_IMG2VID, RECIPE_I2VGEN, RECIPE_FRAME_INTERP,
]

RECIPE_MAP: Dict[str, Recipe] = {r.name: r for r in ALL_RECIPES}


def get_recipe(name: str) -> Optional[Recipe]:
    return RECIPE_MAP.get(name)


def list_recipes_by_hardware(hardware: str = "") -> List[Recipe]:
    """Filter recipes by hardware capability."""
    if not hardware:
        return ALL_RECIPES
    hw = hardware.lower()
    return [r for r in ALL_RECIPES if hw in r.hardware_recommendation.lower()]


def list_recipes_by_tag(tag: str) -> List[Recipe]:
    """Filter recipes by tag."""
    return [r for r in ALL_RECIPES if tag in r.tags]
