"""Pre-built pipeline templates for common training workflows."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class PipelineStep:
    """A single step in a training pipeline."""

    def __init__(
        self,
        node_type: str,
        label: str,
        config: Optional[Dict[str, Any]] = None,
        position: Optional[Dict[str, float]] = None,
    ) -> None:
        self.node_type = node_type
        self.label = label
        self.config = config or {}
        self.position = position or {"x": 0, "y": 0}


class PipelineTemplate:
    """A pre-built workflow pipeline template."""

    def __init__(
        self,
        name: str,
        description: str,
        category: str,
        steps: List[PipelineStep],
        icon: str = "📋",
    ) -> None:
        self.name = name
        self.description = description
        self.category = category
        self.steps = steps
        self.icon = icon

    def to_workflow(self) -> Dict[str, Any]:
        """Convert template to workflow JSON for the node editor."""
        nodes = []
        connections = []
        prev_id = None

        for i, step in enumerate(self.steps):
            node_id = f"{step.node_type}_{i}"
            pos = {
                "x": step.position.get("x", 100 + i * 250),
                "y": step.position.get("y", 200),
            }
            nodes.append({
                "id": node_id,
                "type": step.node_type,
                "position": pos,
                "config": step.config,
            })
            if prev_id:
                connections.append({
                    "id": f"conn_{prev_id}_{node_id}",
                    "source": prev_id,
                    "target": node_id,
                })
            prev_id = node_id

        return {"nodes": nodes, "connections": connections}


# --- Predefined Templates ---

LLM_FINETUNE = PipelineTemplate(
    name="LLM Fine-Tuning",
    description="完整的大语言模型微调流水线：加载模型 → 准备数据 → 训练 → 评估 → 保存",
    category="llm",
    icon="🤖",
    steps=[
        PipelineStep("load_model", "加载LLM", {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_type": "llm",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备数据", {
            "source": "dataset",
            "split": "train",
        }, {"x": 300, "y": 200}),
        PipelineStep("train", "训练", {
            "learning_rate": 5e-5, "batch_size": 8, "num_epochs": 3,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估", {
            "metric": "perplexity",
        }, {"x": 800, "y": 200}),
        PipelineStep("save", "保存模型", {
            "path": "./output/llm-finetuned",
        }, {"x": 1050, "y": 200}),
    ],
)

LORA_TUNE = PipelineTemplate(
    name="LoRA 高效微调",
    description="使用LoRA进行参数高效微调，大幅降低显存占用",
    category="lora",
    icon="🔗",
    steps=[
        PipelineStep("load_model", "加载基础模型", {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_type": "llm",
        }, {"x": 50, "y": 200}),
        PipelineStep("lora_config", "LoRA配置", {
            "r": 16, "alpha": 32, "dropout": 0.05,
        }, {"x": 300, "y": 150}),
        PipelineStep("dataset", "准备数据", {}, {"x": 300, "y": 300}),
        PipelineStep("train", "LoRA训练", {
            "learning_rate": 2e-4, "batch_size": 4, "num_epochs": 5,
        }, {"x": 550, "y": 200}),
        PipelineStep("save", "保存Adapter", {
            "path": "./output/lora-adapter",
        }, {"x": 800, "y": 200}),
    ],
)

QLORA_TUNE = PipelineTemplate(
    name="QLoRA 量化微调",
    description="4-bit量化 + LoRA，单卡即可微调大模型",
    category="qlora",
    icon="⚡",
    steps=[
        PipelineStep("load_model", "加载量化模型", {
            "model_name": "meta-llama/Meta-Llama-3-8B",
            "model_type": "qlora",
        }, {"x": 50, "y": 200}),
        PipelineStep("lora_config", "QLoRA配置", {
            "r": 16, "alpha": 32,
        }, {"x": 300, "y": 150}),
        PipelineStep("dataset", "准备数据", {}, {"x": 300, "y": 300}),
        PipelineStep("train", "QLoRA训练", {
            "learning_rate": 1e-4, "batch_size": 2, "num_epochs": 3,
        }, {"x": 550, "y": 200}),
        PipelineStep("save", "保存Adapter", {
            "path": "./output/qlora-adapter",
        }, {"x": 800, "y": 200}),
    ],
)

LCM_TUNE = PipelineTemplate(
    name="LCM 蒸馏训练",
    description="Latent Consistency Model蒸馏训练，加速图像生成",
    category="lcm",
    icon="🎨",
    steps=[
        PipelineStep("load_model", "加载LCM", {
            "model_name": "SimianLuo/LCM_Dreamshaper_v7",
            "model_type": "lcm",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备图像数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "LCM训练", {
            "learning_rate": 1e-5, "batch_size": 4, "num_epochs": 10,
        }, {"x": 550, "y": 200}),
        PipelineStep("save", "保存UNet", {
            "path": "./output/lcm-unet",
        }, {"x": 800, "y": 200}),
    ],
)

CNN_CLASSIFY = PipelineTemplate(
    name="CNN 图像分类",
    description="使用ResNet进行迁移学习图像分类",
    category="cnn",
    icon="🖼️",
    steps=[
        PipelineStep("load_model", "加载ResNet", {
            "model_name": "microsoft/resnet-50",
            "model_type": "cnn",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备图像数据", {
            "split": "train",
        }, {"x": 300, "y": 200}),
        PipelineStep("train", "CNN训练", {
            "learning_rate": 1e-4, "batch_size": 32, "num_epochs": 5,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估", {
            "metric": "accuracy",
        }, {"x": 800, "y": 200}),
    ],
)

ALL_TEMPLATES = [
    LLM_FINETUNE, LORA_TUNE, QLORA_TUNE, LCM_TUNE, CNN_CLASSIFY,
]
TEMPLATE_MAP = {t.name: t for t in ALL_TEMPLATES}


# === 🆕 New templates ===

MULTIMODAL_VLM = PipelineTemplate(
    name="多模态VLM训练",
    description="视觉-语言多模态模型训练：融合图像和文本进行理解与生成",
    category="multimodal",
    icon="🖼️",
    steps=[
        PipelineStep("load_model", "加载多模态模型", {
            "model_name": "openai/clip-vit-base-patch32",
            "model_type": "multimodal",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备图文数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "多模态训练", {
            "learning_rate": 5e-5, "batch_size": 8, "num_epochs": 5,
        }, {"x": 550, "y": 200}),
        PipelineStep("save", "保存模型", {
            "path": "./output/multimodal",
        }, {"x": 800, "y": 200}),
    ],
)

GPT_FINETUNE = PipelineTemplate(
    name="GPT微调",
    description="GPT系列模型微调（GPT-2, GPT-Neo, GPT-J等）",
    category="gpt",
    icon="🤖",
    steps=[
        PipelineStep("load_model", "加载GPT", {
            "model_name": "gpt2", "model_type": "gpt",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备文本数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "GPT训练", {
            "learning_rate": 5e-5, "batch_size": 4, "num_epochs": 3,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估", {
            "metric": "perplexity",
        }, {"x": 800, "y": 200}),
        PipelineStep("save", "保存模型", {
            "path": "./output/gpt-finetuned",
        }, {"x": 1050, "y": 200}),
    ],
)

BERT_CLASSIFY = PipelineTemplate(
    name="BERT文本分类",
    description="BERT微调用于文本分类、情感分析、NLI等NLU任务",
    category="bert",
    icon="📝",
    steps=[
        PipelineStep("load_model", "加载BERT", {
            "model_name": "bert-base-uncased", "model_type": "bert",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备文本数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "BERT微调", {
            "learning_rate": 2e-5, "batch_size": 16, "num_epochs": 3,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估", {
            "metric": "accuracy",
        }, {"x": 800, "y": 200}),
    ],
)

RNN_SEQUENCE = PipelineTemplate(
    name="RNN序列建模",
    description="循环神经网络用于序列分类和预测任务",
    category="rnn",
    icon="🔁",
    steps=[
        PipelineStep("load_model", "创建RNN", {
            "model_type": "rnn",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备序列数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "RNN训练", {
            "learning_rate": 1e-3, "batch_size": 32, "num_epochs": 10,
        }, {"x": 550, "y": 200}),
        PipelineStep("save", "保存模型", {
            "path": "./output/rnn-model",
        }, {"x": 800, "y": 200}),
    ],
)

LSTM_CLASSIFY = PipelineTemplate(
    name="LSTM文本分类",
    description="双向LSTM + Attention用于文本分类和序列标注",
    category="lstm",
    icon="🔣",
    steps=[
        PipelineStep("load_model", "创建LSTM", {
            "model_type": "lstm",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备文本数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "LSTM训练", {
            "learning_rate": 5e-4, "batch_size": 16, "num_epochs": 8,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估", {
            "metric": "accuracy",
        }, {"x": 800, "y": 200}),
    ],
)

# === 🆕 From-scratch templates ===

SCRATCH_TRANSFORMER = PipelineTemplate(
    name="从零训练Transformer",
    description="随机初始化一个Decoder-Only Transformer，从头训练语言模型",
    category="scratch",
    icon="🏗️",
    steps=[
        PipelineStep("load_model", "构建Transformer", {
            "model_type": "scratch-transformer",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "生成训练数据", {
            "source": "synthetic",
        }, {"x": 300, "y": 200}),
        PipelineStep("train", "从头训练", {
            "learning_rate": 3e-4, "batch_size": 32, "num_epochs": 10,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估困惑度", {
            "metric": "perplexity",
        }, {"x": 800, "y": 200}),
        PipelineStep("save", "保存模型", {
            "path": "./output/scratch-transformer",
        }, {"x": 1050, "y": 200}),
    ],
)

SCRATCH_CNN = PipelineTemplate(
    name="从零训练CNN",
    description="构建CNN卷积神经网络并从头训练图像分类",
    category="scratch",
    icon="🏗️",
    steps=[
        PipelineStep("load_model", "构建CNN", {
            "model_type": "scratch-cnn",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "生成图像数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "从头训练", {
            "learning_rate": 1e-3, "batch_size": 64, "num_epochs": 15,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估准确率", {
            "metric": "accuracy",
        }, {"x": 800, "y": 200}),
    ],
)

# === 🆕 TPU/NPU training templates ===

TPU_LLM_FINETUNE = PipelineTemplate(
    name="TPU LLM微调 (Google Cloud)",
    description="使用Google Cloud TPU进行大语言模型微调，torch_xla加速",
    category="tpu",
    icon="🟣",
    steps=[
        PipelineStep("load_model", "加载LLM (TPU)", {
            "model_name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "model_type": "llm",
        }, {"x": 50, "y": 200}),
        PipelineStep("lora_config", "LoRA配置", {
            "r": 16, "alpha": 32,
        }, {"x": 300, "y": 150}),
        PipelineStep("dataset", "准备数据", {}, {"x": 300, "y": 300}),
        PipelineStep("train", "TPU训练", {
            "learning_rate": 5e-5, "batch_size": 8, "num_epochs": 3,
        }, {"x": 550, "y": 200}),
        PipelineStep("save", "保存 TPU模型", {
            "path": "./output/tpu-llm",
        }, {"x": 800, "y": 200}),
    ],
)

NPU_SCRATCH = PipelineTemplate(
    name="NPU从零训练 (Ascend)",
    description="使用华为Ascend NPU从零训练Transformer，torch_npu加速",
    category="npu",
    icon="🟠",
    steps=[
        PipelineStep("load_model", "构建Transformer", {
            "model_type": "scratch-transformer",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "生成训练数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "NPU训练", {
            "learning_rate": 3e-4, "batch_size": 32, "num_epochs": 5,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估", {
            "metric": "perplexity",
        }, {"x": 800, "y": 200}),
    ],
)

# === 🆕 MoE templates ===

MOE_SCRATCH = PipelineTemplate(
    name="MoE从零训练",
    description="构建稀疏MoE Transformer（8专家，top-2路由），从零开始训练",
    category="moe",
    icon="🏗️",
    steps=[
        PipelineStep("load_model", "构建MoE", {
            "model_type": "moe-from-scratch",
            "n_experts": 8, "top_k": 2,
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "生成训练数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "MoE训练", {
            "learning_rate": 3e-4, "batch_size": 16, "num_epochs": 10,
        }, {"x": 550, "y": 200}),
        PipelineStep("evaluate", "评估困惑度", {
            "metric": "perplexity",
        }, {"x": 800, "y": 200}),
        PipelineStep("save", "保存 MoE", {
            "path": "./output/moe-scratch",
        }, {"x": 1050, "y": 200}),
    ],
)

MOE_FINETUNE = PipelineTemplate(
    name="MoE微调 (Mixtral/DeepSeek)",
    description="加载预训练MoE模型（Mixtral 8x7B、DeepSeek MoE等）进行微调",
    category="moe",
    icon="🤖",
    steps=[
        PipelineStep("load_model", "加载 MoE模型", {
            "model_name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "model_type": "moe-finetune",
        }, {"x": 50, "y": 200}),
        PipelineStep("dataset", "准备数据", {}, {"x": 300, "y": 200}),
        PipelineStep("train", "MoE微调", {
            "learning_rate": 2e-5, "batch_size": 4, "num_epochs": 3,
        }, {"x": 550, "y": 200}),
        PipelineStep("save", "保存微调MoE", {
            "path": "./output/moe-finetuned",
        }, {"x": 800, "y": 200}),
    ],
)

ALL_TEMPLATES = [
    LLM_FINETUNE, LORA_TUNE, QLORA_TUNE, LCM_TUNE, CNN_CLASSIFY,
    MULTIMODAL_VLM, GPT_FINETUNE, BERT_CLASSIFY, RNN_SEQUENCE, LSTM_CLASSIFY,
    SCRATCH_TRANSFORMER, SCRATCH_CNN,
    TPU_LLM_FINETUNE, NPU_SCRATCH,
    MOE_SCRATCH, MOE_FINETUNE,
]
TEMPLATE_MAP = {t.name: t for t in ALL_TEMPLATES}
