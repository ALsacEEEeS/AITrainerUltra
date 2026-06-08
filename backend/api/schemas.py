"""Pydantic schemas for API request/response validation with detailed error messages."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HyperParameterSchema(BaseModel):
    learning_rate: float = Field(
        default=5e-5,
        description="学习率. 建议: Adam=1e-5~5e-4, SGD=0.001~0.1",
        gt=0,
    )
    batch_size: int = Field(
        default=8,
        description="批次大小. MPS建议8-64, CUDA建议16-128, 大模型建议1-8",
        ge=1,
    )
    num_epochs: int = Field(
        default=3,
        description="训练轮数. 小数据集建议10-50, 大数据集建议1-5",
        ge=1,
    )
    warmup_steps: int = Field(
        default=100,
        description="预热步数. 通常为总步数的5-10%",
        ge=0,
    )
    weight_decay: float = Field(
        default=0.01,
        description="权重衰减(L2正则化). AdamW建议0.01-0.1, Adam建议0",
        ge=0,
    )
    fp16: bool = Field(
        default=False,
        description="启用混合精度训练(fp16). CUDA和MPS均支持, 可提速~50%并减少显存",
    )
    bf16: bool = Field(
        default=False,
        description="启用bfloat16. M3+/M4/M5推荐, TPU原生支持",
    )
    gradient_accumulation_steps: int = Field(
        default=1,
        description="梯度累积步数. 等效增大batch_size: 实际batch = batch_size * accumulation",
        ge=1,
    )
    max_grad_norm: float = Field(
        default=1.0,
        description="梯度裁剪阈值. 防止梯度爆炸, 建议0.5-5.0, 0=禁用",
        ge=0,
    )
    logging_steps: int = Field(
        default=10,
        description="每N步记录一次日志",
        ge=1,
    )
    save_steps: int = Field(
        default=500,
        description="每N步保存一次检查点",
        ge=1,
    )
    eval_steps: int = Field(
        default=500,
        description="每N步评估一次",
        ge=1,
    )


class DatasetSchema(BaseModel):
    path: str = Field(
        default="",
        description="数据集路径: HuggingFace ID (如 'imdb', 'wikitext') 或本地路径",
    )
    name: str = Field(
        default="",
        description="数据集子集名称 (如 'wikitext-2-raw-v1')",
    )
    split: str = Field(
        default="train",
        description="数据集划分: train | test | validation",
    )
    text_column: str = Field(
        default="text",
        description="文本列名 (HuggingFace数据集的列名)",
    )
    max_samples: Optional[int] = Field(
        default=None,
        description="最大样本数. None=使用全部, 小内存设备建议设500-5000",
        ge=1,
    )
    max_seq_length: int = Field(
        default=2048,
        description="最大序列长度( tokens). GPT/LLM建议512-8192, BERT建议128-512, M5支持128K",
        ge=1,
    )


class LoraParams(BaseModel):
    r: int = Field(default=16, description="LoRA秩. 8-64, 越大越强但越贵", ge=1, le=1024)
    alpha: int = Field(default=32, description="LoRA alpha缩放", ge=1)
    dropout: float = Field(default=0.05, description="LoRA dropout", ge=0, le=1)
    target_modules: Optional[List[str]] = Field(
        default=None,
        description="目标模块列表. 如 ['q_proj', 'v_proj'], None=自动检测",
    )


class QLoraParams(BaseModel):
    bits: int = Field(default=4, description="量化位数: 4或8", ge=2, le=8)
    double_quant: bool = Field(default=True, description="双重量化(进一步省显存)")
    quant_type: str = Field(
        default="nf4",
        description="量化类型: nf4 (推荐) | fp4",
    )


class TrainingJobRequest(BaseModel):
    model_type: str = Field(
        default="llm",
        description="模型类型: llm, gpt, bert, t5, phi, moe, cnn, rnn, lstm, lora, qlora, whisper, diffusion, flux, detr, sam, embedding, scratch-transformer, clip, blip, multimodal, lcm",
    )
    model_name: str = Field(
        default="",
        description="HuggingFace模型ID或本地路径. 如 'gpt2', 'bert-base-uncased', 'mistralai/Mistral-7B-v0.1', 'microsoft/phi-2', 'openai/whisper-tiny'",
    )
    output_dir: str = Field(default="./output", description="输出目录")
    task: str = Field(
        default="text-generation",
        description="任务类型: text-generation, sequence-classification, token-classification, image-classification, image-generation, speech-recognition, text2text-generation, feature-extraction, object-detection, image-segmentation, multimodal",
    )
    device_strategy: str = Field(
        default="auto",
        description="设备策略: auto, cuda, mps, rocm, tpu, npu, xpu, cpu. auto=自动选择最佳设备",
    )
    hyperparameters: HyperParameterSchema = Field(default_factory=HyperParameterSchema)
    dataset: DatasetSchema = Field(default_factory=DatasetSchema)
    lora: Optional[LoraParams] = None
    qlora: Optional[QLoraParams] = None
    scratch_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description="从零训练架构参数: vocab_size, d_model, n_layers, n_heads, d_ff, max_seq_len, dropout, num_classes",
    )

    @field_validator("model_type")
    @classmethod
    def validate_model_type(cls, v: str) -> str:
        allowed = {"llm", "gpt", "bert", "t5", "phi", "moe", "cnn", "rnn", "lstm",
                    "lora", "qlora", "scratch-transformer", "scratch-cnn", "scratch-lstm",
                    "clip", "blip", "multimodal", "lcm",
                    "whisper", "diffusion", "flux", "detr", "sam", "embedding",
                    "video-diffusion", "i2vgen-xl", "frame-interpolation"}
        if v not in allowed and not v.startswith("scratch-") and not v.startswith("moe-"):
            raise ValueError(
                f"不支持的模型类型 '{v}'. "
                f"请选择以下类型之一: {', '.join(sorted(allowed))}. "
                f"如需自定义类型, 请先在 registry 中注册"
            )
        return v

    @field_validator("task")
    @classmethod
    def validate_task(cls, v: str) -> str:
        allowed_tasks = {
            "text-generation", "sequence-classification", "token-classification",
            "image-classification", "image-generation", "video-generation",
            "speech-recognition", "text2text-generation", "feature-extraction",
            "object-detection", "image-segmentation", "multimodal",
            "language-modeling", "question-answering", "summarization", "translation",
        }
        if v not in allowed_tasks:
            raise ValueError(
                f"不支持的任务类型 '{v}'. 可选: {', '.join(sorted(allowed_tasks))}"
            )
        return v


class TrainingStatus(BaseModel):
    running: bool
    job_id: Optional[str] = None
    model_type: Optional[str] = None
    progress: Optional[float] = None
    error: Optional[str] = None
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None


class ModelInfo(BaseModel):
    name: str
    type: str
    description: str = ""
    loaded: bool = False
    error: Optional[str] = None


class ChatMessage(BaseModel):
    role: str = Field(default="user", description="消息角色: user | assistant | system")
    content: str = Field(default="", description="消息内容")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant", "system"):
            raise ValueError(f"无效角色 '{v}'. 可选: user, assistant, system")
        return v


class ChatRequest(BaseModel):
    model_id: str = Field(default="", description="模型ID或路径")
    messages: List[ChatMessage] = Field(
        default_factory=lambda: [ChatMessage(content="Hello")],
        description="对话消息列表",
    )
    max_tokens: int = Field(
        default=2048,
        description="最大生成token数. 短回答64-256, 长文1024-4096, M4/M5支持128K上下文",
        ge=1,
    )
    temperature: float = Field(
        default=0.7,
        description="生成温度. 0=确定性, 0.7=平衡, 1.0=创造性. 代码建议0.2, 创意写作建议0.8",
        ge=0,
        le=2,
    )
    top_p: float = Field(
        default=0.9,
        description="核采样top_p. 0.9=保留90%概率质量的token",
        ge=0,
        le=1,
    )
    top_k: int = Field(
        default=50,
        description="top-k采样. 仅从概率最高的K个token中采样, 0=禁用",
        ge=0,
    )
    repetition_penalty: float = Field(
        default=1.0,
        description="重复惩罚. 1.0=无惩罚, 1.1=轻微, 1.2=强力去重",
        ge=0,
    )
    device: str = Field(default="auto", description="推理设备: auto | cuda | mps | cpu")


class WorkflowNode(BaseModel):
    id: str = Field(default="", description="节点唯一ID")
    type: str = Field(default="", description="节点类型: load_model | dataset | train | evaluate | save | chat | lora_config")
    position: Dict[str, float] = Field(default_factory=dict, description="节点位置 (x, y)")
    config: Dict[str, Any] = Field(default_factory=dict, description="节点配置")


class WorkflowConnection(BaseModel):
    id: str = Field(default="", description="连接唯一ID")
    source: str = Field(default="", description="源节点ID")
    target: str = Field(default="", description="目标节点ID")
    source_handle: str = Field(default="output", description="源接口")
    target_handle: str = Field(default="input", description="目标接口")


class WorkflowRequest(BaseModel):
    nodes: List[WorkflowNode]
    connections: List[WorkflowConnection]


class ApiResponse(BaseModel):
    """统一API响应格式。所有接口返回此格式以便前端统一处理。"""
    success: bool = True
    message: str = ""
    data: Optional[Any] = None
    error_code: Optional[str] = Field(
        default=None,
        description="错误码. VALIDATION_ERROR | DEVICE_ERROR | MODEL_ERROR | DATA_ERROR | RUNNING_ERROR | NOT_FOUND",
    )
    error_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="错误详情. 包含字段级验证错误、设备信息、异常堆栈等",
    )
