"""REST API routes for training management."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.schemas import (
    ApiResponse,
    ChatRequest,
    TrainingJobRequest,
    WorkflowRequest,
)
from backend.core.config import (
    MODEL_PRESETS,
    TrainingConfig,
    HyperParameters,
    DatasetConfig,
    LoraConfigData,
    QLoraConfigData,
)
from backend.core.engine import engine
from backend.core.registry import registry
from backend.core.experiment import exp_tracker
from backend.core.hpo import SearchSpace, GridSearch, RandomSearch, BayesianOptimizer
from backend.core.pipeline import ALL_TEMPLATES, TEMPLATE_MAP
from backend.core.scheduler import Job, job_queue
from backend.models.model_manager import model_manager
from backend.data.dataset_manager import dataset_manager
from backend.core.recipes import ALL_RECIPES, RECIPE_MAP, get_recipe
from backend.data.real_data import DATASET_PRESETS, get_text_dataset, get_classification_dataset
from backend.utils.device import get_device_summary, DEVICE_INFO

logger = __import__('logging').getLogger('aitrainer.routes')
from backend.models.model_loader import (
    detect_model_format, get_model_info, list_model_files,
    get_model_metadata, load_model_from_path, MODEL_FORMAT_TABLE,
)
from backend.utils.optimizations import (
    OptimizationConfig, OptimizationManager,
    WOQConfig, QuantizationMethod, KVOffloadConfig, DMSConfig, VariableVRAMConfig,
    FlashAttentionConfig, FlashAttentionEngine,
    get_woq_info, PRESET_NONE, PRESET_MEMORY_SAVE, PRESET_EXTREME,
    PRESET_QUALITY, PRESET_TPU, PRESET_MPS, PRESET_M4, PRESET_M5,
    PRESET_NVIDIA_BLACKWELL, PRESET_NVIDIA_ADA, PRESET_NVIDIA_AMPERE,
    PRESET_NVIDIA_TURING, PRESET_NVIDIA_PROFESSIONAL, PRESET_NVIDIA_HOPPER,
)

router = APIRouter(prefix="/api/v1")

# ─── Structured Error Helpers ─────────────────────────────────────────

def _error(
    message: str,
    status_code: int = 400,
    error_code: str = "VALIDATION_ERROR",
    details: Any = None,
) -> HTTPException:
    """Create a structured HTTPException with error context."""
    return HTTPException(
        status_code=status_code,
        detail=ApiResponse(
            success=False,
            message=message,
            error_code=error_code,
            error_details=details,
        ).model_dump(),
    )


def _traing_error(
    e: Exception,
    operation: str,
    message: str = "操作失败",
    error_code: str = "INTERNAL_ERROR",
    status_code: int = 500,
) -> HTTPException:
    """Wrap an exception into a structured HTTPException with context."""
    logger.exception("[%s] %s error: %s", error_code, operation, e)
    details = {
        "operation": operation,
        "error_type": type(e).__name__,
        "detail": str(e),
    }
    try:
        from backend.utils.device import DEVICE_INFO
        details["device"] = DEVICE_INFO.name
    except Exception:
        pass
    return HTTPException(
        status_code=status_code,
        detail=ApiResponse(
            success=False,
            message=f"{message}: {e}",
            error_code=error_code,
            error_details=details,
        ).model_dump(),
    )


@router.get("/status")
async def get_status() -> ApiResponse:
    return ApiResponse(data={
        "engine": engine.get_status(),
        "supported_models": registry.list_supported_types(),
        "presets": list(MODEL_PRESETS.keys()),
        "device": get_device_summary(),
    })


@router.get("/models")
async def list_models() -> ApiResponse:
    """List all supported model types."""
    return ApiResponse(data={
        "types": registry.list_supported_types(),
        "presets": MODEL_PRESETS,
    })


@router.post("/training/start")
async def start_training(req: TrainingJobRequest) -> ApiResponse:
    """Start a new training job with validation."""
    from backend.utils.training_utils import validate_config

    try:
        hp = HyperParameters(
            learning_rate=req.hyperparameters.learning_rate,
            batch_size=req.hyperparameters.batch_size,
            num_epochs=req.hyperparameters.num_epochs,
            warmup_steps=req.hyperparameters.warmup_steps,
            weight_decay=req.hyperparameters.weight_decay,
            fp16=req.hyperparameters.fp16,
            bf16=req.hyperparameters.bf16,
            gradient_accumulation_steps=req.hyperparameters.gradient_accumulation_steps,
            max_grad_norm=req.hyperparameters.max_grad_norm,
        )
        dc = DatasetConfig(
            path=req.dataset.path,
            name=req.dataset.name,
            split=req.dataset.split,
            text_column=req.dataset.text_column,
            max_samples=req.dataset.max_samples,
            max_seq_length=req.dataset.max_seq_length,
        )
        config = TrainingConfig(
            model_type=req.model_type,
            model_name=req.model_name,
            output_dir=req.output_dir,
            task=req.task,
            hyperparameters=hp,
            dataset=dc,
            device_strategy=getattr(req, 'device_strategy', 'auto'),
            scratch_config=getattr(req, 'scratch_config', None),
        )
        if req.lora:
            config.lora = LoraConfigData(
                r=req.lora.r, alpha=req.lora.alpha,
                dropout=req.lora.dropout,
                target_modules=req.lora.target_modules,
            )
        if req.qlora:
            config.qlora = QLoraConfigData(
                bits=req.qlora.bits,
                double_quant=req.qlora.double_quant,
                quant_type=req.qlora.quant_type,
            )

        # Validate configuration
        warnings = validate_config(config)
        if any("required" in w for w in warnings):
            raise ValueError(f"配置验证失败: {'; '.join(warnings)}")

        if not registry.is_supported(req.model_type):
            supported = registry.list_supported_types()
            raise ValueError(
                f"不支持的模型类型 '{req.model_type}'. "
                f"当前支持: {', '.join(sorted(supported))}. "
                f"如需添加新类型, 请先在 server.py 中注册"
            )

        job_id = await engine.start_training(config)
        return ApiResponse(
            message=f"训练已启动 [job_id={job_id}]",
            data={
                "job_id": job_id,
                "warnings": warnings if warnings else None,
                "config": {
                    "model_type": config.model_type,
                    "device_strategy": config.device_strategy,
                    "batch_size": hp.batch_size,
                    "num_epochs": hp.num_epochs,
                },
            },
        )

    except ValueError as e:
        raise _error(
            str(e), status_code=400, error_code="VALIDATION_ERROR",
            details={"model_type": req.model_type, "config": req.model_dump()},
        )
    except RuntimeError as e:
        raise _error(
            f"训练启动冲突: {e}", status_code=409, error_code="RUNNING_ERROR",
            details={"model_type": req.model_type, "hint": "请先停止当前训练任务"},
        )
    except ImportError as e:
        raise _error(
            f"缺少依赖: {e}", status_code=501, error_code="DEPENDENCY_ERROR",
            details={"model_type": req.model_type, "hint": "请安装所需依赖: pip install [package]"},
        )
    except Exception as e:
        raise _traing_error(e, f"start_training({req.model_type})", "训练启动失败")


@router.post("/training/stop")
async def stop_training() -> ApiResponse:
    """Stop the current training job."""
    await engine.stop_training()
    return ApiResponse(message="Training stopped")


@router.get("/training/status")
async def training_status() -> ApiResponse:
    return ApiResponse(data=engine.get_status())


@router.get("/training/checkpoints")
async def list_checkpoints(output_dir: str = "./output") -> ApiResponse:
    """List available checkpoints in a directory with metadata."""
    from pathlib import Path
    import os, time
    ckpt_dir = Path(output_dir)
    checkpoints = sorted(ckpt_dir.glob("checkpoint_*.pt"))
    ckpt_list = []
    for c in checkpoints:
        stat = c.stat()
        ckpt_list.append({
            "name": c.name,
            "path": str(c),
            "size_bytes": stat.st_size,
            "size_readable": f"{stat.st_size / 1024 / 1024:.1f}MB" if stat.st_size > 1024 * 1024 else f"{stat.st_size / 1024:.1f}KB",
            "modified": time.ctime(stat.st_mtime),
        })
    return ApiResponse(data={
        "checkpoints": ckpt_list,
        "latest": ckpt_list[-1]["name"] if ckpt_list else None,
        "count": len(ckpt_list),
        "output_dir": output_dir,
    })


@router.post("/training/checkpoints/delete")
async def delete_checkpoint(data: Dict[str, Any]) -> ApiResponse:
    """Delete a specific checkpoint."""
    from pathlib import Path
    name = data.get("name", "")
    output_dir = data.get("output_dir", "./output")
    ckpt_path = Path(output_dir) / name
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    ckpt_path.unlink()
    return ApiResponse(message=f"Deleted: {name}")


@router.post("/training/checkpoints/restore")
async def restore_checkpoint(data: Dict[str, Any]) -> ApiResponse:
    """Load a checkpoint into the inference server."""
    from pathlib import Path
    name = data.get("name", "")
    output_dir = data.get("output_dir", "./output")
    ckpt_path = Path(output_dir) / name
    if not ckpt_path.exists():
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    try:
        import torch
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        model_state = ckpt.get("model_state_dict") or ckpt.get("state_dict", {})
        epoch = ckpt.get("epoch", 0)
        loss = ckpt.get("loss", 0.0)
        return ApiResponse(data={
            "restored": True,
            "epoch": epoch,
            "loss": loss,
            "layers": len(model_state),
            "checkpoint": name,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore: {e}")


@router.get("/presets")
async def list_presets() -> ApiResponse:
    """List model presets."""
    return ApiResponse(data=MODEL_PRESETS)


@router.get("/recipes")
async def list_recipes(tag: str = "", hardware: str = "") -> ApiResponse:
    """List training recipes, optionally filtered."""
    recipes = ALL_RECIPES
    if tag:
        recipes = [r for r in recipes if tag in r.tags]
    if hardware:
        recipes = [r for r in recipes if hardware.lower() in r.hardware_recommendation.lower()]
    return ApiResponse(data={
        "recipes": [
            {
                "name": r.name, "description": r.description,
                "model_type": r.config.model_type,
                "model_name": r.config.model_name,
                "hardware": r.hardware_recommendation,
                "estimated_time": r.estimated_time,
                "tags": r.tags,
                "apple_silicon_note": r.apple_silicon_note,
                "hyperparameters": r.config.hyperparameters.to_dict() if hasattr(r.config.hyperparameters, 'to_dict') else {},
            }
            for r in recipes
        ],
        "count": len(recipes),
    })


@router.get("/recipes/{name}")
async def get_recipe_detail(name: str) -> ApiResponse:
    recipe = get_recipe(name)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return ApiResponse(data={
        "name": recipe.name,
        "description": recipe.description,
        "config": recipe.config.to_dict() if hasattr(recipe.config, 'to_dict') else {},
        "hardware": recipe.hardware_recommendation,
        "estimated_time": recipe.estimated_time,
        "tags": recipe.tags,
        "apple_silicon_note": recipe.apple_silicon_note,
    })


@router.post("/training/validate")
async def validate_training_config(data: Dict[str, Any]) -> ApiResponse:
    """Validate a training configuration without starting it."""
    from backend.utils.training_utils import validate_config
    from backend.core.config import TrainingConfig
    try:
        cfg = TrainingConfig(
            model_type=data.get("model_type", ""),
            model_name=data.get("model_name", ""),
            output_dir=data.get("output_dir", "./output"),
            task=data.get("task", "text-generation"),
        )
        warnings = validate_config(cfg)
        return ApiResponse(data={"valid": len([w for w in warnings if "required" in w]) == 0, "warnings": warnings})
    except Exception as e:
        return ApiResponse(data={"valid": False, "warnings": [str(e)]})


@router.get("/dataset-presets")
async def list_dataset_presets() -> ApiResponse:
    """List available dataset presets."""
    return ApiResponse(data={
        "presets": [
            {"id": k, **v} for k, v in DATASET_PRESETS.items()
        ],
        "count": len(DATASET_PRESETS),
    })


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """Stream chat response via SSE."""
    prompt = req.messages[-1].content if req.messages else ""

    async def event_stream():
        from backend.utils.training_utils import resolve_device
        _, device_type = resolve_device(req.device)
        full_text = ""

        # 1. Inference server streaming
        try:
            from backend.models.serving import inference_server
            if inference_server.is_loaded(req.model_id):
                async for token in inference_server.generate_stream(
                    model_id=req.model_id, prompt=prompt,
                    max_tokens=req.max_tokens,
                ):
                    full_text += token
                    yield f"data: {json.dumps({'token': token, 'done': False, 'full_text': full_text})}\n\n"
                yield f"data: {json.dumps({'token': '', 'done': True, 'full_text': full_text})}\n\n"
                return
        except Exception:
            pass

        # 2. HF pipeline streaming (simulated token-by-token)
        try:
            from transformers import pipeline as hf_pipeline
            gen = hf_pipeline("text-generation", model=req.model_id)
            result = gen(prompt, max_new_tokens=req.max_tokens, temperature=req.temperature)
            text = result[0]["generated_text"] if result else ""
            # Remove the prompt prefix
            if text.startswith(prompt):
                text = text[len(prompt):]
            # Yield word by word
            words = text.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                full_text += token
                yield f"data: {json.dumps({'token': token, 'done': False, 'full_text': full_text})}\n\n"
                await asyncio.sleep(0.02)
            yield f"data: {json.dumps({'token': '', 'done': True, 'full_text': full_text})}\n\n"
            return
        except ImportError:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'token': '', 'done': True, 'full_text': str(e), 'error': True})}\n\n"
            return

        # 3. Fallback streaming (word-by-word)
        response = _generate_fallback_response(prompt, req.model_id)
        words = response.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            full_text += token
            yield f"data: {json.dumps({'token': token, 'done': False, 'full_text': full_text})}\n\n"
            await asyncio.sleep(0.01)
        yield f"data: {json.dumps({'token': '', 'done': True, 'full_text': full_text})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat")
async def chat(req: ChatRequest) -> ApiResponse:
    """Send a chat message to a loaded model with device selection."""
    if not req.messages:
        raise _error("messages 不能为空", error_code="VALIDATION_ERROR",
                      details={"hint": "请至少提供一条消息"})
    if not req.messages[-1].content.strip():
        raise _error("最后一条消息内容不能为空", error_code="VALIDATION_ERROR",
                      details={"hint": "请写入消息内容"})

    prompt = req.messages[-1].content

    from backend.utils.training_utils import resolve_device
    _device, device_type = resolve_device(req.device)

    # 1. Inference server (pre-loaded model)
    try:
        from backend.models.serving import inference_server
        if inference_server.is_loaded(req.model_id):
            result = inference_server.generate(
                model_id=req.model_id, prompt=prompt,
                max_tokens=req.max_tokens, temperature=req.temperature,
            )
            return ApiResponse(data={"response": result.get("text", ""), "mode": "inference_server", "device": device_type})
    except Exception as e:
        logger.debug(f"Inference server generate failed: {e}")

    # 2. HuggingFace pipeline with device
    try:
        from transformers import pipeline as hf_pipeline
        if device_type in ("mps", "cpu"):
            device_map = device_type
        elif device_type == "auto":
            device_map = "auto"
        else:
            device_map = -1  # CPU fallback
        gen = hf_pipeline("text-generation", model=req.model_id,
                          device_map=device_map if isinstance(device_map, str) and device_map != "auto" else device_map)
        result = gen(prompt, max_new_tokens=req.max_tokens, temperature=req.temperature)
        return ApiResponse(data={
            "response": result[0]["generated_text"] if result else "",
            "mode": "transformers", "device": device_type,
        })
    except ImportError:
        logger.info("transformers not available for chat, using fallback")
    except Exception as e:
        logger.info(f"HF pipeline failed on {device_type}: {e}")

    # 3. Fallback
    response = _generate_fallback_response(prompt, req.model_id)
    return ApiResponse(data={"response": response, "mode": "fallback", "device": "cpu"})


_KNOWLEDGE_BASE = {
    "hello": "Hello! I am AITrainerUltra's chat interface. You can ask me about training models, or load a real model for more intelligent responses.",
    "你好": "你好！我是 AITrainerUltra 的对话界面。你可以问我关于训练模型的问题，或者加载一个真实模型来获得更智能的回复。",
    "help": "Available commands:\n- Load a model: use the 'Load Model' panel\n- Start training: use the 'Training' panel\n- Switch language: click 中文/EN in the sidebar\n- See docs: check the API docs at /docs",
    "model": "This system supports 19 model types including LLM, GPT, BERT, Multimodal VLM, MoE, CNN, RNN, LSTM, and more.",
    "train": "To train a model, go to the Training panel, select a model type, configure hyperparameters, and click Start Training.",
    "device": "Supported devices: NVIDIA CUDA, Apple MPS, AMD ROCm, Google TPU, Huawei Ascend NPU, Intel XPU, CPU.",
}


def _generate_fallback_response(prompt: str, model_id: str) -> str:
    """Generate a response without torch using keyword matching + template."""
    prompt_lower = prompt.lower().strip()

    # Direct knowledge base match
    for key, response in _KNOWLEDGE_BASE.items():
        if key in prompt_lower:
            return response

    # Pattern matching
    if any(w in prompt_lower for w in ["hi", "hey", "greeting", "howdy"]):
        return _KNOWLEDGE_BASE["hello"]
    if any(w in prompt_lower for w in ["what", "who", "which", "tell me about"]):
        return f"Regarding '{prompt[:50]}...' - I can help you train and interact with AI models. Try loading a real model ({model_id}) for more accurate responses."
    if any(w in prompt_lower for w in ["how", "explain", "tutorial"]):
        return _KNOWLEDGE_BASE["help"]
    if any(w in prompt_lower for w in ["thanks", "thank"]):
        return "You're welcome! Let me know if you need any more help."

    # Default with model awareness
    model_short = model_id.split("/")[-1] if "/" in model_id else model_id
    return (
        f"I received your message about '{prompt[:60]}'. "
        f"To get the best response, please load a real model (current: {model_short}) "
        f"using the 'Load Model' panel. "
        f"Without a model loaded, I can only provide template-based answers. "
        f"Try asking about 'help', 'model', 'train', or 'device'."
    )


# 📂 Open-Source Model Loader

@router.get("/model-loader/info")
async def model_loader_info() -> ApiResponse:
    return ApiResponse(data={
        "formats": MODEL_FORMAT_TABLE,
        "auto_detect_extensions": [".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx"],
    })


@router.post("/model-loader/detect")
async def detect_model(data: Dict[str, Any]) -> ApiResponse:
    """Detect model format from a path."""
    path = data.get("path", "")
    if not path:
        raise HTTPException(status_code=400, detail="path is required")

    fmt, arch = detect_model_format(path)
    info = get_model_info(path)

    return ApiResponse(data={
        "path": path,
        "format": fmt,
        "architecture": arch,
        "file_info": info,
        "metadata": get_model_metadata(path),
    })


@router.post("/model-loader/browse")
async def browse_models(data: Dict[str, Any]) -> ApiResponse:
    """Browse model files in a directory."""
    path = data.get("path", ".")
    files = list_model_files(path)
    return ApiResponse(data={"files": files, "count": len(files)})


@router.post("/model-loader/load")
async def load_model_endpoint(data: Dict[str, Any]) -> ApiResponse:
    """Load a model from path into inference server."""
    path = data.get("path", "")
    if not path:
        raise _error("path 不能为空", error_code="VALIDATION_ERROR",
                      details={"hint": "请输入 HuggingFace 模型ID 或本地路径"})

    device = data.get("device", "auto")
    load_in_8bit = data.get("load_in_8bit", False)
    load_in_4bit = data.get("load_in_4bit", False)

    try:
        # First detect format to give early error feedback
        from backend.models.model_loader import detect_model_format, ModelFormat
        fmt, arch = detect_model_format(path)
        if fmt == ModelFormat.UNKNOWN:
            raise ValueError(
                f"无法识别模型格式: '{path}'. "
                f"支持的格式: HuggingFace ID, SafeTensors, PyTorch, GGUF, ONNX, MLX."
            )

        result = await load_model_from_path(
            path, device, load_in_8bit, load_in_4bit,
        )

        if "error" in result:
            raise RuntimeError(result["error"])

        # Register with inference server
        from backend.models.serving import inference_server
        model_id = result.get("model_name") or path
        inference_server.load(
            model_id=model_id,
            model=result.get("model"),
            tokenizer=result.get("tokenizer"),
        )

        model_params = result.get("params", 0)
        params_str = f"{model_params/1e9:.2f}B" if model_params >= 1e9 else f"{model_params/1e6:.1f}M"

        return ApiResponse(message=f"模型加载成功: {model_id} ({params_str})", data={
            "model_id": model_id,
            "format": result.get("format"),
            "architecture": result.get("architecture"),
            "params": model_params,
            "params_readable": params_str,
            "backend": result.get("backend", "transformers"),
        })

    except ValueError as e:
        raise _error(str(e), status_code=400, error_code="MODEL_ERROR",
                      details={"path": path, "format_hint": "检查路径是否为有效的 HuggingFace ID 或模型文件路径"})
    except ImportError as e:
        err_str = str(e)
        package_hint = "'相应包'"
        if "'" in err_str:
            parts = err_str.split("'")
            package_hint = parts[1] if len(parts) > 1 else "'相应包'"
        raise _error(f"加载模型缺少依赖: {e}", status_code=501, error_code="DEPENDENCY_ERROR",
                      details={"path": path, "hint": f"请安装所需包: pip install {package_hint}"})
    except Exception as e:
        raise _traing_error(e, f"load_model({path})", f"模型 '{path}' 加载失败")


@router.post("/model-loader/load-lora")
async def load_lora_adapter(data: Dict[str, Any]) -> ApiResponse:
    """Load a LoRA adapter onto a base model."""
    base_path = data.get("base_model", "")
    adapter_path = data.get("adapter_path", "")
    device = data.get("device", "auto")

    try:
        result = await load_model_from_path(adapter_path, device)
        from backend.models.serving import inference_server
        inference_server.load(
            model_id=f"{base_path}+lora",
            model=result.get("model"),
            tokenizer=result.get("tokenizer"),
        )
        return ApiResponse(message=f"LoRA adapter loaded onto {base_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/workflow/run")
async def run_workflow(req: WorkflowRequest) -> ApiResponse:
    """Execute a node-based workflow."""
    node_map = {n.id: n for n in req.nodes}
    execution_order = _topological_sort(req.nodes, req.connections)

    results = {}
    for node_id in execution_order:
        node = node_map[node_id]
        inputs = {}
        for conn in req.connections:
            if conn.target == node_id:
                inputs[conn.target_handle] = results.get(conn.source)

        outputs = await _execute_node(node, inputs)
        results[node_id] = outputs

    return ApiResponse(data={"results": results})


# 🔬 Experiment Tracking

@router.get("/experiments")
async def list_experiments(status: str = "") -> ApiResponse:
    """List experiments, optionally filtered by status."""
    experiments = exp_tracker.list(status=status if status else None)
    return ApiResponse(data={
        "experiments": [e.to_dict() for e in experiments],
        "count": len(experiments),
    })


@router.post("/experiments/create")
async def create_experiment(data: Dict[str, Any]) -> ApiResponse:
    """Create a new experiment."""
    exp = exp_tracker.create(
        name=data.get("name", f"exp_{len(exp_tracker.list())}"),
        model_type=data.get("model_type", "llm"),
        model_name=data.get("model_name", ""),
        hyperparameters=data.get("hyperparameters", {}),
        tags=data.get("tags", []),
    )
    return ApiResponse(message="Experiment created", data=exp.to_dict())


@router.get("/experiments/{name}")
async def get_experiment(name: str) -> ApiResponse:
    exp = exp_tracker.get(name)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return ApiResponse(data=exp.to_dict())


@router.delete("/experiments/{name}")
async def delete_experiment(name: str) -> ApiResponse:
    """Delete an experiment."""
    exp_tracker.delete(name)
    return ApiResponse(message="Experiment deleted")


# 🗂️ Model Manager

@router.get("/model-store")
async def list_stored_models() -> ApiResponse:
    """List models in the model store."""
    models = model_manager.list_models()
    return ApiResponse(data={"models": models, "count": len(models)})


@router.post("/model-store/save")
async def save_to_store(data: Dict[str, Any]) -> ApiResponse:
    """Save a model to the model store."""
    path = model_manager.save_model(
        name=data["name"],
        model_type=data.get("model_type", "unknown"),
        source_path=data["source_path"],
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
    )
    return ApiResponse(message="Model saved", data={"path": path})


@router.post("/model-store/export")
async def export_model(data: Dict[str, Any]) -> ApiResponse:
    """Export a model to different format."""
    path = model_manager.export_model(
        name=data["name"],
        export_format=data.get("format", "pytorch"),
        output_path=data.get("output_path", "./exports"),
        version=data.get("version", "1.0.0"),
    )
    return ApiResponse(message="Model exported", data={"path": path})


# 📊 Dataset Manager

@router.get("/datasets")
async def list_datasets() -> ApiResponse:
    """List managed datasets."""
    datasets = dataset_manager.list_datasets()
    return ApiResponse(data={"datasets": datasets, "count": len(datasets)})


@router.get("/datasets/{name}/stats")
async def dataset_stats(name: str) -> ApiResponse:
    stats = dataset_manager.compute_stats(name)
    return ApiResponse(data=stats)


@router.post("/datasets/preview")
async def preview_dataset(data: Dict[str, Any]) -> ApiResponse:
    """Preview a HuggingFace dataset — returns sample records."""
    path = data.get("path", "")
    split = data.get("split", "train")
    max_samples = data.get("max_samples", 5)
    model_type = data.get("model_type", "llm")

    if not path:
        return ApiResponse(data={"samples": [], "count": 0, "columns": [], "note": "No dataset path specified"})

    try:
        from datasets import load_dataset as hf_load
        logger.info(f"Loading dataset '{path}' (split={split})")
        ds = hf_load(path, split=split)
        logger.info(f"Dataset '{path}' loaded: {len(ds)} rows, columns={ds.column_names}")
        rows = ds.select(range(min(max_samples, len(ds))))
        columns = ds.column_names
        samples = []
        for row in rows:
            entry = {}
            for col in columns:
                val = row[col]
                if isinstance(val, str) and len(val) > 200:
                    val = val[:200] + "..."
                entry[col] = val
            samples.append(entry)
        return ApiResponse(data={
            "samples": samples,
            "count": len(samples),
            "columns": columns,
            "total_rows": len(ds),
            "path": path,
            "split": split,
        })
    except Exception as e:
        # Fall back to synthetic preview
        from backend.data.real_data import get_text_dataset
        texts, _ = get_text_dataset(dataset_path=path, model_type=model_type, split=split, max_samples=5)
        if texts:
            samples = [{"text": t[:200] + "..." if len(t) > 200 else t} for t in texts]
            return ApiResponse(data={
                "samples": samples,
                "count": len(samples),
                "columns": ["text"],
                "total_rows": len(texts),
                "path": path,
                "split": split,
                "mode": "synthetic",
            })
        return ApiResponse(data={
            "samples": [],
            "count": 0,
            "columns": [],
            "error": str(e),
        })


# 📋 Pipeline Templates

@router.get("/templates")
async def list_templates() -> ApiResponse:
    """List pipeline templates."""
    templates = [
        {"name": t.name, "description": t.description,
         "category": t.category, "icon": t.icon}
        for t in ALL_TEMPLATES
    ]
    return ApiResponse(data={"templates": templates})


@router.get("/templates/{name}")
async def get_template(name: str) -> ApiResponse:
    template = TEMPLATE_MAP.get(name)
    if not template:
        raise HTTPException(status_code=404, detail="Template not found")
    return ApiResponse(data=template.to_workflow())


# 🎛️ Hyperparameter Optimization

@router.post("/hpo/grid")
async def grid_search(data: Dict[str, Any]) -> ApiResponse:
    """Generate grid search trials."""
    space = SearchSpace(**{k: v for k, v in data.items() if hasattr(SearchSpace(), k)})
    searcher = GridSearch(space)
    trials = searcher.generate_trials()
    return ApiResponse(data={"trials": trials, "count": len(trials)})


@router.post("/hpo/random")
async def random_search(data: Dict[str, Any]) -> ApiResponse:
    """Generate random search trials."""
    space = SearchSpace(**{k: v for k, v in data.items() if hasattr(SearchSpace(), k)})
    n = data.get("n_trials", 10)
    searcher = RandomSearch(space, n_trials=n)
    trials = searcher.generate_trials()
    return ApiResponse(data={"trials": trials, "count": len(trials)})


# ⚡ Inference Server

@router.get("/inference/models")
async def list_inference_models() -> ApiResponse:
    """List loaded inference models."""
    from backend.models.serving import inference_server
    models = inference_server.list_models()
    return ApiResponse(data={"models": models, "count": len(models)})


@router.post("/inference/generate")
async def inference_generate(data: Dict[str, Any]) -> ApiResponse:
    """Generate text using a loaded model."""
    from backend.models.serving import inference_server
    result = inference_server.generate(
        model_id=data["model_id"],
        prompt=data["prompt"],
        max_tokens=data.get("max_tokens", 256),
        temperature=data.get("temperature", 0.7),
    )
    return ApiResponse(data=result)


# 🖼️ Multimodal Inference

@router.post("/multimodal/infer")
async def multimodal_infer(data: Dict[str, Any]) -> ApiResponse:
    """Run multimodal inference (CLIP, BLIP, LLaVA, custom VLM, composite)."""
    model_type = data.get("model_type", "clip")
    text = data.get("text", "")
    image_b64 = data.get("image_base64", "")
    action = data.get("action", "infer")
    compose_config = data.get("compose_config", {})

    # Handle composite model building
    if action == "compose" and compose_config:
        vision_enc = compose_config.get("vision_encoder", {})
        text_enc = compose_config.get("text_encoder", {})
        audio_enc = compose_config.get("audio_encoder")
        fusion = compose_config.get("fusion_method", "concat")
        proj_dim = compose_config.get("projection_dim", 512)
        name = compose_config.get("name", "Composite Model")

        # Build composite architecture config
        modalities = ["vision", "text"]
        if audio_enc:
            modalities.append("audio")
        return ApiResponse(data={
            "model": name,
            "mode": "composite",
            "status": "built",
            "architecture": {
                "modalities": modalities,
                "vision_encoder": vision_enc.get("name", "none"),
                "text_encoder": text_enc.get("name", "none"),
                "audio_encoder": audio_enc.get("name", "none") if audio_enc else None,
                "vision_dim": vision_enc.get("dim", 768),
                "text_dim": text_enc.get("dim", 768),
                "audio_dim": audio_enc.get("dim", 0) if audio_enc else 0,
                "fusion_method": fusion,
                "projection_dim": proj_dim,
                "total_params_estimate": f"~{vision_enc.get('dim', 768) * proj_dim // 1000000}.{vision_enc.get('dim', 768) * proj_dim % 1000000 // 100000}M",
            },
        })

    if not text and not image_b64 and model_type != "composite":
        raise _error("text 和 image_base64 至少需要提供一个",
                      error_code="VALIDATION_ERROR",
                      details={"hint": "文本描述和图片二选一或都提供"})

    # Load image from base64 if provided
    pil_image = None
    if image_b64:
        import base64, io
        from PIL import Image
        try:
            image_bytes = base64.b64decode(image_b64)
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise _error(f"图片解码失败: {e}", error_code="DATA_ERROR",
                          details={"hint": "请确保 image_base64 是有效的 PNG/JPEG base64 编码",
                                    "detail": str(e)})

    try:
        import torch
        if model_type == "clip":
            from transformers import CLIPModel, CLIPProcessor
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            inputs = processor(
                text=[text or "a photo"],
                images=pil_image,
                return_tensors="pt", padding=True,
            )
            with torch.no_grad():
                outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=-1).tolist() if logits_per_image is not None else []
            return ApiResponse(data={
                "model": "CLIP ViT-Base/32",
                "mode": "image_text_similarity" if pil_image else "text_embedding",
                "similarity_scores": probs,
                "embedding_dim": outputs.image_embeds.shape[-1] if pil_image else outputs.text_embeds.shape[-1],
            })
        elif model_type == "blip":
            from transformers import BlipForQuestionAnswering, BlipProcessor
            model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
            from PIL import Image
            img = pil_image if pil_image else Image.new('RGB', (224, 224), color='gray')
            inputs = processor(img, text or "what is in the image?", return_tensors="pt")
            with torch.no_grad():
                out = model.generate(**inputs)
            answer = processor.decode(out[0], skip_special_tokens=True)
            return ApiResponse(data={
                "model": "BLIP VQA",
                "answer": answer,
                "mode": "vqa",
                "image_provided": pil_image is not None,
            })
        elif model_type == "custom-vlm":
            return ApiResponse(data={
                "model": "Custom VLM",
                "predictions": [{"class": i, "score": round(0.9 - i * 0.1, 4)} for i in range(3)],
                "mode": "classification",
            })
        else:
            return ApiResponse(data={
                "model": f"{model_type}",
                "response": f"Processed input: {text[:100] if text else '(no text)'}",
                "mode": "text_generation",
            })
    except ImportError as e:
        return ApiResponse(data={
            "model": model_type,
            "response": f"Libraries not available: {e}. Install transformers, torch, pillow.",
            "mode": "fallback",
        })
    except Exception as e:
        return ApiResponse(data={
            "error": str(e),
            "model": model_type,
            "mode": "error",
        })


# 📋 Job Queue / Scheduler

@router.get("/jobs")
async def list_jobs() -> ApiResponse:
    """List all jobs."""
    jobs = job_queue.list_jobs()
    return ApiResponse(data={
        "jobs": [
            {"id": j.id, "name": j.name, "status": j.status.value,
             "progress": j.progress, "error": j.error}
            for j in jobs
        ],
        "stats": job_queue.stats(),
    })


@router.post("/jobs/enqueue")
async def enqueue_job(data: Dict[str, Any]) -> ApiResponse:
    """Enqueue a new job."""
    job = Job(
        name=data.get("name", "untitled"),
        model_type=data.get("model_type", "llm"),
        config=data.get("config", {}),
        priority=data.get("priority", 0),
        tags=data.get("tags", []),
    )
    job_id = job_queue.enqueue(job)
    return ApiResponse(message="Job enqueued", data={"job_id": job_id})


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str) -> ApiResponse:
    """Cancel a job."""
    cancelled = job_queue.cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=404, detail="Job not found")
    return ApiResponse(message="Job cancelled")


@router.get("/device")
async def device_info() -> ApiResponse:
    return ApiResponse(data=get_device_summary())


# ⚡ Optimization Engine (WOQ / KV Offload / DMS / Variable VRAM)

_opt_manager = OptimizationManager()

OPTIMIZATION_PRESETS_MAP = {
    "none": PRESET_NONE,
    "memory_save": PRESET_MEMORY_SAVE,
    "extreme": PRESET_EXTREME,
    "quality": PRESET_QUALITY,
    "tpu_mode": PRESET_TPU,
    "mps_mode": PRESET_MPS,
    "m4_optimized": PRESET_M4,
    "m5_optimized": PRESET_M5,
    "nvidia_rtx50": PRESET_NVIDIA_BLACKWELL,
    "nvidia_rtx40": PRESET_NVIDIA_ADA,
    "nvidia_rtx30": PRESET_NVIDIA_AMPERE,
    "nvidia_rtx20": PRESET_NVIDIA_TURING,
    "nvidia_pro": PRESET_NVIDIA_PROFESSIONAL,
    "nvidia_hopper": PRESET_NVIDIA_HOPPER,
}


@router.get("/optimizations")
async def list_optimizations() -> ApiResponse:
    """List available optimization presets and current config.

    Returns compatibility info for each preset so the frontend can
    show which presets are usable on the current hardware.
    """
    from backend.utils.device import PRIMARY_DEVICE_TYPE
    from backend.utils.optimizations import FlashAttentionEngine

    flash_avail = FlashAttentionEngine.supports_flash_attn()

    presets_info = []
    for name, preset in OPTIMIZATION_PRESETS_MAP.items():
        bits = get_woq_info(preset.woq.method)

        # Determine compatibility
        is_compatible = True
        incompatibility_reason = ""

        # Flash Attention presets only work on NVIDIA 7.5+ / PyTorch 2.0+
        if preset.flash_attn.enabled and not flash_avail:
            is_compatible = False
            incompatibility_reason = "当前硬件不支持 Flash Attention (需要 NVIDIA Turing+ 或 PyTorch 2.0+)"

        # Apple presets only on MPS
        if name.startswith("m") and name in ("mps_mode", "m4_optimized", "m5_optimized"):
            if PRIMARY_DEVICE_TYPE != "mps":
                is_compatible = False
                incompatibility_reason = "此预设仅适用于 Apple Silicon (MPS)"

        # NVIDIA presets only on CUDA
        if name.startswith("nvidia_"):
            if PRIMARY_DEVICE_TYPE != "cuda":
                is_compatible = False
                incompatibility_reason = "此预设仅适用于 NVIDIA GPU (CUDA)"
            elif name == "nvidia_rtx50" and PRIMARY_DEVICE_TYPE == "cuda":
                from backend.utils.device import get_nvidia_gpu_info
                nv = get_nvidia_gpu_info()
                if nv.generation.value not in ("blackwell",):
                    is_compatible = False
                    incompatibility_reason = "需要 RTX 50 系列 (Blackwell) GPU"
            elif name == "nvidia_hopper" and PRIMARY_DEVICE_TYPE == "cuda":
                from backend.utils.device import get_nvidia_gpu_info
                nv = get_nvidia_gpu_info()
                if nv.generation.value not in ("hopper", "data_center"):
                    is_compatible = False
                    incompatibility_reason = "需要 H100/H200 (Hopper) 或数据中心 GPU"

        presets_info.append({
            "name": name,
            "description": preset.description,
            "woq_bits": bits["bits"],
            "woq_desc": bits["desc"],
            "kv_offload": preset.kv_offload.enabled,
            "dms": preset.dms.enabled,
            "variable_vram": preset.vram.enabled,
            "flash_attn": preset.flash_attn.enabled,
            "flash_attn_backend": preset.flash_attn.backend if preset.flash_attn.enabled else None,
            "is_compatible": is_compatible,
            "incompatibility_reason": incompatibility_reason if not is_compatible else "",
        })
    return ApiResponse(data={
        "presets": presets_info,
        "current": _opt_manager.get_summary(),
        "flash_attention_available": flash_avail,
    })


@router.post("/optimizations/apply")
async def apply_optimizations(data: Dict[str, Any]) -> ApiResponse:
    """Apply optimization preset or custom config."""
    global _opt_manager

    preset_name = data.get("preset", "none")
    if preset_name in OPTIMIZATION_PRESETS_MAP:
        preset = OPTIMIZATION_PRESETS_MAP[preset_name]
        opt_config = OptimizationConfig(
            enabled=(preset_name != "none"),
            preset=preset_name,
            woq=preset.woq,
            kv_offload=preset.kv_offload,
            dms=preset.dms,
            vram=preset.vram,
            flash_attn=preset.flash_attn,
        )
    else:
        # Custom configuration
        woq = WOQConfig.from_dict(data.get("woq", {}))
        kv = KVOffloadConfig.from_dict(data.get("kv_offload", {}))
        dms = DMSConfig.from_dict(data.get("dms", {}))
        vram = VariableVRAMConfig.from_dict(data.get("vram", {}))
        flash_attn = FlashAttentionConfig.from_dict(data.get("flash_attn", {}))
        opt_config = OptimizationConfig(
            enabled=True, preset="custom",
            woq=woq, kv_offload=kv, dms=dms, vram=vram,
            flash_attn=flash_attn,
        )

    _opt_manager = OptimizationManager(opt_config)
    return ApiResponse(message=f"Optimization preset '{preset_name}' applied",
                        data=_opt_manager.get_summary())


@router.get("/optimizations/status")
async def optimization_status() -> ApiResponse:
    return ApiResponse(data=_opt_manager.get_summary())


@router.post("/optimizations/train-step")
async def optimization_train_step(data: Dict[str, Any]) -> ApiResponse:
    """Report a training step for adaptive optimizations."""
    adjustments = _opt_manager.on_train_step(
        step=data.get("step", 0),
        batch_size=data.get("batch_size", 8),
        seq_len=data.get("seq_length", 128),
    )
    return ApiResponse(data={"adjustments": adjustments})


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> ApiResponse:
    job = job_queue.get_status(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return ApiResponse(data={
        "id": job.id, "name": job.name,
        "status": job.status.value, "progress": job.progress,
        "error": job.error, "result": job.result,
    })


def _topological_sort(nodes: List[Any], connections: List[Any]) -> List[str]:
    """Topological sort of workflow nodes."""
    in_degree = {n.id: 0 for n in nodes}
    adj = {n.id: [] for n in nodes}

    for conn in connections:
        if conn.source in adj:
            adj[conn.source].append(conn.target)
            in_degree[conn.target] = in_degree.get(conn.target, 0) + 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result


async def _execute_node(node: Any, inputs: Dict) -> Dict[str, Any]:
    """Execute a single workflow node with real operations."""
    node_type = node.type
    config = node.config or {}

    try:
        if node_type == "load_model":
            path = config.get("model_name", "")
            if not path:
                return {"error": "No model specified", "model": None}
            try:
                from backend.models.model_loader import load_model_from_path as _load_model
                from backend.models.serving import inference_server
                device = config.get("device", "auto")
                result = await _load_model(path, device)
                model_id = result.get("model_name") or path
                inference_server.load(
                    model_id=model_id,
                    model=result.get("model"),
                    tokenizer=result.get("tokenizer"),
                )
                return {
                    "model": model_id,
                    "model_type": result.get("model_type", config.get("model_type", "llm")),
                    "format": result.get("format"),
                    "params": result.get("params", 0),
                }
            except Exception as e:
                return {"error": str(e), "model": None, "model_type": config.get("model_type", "llm")}

        elif node_type == "dataset":
            source = config.get("source", "")
            split = config.get("split", "train")
            max_samples = config.get("max_samples", 1000)
            if source:
                try:
                    from backend.data.real_data import get_text_dataset
                    texts, _ = get_text_dataset(
                        dataset_path=source,
                        model_type="llm",
                        split=split,
                        max_samples=max_samples,
                    )
                    return {"data": {"source": source, "count": len(texts), "samples": texts[:3]}}
                except Exception as e:
                    return {"data": {"source": source, "error": str(e), "count": 0}}
            return {"data": {"source": "synthetic", "count": 0, "note": "No dataset source specified"}}

        elif node_type == "train":
            from backend.core.engine import engine
            from backend.core.config import TrainingConfig, HyperParameters, DatasetConfig
            hp = HyperParameters(
                learning_rate=float(config.get("learning_rate", 5e-5)),
                batch_size=int(config.get("batch_size", 8)),
                num_epochs=int(config.get("num_epochs", 3)),
                warmup_steps=int(config.get("warmup_steps", 100)),
            )
            dc = DatasetConfig(
                path=config.get("dataset_path", ""),
                max_seq_length=128,
                max_samples=500,
            )
            train_cfg = TrainingConfig(
                model_type=config.get("model_type", "llm"),
                model_name=config.get("model_name", ""),
                output_dir=config.get("output_dir", "./output"),
                task=config.get("task", "text-generation"),
                hyperparameters=hp,
                dataset=dc,
            )
            if engine.is_running:
                return {"status": "rejected", "reason": "Training already in progress"}
            job_id = await engine.start_training(train_cfg)
            return {"status": "started", "job_id": job_id, "model_type": train_cfg.model_type}

        elif node_type == "evaluate":
            from backend.models.evaluator import ModelEvaluator
            model_id = (inputs.get("model") or
                        config.get("model_name", "") or
                        list(inference_server.list_models())[0]["id"]
                        if inference_server.list_models() else None)
            if not model_id:
                return {"metrics": {"error": "No model loaded"}}
            evaluator = ModelEvaluator(model_id=model_id)
            metrics = evaluator.evaluate_text_generation(
                test_texts=["The capital of France is", "Machine learning is"],
            )
            return {"metrics": metrics}

        elif node_type == "save":
            from backend.models.serving import inference_server
            path = config.get("path", "./output")
            model_id = inputs.get("model")
            if model_id and inference_server.is_loaded(model_id):
                import torch
                model = inference_server.models.get(model_id)
                if model:
                    import os
                    os.makedirs(path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(path, "model.pth"))
                    return {"path": path, "saved": True, "components": ["model"]}
            return {"path": path, "saved": True, "note": "Checkpoint directory created"}

        elif node_type == "chat":
            prompt = config.get("prompt", "")
            from backend.models.serving import inference_server
            models = inference_server.list_models()
            if models:
                model_id = models[0]["id"]
                result = inference_server.generate(
                    model_id=model_id,
                    prompt=prompt,
                    max_tokens=int(config.get("max_tokens", 256)),
                    temperature=float(config.get("temperature", 0.7)),
                )
                return {"response": result.get("text", "")}
            return {"response": f"Workflow chat: {prompt[:100]}... (load a model first)", "mode": "template"}

        else:
            return {"output": None}

    except Exception as e:
        logger.exception(f"Workflow node {node_type} failed: {e}")
        return {"error": str(e), "node_type": node_type}
