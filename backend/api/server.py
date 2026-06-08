"""FastAPI server configuration with embedded static file serving."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

from backend.api.routes import router
from backend.api.websocket import ws_router
from backend.core.registry import registry

# Structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("aitrainer")

# Optional API key authentication
API_KEY = os.environ.get("AITRAINER_API_KEY", "")


def register_model_types() -> None:
    """Register all supported model types (without loading torch)."""
    registry.register_trainer(
        "llm", "backend.models.llm_trainer.LLMTrainer",
        {"description": "LLM fine-tuning (GPT, LLaMA, Mistral, etc.)"}
    )
    registry.register_trainer(
        "lcm", "backend.models.lcm_trainer.LCMTrainer",
        {"description": "Latent Consistency Model training/distillation"}
    )
    registry.register_trainer(
        "lcm-lora", "backend.models.lcm_trainer.LCMLoraTrainer",
        {"description": "LCM LoRA fine-tuning"}
    )
    registry.register_trainer(
        "cnn", "backend.models.cnn_trainer.CNNTrainer",
        {"description": "CNN training for image classification"}
    )
    registry.register_trainer(
        "lora", "backend.models.lora_trainer.LoRATrainer",
        {"description": "LoRA parameter-efficient fine-tuning"}
    )
    registry.register_trainer(
        "qlora", "backend.models.qlora_trainer.QLoRATrainer",
        {"description": "QLoRA 4-bit quantized fine-tuning"}
    )
    registry.register_trainer(
        "multimodal", "backend.models.multimodal_trainer.MultiModalTrainer",
        {"description": "Multimodal vision-language model training (VLM)"}
    )
    registry.register_trainer(
        "clip", "backend.models.multimodal_trainer.CLIPTrainer",
        {"description": "CLIP contrastive vision-language pretraining"}
    )
    registry.register_trainer(
        "blip", "backend.models.multimodal_trainer.BLIPTrainer",
        {"description": "BLIP vision-language understanding & generation"}
    )
    registry.register_trainer(
        "gpt", "backend.models.gpt_trainer.GPTTrainer",
        {"description": "GPT decoder-only fine-tuning (GPT-2, GPT-Neo, etc.)"}
    )
    registry.register_trainer(
        "bert", "backend.models.bert_trainer.BERTTrainer",
        {"description": "BERT fine-tuning (classification, NER, QA, etc.)"}
    )
    registry.register_trainer(
        "rnn", "backend.models.rnn_trainer.RNNTrainer",
        {"description": "RNN training for sequence modeling"}
    )
    registry.register_trainer(
        "lstm", "backend.models.lstm_trainer.LSTMTrainer",
        {"description": "LSTM training for sequence modeling and text generation"}
    )
    registry.register_trainer(
        "moe", "backend.models.moe_trainer.MoETrainer",
        {"description": "MoE (Mixture of Experts) Transformer - sparse expert model training & fine-tuning"}
    )
    registry.register_trainer(
        "moe-from-scratch", "backend.models.moe_trainer.MoETrainer",
        {"description": "Train a MoE Transformer from scratch with top-2 routing"}
    )
    registry.register_trainer(
        "moe-finetune", "backend.models.moe_trainer.MoEFinetuneTrainer",
        {"description": "Fine-tune pretrained MoE models (Mixtral, DeepSeek, Qwen2-MoE)"}
    )
    registry.register_trainer(
        "scratch-transformer", "backend.models.scratch_trainer.ScratchTransformerTrainer",
        {"description": "Train a Transformer (decoder/encoder) from random initialization"}
    )
    registry.register_trainer(
        "scratch-cnn", "backend.models.scratch_trainer.ScratchCNNTrainer",
        {"description": "Train a CNN classifier from random initialization"}
    )
    registry.register_trainer(
        "scratch-lstm", "backend.models.scratch_trainer.ScratchLSTMTrainer",
        {"description": "Train an LSTM from random initialization"}
    )
    # ─── New Model Types ─────────────────────────────────────────────────
    registry.register_trainer(
        "whisper", "backend.models.whisper_trainer.WhisperTrainer",
        {"description": "Whisper speech recognition / audio transcription"}
    )
    registry.register_trainer(
        "diffusion", "backend.models.diffusion_trainer.DiffusionTrainer",
        {"description": "Stable Diffusion text-to-image generation"}
    )
    registry.register_trainer(
        "flux", "backend.models.diffusion_trainer.FluxTrainer",
        {"description": "Flux (Black Forest Labs) text-to-image generation"}
    )
    registry.register_trainer(
        "t5", "backend.models.t5_trainer.T5Trainer",
        {"description": "T5 encoder-decoder text-to-text fine-tuning"}
    )
    registry.register_trainer(
        "phi", "backend.models.phi_trainer.PhiTrainer",
        {"description": "Microsoft Phi small language model fine-tuning"}
    )
    registry.register_trainer(
        "detr", "backend.models.detr_trainer.DETRTrainer",
        {"description": "DETR object detection fine-tuning"}
    )
    registry.register_trainer(
        "embedding", "backend.models.embedding_trainer.EmbeddingTrainer",
        {"description": "Embedding model (BGE, Instructor) fine-tuning"}
    )
    registry.register_trainer(
        "sam", "backend.models.sam_trainer.SAMTrainer",
        {"description": "Segment Anything Model (SAM) image segmentation"}
    )
    # ─── Video Generation Models ─────────────────────────────────────────
    registry.register_trainer(
        "video-diffusion", "backend.models.video_trainer.VideoDiffusionTrainer",
        {"description": "Stable Video Diffusion — image-to-video generation"}
    )
    registry.register_trainer(
        "i2vgen-xl", "backend.models.video_trainer.I2VGenXLTrainer",
        {"description": "I2VGen-XL — image-to-video generation (阿里)"}
    )
    registry.register_trainer(
        "frame-interpolation", "backend.models.video_trainer.FrameInterpolationTrainer",
        {"description": "Video frame interpolation (FILM, RIFE)"}
    )


def _get_error_code(status_code: int) -> str:
    """Map HTTP status code to error code."""
    mapping = {
        400: "VALIDATION_ERROR",
        401: "AUTH_ERROR",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        405: "METHOD_NOT_ALLOWED",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "RATE_LIMIT",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE",
    }
    return mapping.get(status_code, f"HTTP_{status_code}")


def _is_debug() -> bool:
    import os
    return os.environ.get("AITRAINER_DEBUG", "").lower() in ("1", "true", "yes")


def get_static_dir() -> Path:
    """
    Supports multiple deployment modes:
    1. Development: frontend/dist/
    2. PyInstaller: sys._MEIPASS/frontend/dist/
    3. Environment variable override
    """
    # Check environment override first
    env_dir = os.environ.get("AITRAINER_STATIC_DIR")
    if env_dir:
        return Path(env_dir)

    # Check PyInstaller bundle
    import sys as _sys
    if getattr(_sys, "frozen", False):
        base = Path(_sys._MEIPASS)
        candidates = [
            base / "frontend" / "dist",
            base / "dist",
        ]
        for c in candidates:
            if c.exists():
                return c

    # Development mode
    candidates = [
        Path(__file__).parent.parent.parent / "frontend" / "dist",
        Path(__file__).parent.parent.parent / "dist",
    ]
    for c in candidates:
        if c.exists():
            return c

    return Path("frontend/dist")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AITrainerUltra",
        description="多模型AI训练框架 - Multi-Model AI Training Framework",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import time
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        if not request.url.path.startswith("/assets"):
            logger.info(
                "%s %s → %s (%.0fms)",
                request.method, request.url.path,
                response.status_code, elapsed * 1000,
            )
        return response

    # Optional API key auth middleware
    if API_KEY:
        @app.middleware("http")
        async def api_key_auth(request: Request, call_next):
            if request.url.path.startswith("/api"):
                key = request.headers.get("X-API-Key", "")
                if key != API_KEY:
                    from fastapi.responses import JSONResponse
                    return JSONResponse(
                        {"detail": "Invalid API key"},
                        status_code=401,
                    )
            return await call_next(request)
        logger.info("API key authentication enabled")

    # ─── Global structured error handler ───────────────────────────────
    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception):
        """统一结构化错误处理 - 所有未捕获异常返回标准格式."""
        from fastapi.responses import JSONResponse
        from backend.api.schemas import ApiResponse

        # HTTPException (如 404, 422) — 保留原始状态码
        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content=ApiResponse(
                    success=False,
                    message=exc.detail or str(exc),
                    error_code=_get_error_code(exc.status_code),
                    error_details={"path": request.url.path, "method": request.method},
                ).model_dump(),
            )

        # Pydantic 验证错误
        from pydantic import ValidationError
        if isinstance(exc, ValidationError):
            return JSONResponse(
                status_code=422,
                content=ApiResponse(
                    success=False,
                    message="请求参数验证失败",
                    error_code="VALIDATION_ERROR",
                    error_details={
                        "path": request.url.path,
                        "errors": [
                            {
                                "field": " → ".join(str(l) for l in e["loc"]),
                                "msg": e["msg"],
                                "type": e["type"],
                            }
                            for e in exc.errors()
                        ],
                    },
                ).model_dump(),
            )

        # 记录未预期异常
        logger.exception("未处理的异常: %s %s", request.method, request.url.path)

        return JSONResponse(
            status_code=500,
            content=ApiResponse(
                success=False,
                message=f"服务器内部错误: {type(exc).__name__}",
                error_code="INTERNAL_ERROR",
                error_details={
                    "path": request.url.path,
                    "type": type(exc).__name__,
                    "detail": str(exc) if _is_debug() else None,
                },
            ).model_dump(),
        )

    register_model_types()
    app.include_router(router)
    app.include_router(ws_router)

    # Serve frontend static files
    static_dir = get_static_dir()
    if static_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=str(static_dir / "assets")),
            name="assets",
        )

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        async def serve_frontend():
            index_path = static_dir / "index.html"
            if index_path.exists():
                return HTMLResponse(index_path.read_text(encoding="utf-8"))
            return HTMLResponse("<h1>AITrainerUltra</h1><p>Frontend not built. Run: cd frontend && npm run build</p>")

        @app.exception_handler(404)
        async def fallback_to_frontend(request, exc):
            index_path = static_dir / "index.html"
            if index_path.exists() and not request.url.path.startswith("/api"):
                return HTMLResponse(index_path.read_text(encoding="utf-8"))
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "Not Found"}, status_code=404)

    return app


app = create_app()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    import time
    return {"status": "ok", "service": "AITrainerUltra", "version": "2.0.0", "timestamp": time.time()}
