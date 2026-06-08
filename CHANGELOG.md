# Changelog

All notable changes to AITrainerUltra will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2025-06-06

### Added

- **25+ model types**: LLM, GPT, BERT, T5, Phi, MoE, Whisper, Stable Diffusion, Flux, Stable Video Diffusion, DETR, SAM, CLIP, BLIP, Embedding, LCM, RNN, LSTM, CNN, and scratch trainers (Transformer, CNN, LSTM)
- **Visual node-based workflow editor**: 20+ node types with drag-and-drop connections
- **Model chat interface**: Real-time streaming responses via SSE
- **Experiment tracking**: Create, compare, and manage training experiments
- **Model optimization engine**: WOQ (INT8/NF4/INT2/GPTQ/AWQ/GGUF), KV Cache Offload, Dynamic Memory Compression (DMS), Variable VRAM, Flash Attention (SDPA/xformers/Triton)
- **20 pre-configured training recipes**: Hardware-tuned presets for various model-device combinations
- **Hyperparameter optimization**: Grid search, random search, and Bayesian optimization
- **Custom composite multimodal model builder**: Combine vision + text + audio encoders
- **Video generation**: Stable Video Diffusion, I2VGen-XL, Frame Interpolation
- **Device support**: 7 compute platforms (CUDA, MPS, ROCm, TPU, NPU, XPU, CPU) with chip-specific presets
- **Bilingual UI**: Chinese and English language support
- **Inno Setup installer**: Windows installation wizard
- **Docker support**: docker-compose for backend + frontend
- **CI pipeline**: GitHub Actions for linting, testing, and type checking

### Packaging

- One-command packaging via `python package.py` (frontend → backend → .exe)
- PyInstaller standalone executable
- PyInstaller entry point with auto port release and browser launch
- Virtual environment auto-setup on first launch
