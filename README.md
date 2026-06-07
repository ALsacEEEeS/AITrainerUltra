# 🧠 AITrainerUltra

> **Multi-Model AI Training Framework** — 多模型AI训练框架
> 25+ model types · Visual workflow · Chat interface · Model optimization · Apple Silicon optimized

![Version](https://img.shields.io/badge/version-2.0.0-blue?style=flat-square)
![Python](https://img.shields.io/badge/python-3.11-green?style=flat-square)
![React](https://img.shields.io/badge/react-18.3-cyan?style=flat-square)
![Models](https://img.shields.io/badge/models-25%2B-brightgreen?style=flat-square)
![Platforms](https://img.shields.io/badge/platforms-7-orange?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)
[![CI](https://github.com/your-org/AITrainerUltra/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/AITrainerUltra/actions/workflows/ci.yml)

---

## 📖 Overview / 概述

**English:** AITrainerUltra is a comprehensive multi-model AI training framework with a visual node editor, chat interface, experiment tracking, model optimization (WOQ/KV Offload/DMS/Variable VRAM/Flash Attention), and support for **25+ model types** including LLM, GPT, BERT, T5, Phi, Whisper, Stable Diffusion, Flux, Stable Video Diffusion, DETR, SAM, and more. It runs on **7 compute platforms**: CPU, NVIDIA CUDA (including RTX 50-series/Blackwell/Hopper/Professional), Apple MPS (M1–M5 with chip-specific tuning), AMD ROCm, Google TPU, Huawei Ascend NPU, and Intel XPU.

**中文:** AITrainerUltra 是一个全功能多模型AI训练框架，提供可视化节点编辑器、对话交互界面、实验追踪、模型优化（WOQ/KV卸载/DMS/自适应VRAM/Flash Attention），支持 **25+种模型类型**，包括 LLM、GPT、BERT、T5、Phi、Whisper、Stable Diffusion、Flux、Stable Video Diffusion、DETR、SAM 等。可在 **7种计算平台** 上运行：CPU、NVIDIA CUDA（含RTX 50系列/Blackwell/Hopper/专业卡）、Apple MPS（M1–M5芯片级优化）、AMD ROCm、Google TPU、华为昇腾 NPU、Intel XPU。

---

## ✨ Features / 特性

| English | 中文 |
|---------|------|
| **25+ model types** (LLM, GPT, BERT, T5, Phi, Whisper, Diffusion, Flux, SVD, DETR, SAM, Embedding, Multimodal, MoE, etc.) | **25+种模型类型** |
| Visual node-based workflow editor (20+ node types, custom nodes) | 可视化节点式工作流编辑器（20+节点类型，支持自定义节点） |
| Model chat interface | 模型对话交互界面 |
| Experiment tracking & comparison | 实验追踪与对比 |
| Model optimization (WOQ, KV Offload, DMS, Variable VRAM, **Flash Attention**) | 模型优化引擎（含**Flash Attention**） |
| **Apple Silicon (M1–M5) chip-specific optimization presets** | **Apple Silicon M1–M5 芯片级优化预设** |
| **NVIDIA RTX 50-series / Blackwell / Hopper / Professional GPU presets** | **NVIDIA RTX 50系列/Blackwell/Hopper/专业卡预设** |
| **Custom composite multimodal model builder** (vision + text + audio) | **自定义合成多模态模型**（视觉+文本+音频自由组合） |
| **Video generation** (Stable Video Diffusion, I2VGen-XL, Frame Interpolation) | **视频生成**（SVD、I2VGen-XL、帧插值） |
| Bilingual UI (Chinese / English) | 中英双语界面 |
| **Built-in help documentation** | **内置帮助文档** |
| 20 pre-configured training recipes (with **Apple Silicon tips**) | 20个预配置训练食谱（含**Apple优化提示**） |
| 7 supported compute platforms | 7种计算平台 |
| **One-command packaging** (`.exe`) | **一键打包** (.exe) |
| **Auto port release** — no more "address in use" errors | **自动释放端口** — 告别端口占用 |
| **Virtual environment auto-setup** | **虚拟环境自动创建** |

## 🚀 Quick Start / 快速开始

### One-Command Launch / 一键启动

```bash
python start.py              # Start backend + frontend (dev mode)
python start.py --setup      # First-time: create venv + install deps
python start.py --prod       # Production: build frontend + start backend
python start.py --backend    # Backend only
python start.py --frontend   # Frontend only
```

The script **automatically frees port 8000** if it's occupied, so you never see "address already in use".

### Manual Start / 手动启动

```bash
# 1. Install backend dependencies / 安装后端依赖
pip install -r backend/requirements.txt

# 2. Start server / 启动服务
python -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000 --reload
# → API: http://localhost:8000/docs

# 3. (In another terminal) Build & start frontend / 启动前端
cd frontend && npm install && npm run dev
# → Web UI: http://localhost:3000
```

## 🖥️ Device Selection / 设备选择

**English:** Select your compute device from the training panel or compact device selector in the top bar. Auto-detect picks the best device for your hardware.

**中文:** 在训练面板或顶栏的紧凑设备选择器中切换计算设备。自动检测会为你的硬件选择最佳设备。

| Device / 设备 | Type / 类型 | Auto-detect / 自动识别 |
|---------------|-------------|----------------------|
| 🟢 **NVIDIA CUDA** (RTX 20/30/40/50, A-series, H100) | GPU | `torch.cuda.is_available()` + generation detection |
| 🔵 **Apple MPS** (M1–M5 with chip-specific tuning) | GPU | `torch.backends.mps.is_available()` + chip model |
| 🔴 **AMD ROCm** | GPU | `torch.version.hip` |
| 🟣 **Google TPU** | TPU | `torch_xla` |
| 🟠 **Huawei Ascend NPU** | NPU | `torch_npu.npu.is_available()` |
| 🔷 **Intel XPU** | GPU | `intel_extension_for_pytorch` |
| 🟡 **CPU** | CPU | Default (fallback) |

### NVIDIA GPU Generation Detection

| Generation | Models | Flash Attn | FP8 |
|-----------|--------|-----------|-----|
| **Blackwell** | RTX 5090, 5080, 5070 | ✅ | ✅ |
| **Ada Lovelace** | RTX 4090, 4080, 4070 | ✅ | ❌ |
| **Ampere** | RTX 3090, 3080, 3070 | ✅ | ❌ |
| **Turing** | RTX 2080, 2070, 2060 | ✅ (limited) | ❌ |
| **Professional** | RTX A6000, A5000, Quadro | ✅ | ✅ |
| **Hopper / Datacenter** | H100, H200, A100 | ✅ | ✅ |

## 🎯 Supported Models / 支持的模型 (25+种)

### Language Models / 语言模型

| Model / 模型 | Type / 类型 | Training / 训练方式 |
|-------------|-------------|-------------------|
| **LLM** | Decoder Transformer (GPT/LLaMA/Mistral) | Full fine-tune / LoRA / QLoRA / DPO |
| **GPT** | GPT-2 / GPT-Neo | Causal LM fine-tune |
| **BERT** | Encoder Transformer | MLM / classification / NER / QA |
| **T5** | Encoder-Decoder | Text-to-text fine-tune |
| **Phi** | Microsoft Phi-2/3/4 | Small LLM fine-tune |
| **MoE** | Mixtral / DeepSeek / Qwen2-MoE | Sparse expert training |
| **RNN** | Multi-layer RNN | Sequence modeling |
| **LSTM** | BiLSTM + Attention | Classification |

### Vision Models / 视觉模型

| Model / 模型 | Type / 类型 | Training / 训练方式 |
|-------------|-------------|-------------------|
| **Stable Diffusion** | UNet + VAE | Text-to-image LoRA / fine-tune |
| **Flux** | Black Forest Labs | Text-to-image |
| **CNN** | ResNet / Conv2D | Transfer learning |
| **DETR** | Detection Transformer | Object detection |
| **SAM** | Segment Anything Model | Image segmentation |
| **Vision Transformer (ViT)** | Transformer encoder | Image classification |

### Video Models / 视频模型

| Model / 模型 | Type / 类型 | Training / 训练方式 |
|-------------|-------------|-------------------|
| **Stable Video Diffusion** | SVD | Image-to-video |
| **I2VGen-XL** | Alibaba DAMO | Text/image-to-video |
| **Frame Interpolation** | FILM / RIFE | Frame rate upscaling |

### Multimodal & Audio / 多模态 & 音频

| Model / 模型 | Type / 类型 | Training / 训练方式 |
|-------------|-------------|-------------------|
| **Multimodal VLM** | Custom composite | Vision-language fusion |
| **CLIP** | ViT + GPT-2 | Contrastive learning |
| **BLIP** | ViT + BERT | VQA fine-tune |
| **Whisper** | Encoder-Decoder | Speech recognition |
| **Embedding** | BGE / Instructor | Text vectorization |
| **LCM** | UNet + Scheduler | Distillation |

### From Scratch / 从零训练

| Model / 模型 | Description / 说明 |
|-------------|-------------------|
| **Scratch Transformer** | Decoder-only or encoder-only |
| **Scratch CNN** | Convolutional classifier |
| **Scratch LSTM** | Sequence classifier |

### Custom Nodes / 自定义节点

| Node / 节点 | Description / 说明 |
|-------------|-------------------|
| **Custom Python Node** | Write arbitrary Python code in the workflow |
| **Merge Models** | Model merging (linear, SLERP, TIES, DARE) |
| **Data Augment** | Image/text augmentation |
| **Hyperparameter Opt** | Grid / random / Bayesian search |
| **Format Transform** | safetensors / ONNX / GGUF / MLX |

## ⚡ Optimization Engine / 优化引擎

| Optimization / 优化 | Description / 说明 | VRAM Savings | Compatible Hardware |
|-------------------|-------------------|-------------|-------------------|
| **WOQ** | Weight-Only Quantization (INT8/NF4/INT2/GPTQ/AWQ/GGUF) | 2x – 16x | All devices |
| **KV Offload** | KV cache offloading to CPU/disk | 50%+ | All devices |
| **DMS** | Dynamic Memory Compression | 50% – 75% | All devices |
| **Variable VRAM** | Adaptive batch/seq/precision | Prevents OOM | All devices |
| **Flash Attention** | Memory-efficient attention (SDPA/xformers/Triton) | O(n) vs O(n²) | NVIDIA Turing+ / M3+ SDPA |

### GPU-Specific Presets / GPU专用预设

| Preset / 预设 | Target / 目标 | Key Features / 关键特性 |
|-------------|---------------|----------------------|
| 🟢 **nvidia_rtx50** | RTX 5090/5080/5070 | FP8 + Flash Attn, max_batch=1024, max_seq=65536 |
| 🟢 **nvidia_rtx40** | RTX 4090/4080/4070 | FP16 + Flash Attn, max_batch=512, max_seq=32768 |
| 🟢 **nvidia_rtx30** | RTX 3090/3080/3070 | INT8 + Flash Attn, KV offload |
| 🔵 **nvidia_rtx20** | RTX 2080/2070/2060 | INT8 + KV offload |
| 💎 **nvidia_pro** | RTX A6000, Quadro | FP8 + Flash Attn, max_batch=2048 |
| ⚡ **nvidia_hopper** | H100/H200 | FP8 + Flash Attn, max_batch=4096, max_seq=262144 |
| 🍎 **m4_optimized** | Apple M4 | Unified memory, ANE, max_seq=32768 |
| 🍎 **m5_optimized** | Apple M5 | Unified memory, ANE, max_seq=131072 |

## 🧩 Custom Composite Multimodal Model / 自定义合成多模态

**English:** Freely combine vision encoders (ViT/ResNet/DINOv2/CLIP) + text encoders (BERT/T5/Phi) + audio encoders (Whisper/HuBERT) into a custom multimodal model. Choose fusion method: concatenation, cross-attention, weighted sum, or MLP gating.

**中文:** 自由组合视觉编码器（ViT/ResNet/DINOv2/CLIP）+ 文本编码器（BERT/T5/Phi）+ 音频编码器（Whisper/HuBERT），构建自定义多模态模型。支持拼接、交叉注意力、加权求和、MLP门控等融合方式。

## 📋 Training Recipes / 训练食谱 (20个)

| Recipe / 食谱 | Hardware / 硬件建议 | Apple Silicon 🍎 |
|--------------|-------------------|-----------------|
| LLM Full Fine-tune (Small) | NVIDIA 16GB+ / M2+ 16GB+ | ✅ |
| LLM Full Fine-tune (Large) | NVIDIA 48GB+ / M3+ 36GB+ | ✅ |
| LLM LoRA Fine-tune | NVIDIA 8GB+ / M1+ | ✅ |
| LLM QLoRA 4-bit | NVIDIA 8GB+ / M1+ | ✅ |
| LLM DPO Training | NVIDIA 24GB+ / M3+ 24GB+ | ✅ |
| LoRA Fine-tune (Large, r=64) | NVIDIA 16GB+ / M2+ 16GB+ | ✅ |
| Flash Attention LLM | NVIDIA 16GB+ / M3+ SDPA | ✅ |
| QLoRA 2-bit Extreme | NVIDIA 8GB+ / M1+ MLX | ✅ |
| MPS LLM (Apple Optimized) | **Apple M1–M5** | ✅ **专为Mac** |
| M4/M5 Multimodal VLM | **Apple M4/M5** | ✅ **专为M4/M5** |
| MPS LoRA + Vision-Language | **Apple M1–M5** | ✅ **专为Mac** |
| Stable Video Diffusion | NVIDIA 24GB+ / M3 Max | ✅ |
| I2VGen-XL | NVIDIA 32GB+ | — |
| Frame Interpolation | Any GPU / M1+ | ✅ |
| +6 more... | | |

## 🌐 Language Support / 语言支持

**English:** Toggle between Chinese (中文) and English (EN) using the switch in the bottom-left sidebar. The entire UI, help documentation, and training recipes are bilingual.

**中文:** 点击左侧导航栏底部的「中文/EN」按钮切换语言。所有界面、帮助文档和训练食谱均为中英双语。

## 📦 Packaging / 打包

### One-Command Package / 一键打包

```bash
python package.py                    # Full build (frontend + backend + .exe)
python package.py --quick            # Skip UPX compression (faster)
python package.py --skip-frontend    # Use existing frontend build
python package.py --no-exe           # Build frontend + backend only
python package.py --frontend-only    # Build frontend only
```

Or double-click `package.bat` on Windows.

### Output / 输出

```
dist/AITrainerUltra.exe   # Standalone executable (~200-300 MB)
```

## 🔧 Virtual Environment / 虚拟环境

```bash
# Auto-setup on first launch / 首次启动自动创建
python start.py --setup

# Or skip / 或跳过
python start.py --skip-venv
```

## 📁 Project Structure / 项目结构

```
AITrainerUltra/
├── backend/                    # Python FastAPI backend
│   ├── api/                    # REST API (45+ endpoints) + WebSocket
│   ├── core/                   # Engine, config, registry, events, HPO, pipeline, recipes
│   ├── models/                 # 25+ model trainers
│   ├── data/                   # Dataset, processors, tokenizer utils
│   └── utils/                  # Device, optimizations, layers, checkpoint, logging, metrics
├── frontend/                   # React + Vite + Tailwind
│   ├── src/
│   │   ├── components/         # 11 panel components
│   │   ├── i18n/               # Bilingual support (zh/en)
│   │   ├── store/              # Zustand state management
│   │   └── api/                # REST + WebSocket client
├── build/                      # PyInstaller build scripts
├── package.py                  # One-command packaging script
├── package.bat                 # Windows double-click packaging
├── start.py                    # One-command launcher (auto port release + venv)
├── pyproject.toml              # Python project config
├── CHANGELOG.md                # Release history
├── CONTRIBUTING.md             # Contribution guidelines
├── SECURITY.md                 # Security policy
├── CODE_OF_CONDUCT.md          # Community standards
├── LICENSE                     # MIT License
└── README.md                   # This file (bilingual)
```

## 🤝 Contributing / 贡献指南

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

Major ways to contribute:
- **Add a new model type** — Create a trainer class extending `BaseTrainer` and register it
- **Improve the UI** — Enhance panels, add new components, fix bugs
- **Documentation** — Improve README, translations, or API docs
- **Bug reports & feature requests** — Open [GitHub Issues](https://github.com/your-org/AITrainerUltra/issues)

## 📜 License / 许可证

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

*AITrainerUltra v2.0.0 — 25+ models · 20 recipes · 45+ APIs · 7 platforms · 2 languages · 1 workflow*
