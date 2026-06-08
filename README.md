# AITrainerUltra

多模型 AI 训练框架 · Multi-Model AI Training Framework

[![Version](https://img.shields.io/badge/version-2.1.0-blue)](https://github.com/AITrainerUltra)
[![Python](https://img.shields.io/badge/python-3.11-green)](https://python.org)
[![React](https://img.shields.io/badge/react-18.3-cyan)](https://react.dev)
[![Platforms](https://img.shields.io/badge/platforms-7-orange)](#compute-platforms)

---

## 快速开始 / Quick Start

### 下载安装包 / Download Installer

从 [Releases](https://github.com/AITrainerUltra/AITrainerUltra/releases) 下载 `AITrainerUltra-Installer-v2.1.0.exe`，双击安装，完成后双击桌面快捷方式即可运行。

Download the installer, run it, then launch from the desktop shortcut.

### 便携版 / Portable

下载 `AITrainerUltra.exe`，双击直接运行，无需安装。

Download and double-click — no installation needed.

### 从源码运行 / Run from Source

```bash
# 后端 / Backend
pip install -r backend/requirements.txt
python -m uvicorn backend.api.server:app --host 127.0.0.1 --port 8000 --reload

# 前端 / Frontend（另一个终端 / another terminal）
cd frontend && npm install && npm run dev
```

打开浏览器访问 `http://localhost:8000`

---

## 特性 / Features

- **25+ 模型类型** — LLM, GPT, BERT, T5, Phi, Whisper, Stable Diffusion, Flux, SVD, DETR, SAM, MoE, CLIP, BLIP 等
- **可视化工作流** — 节点编辑器，20+ 节点类型，支持自定义节点
- **实验追踪** — 训练指标记录、对比、可视化
- **模型优化** — WOQ 量化、KV 缓存卸载、Flash Attention、自适应 VRAM
- **训练食谱** — 20 个预配置方案，含 Apple Silicon 优化提示
- **中英双语** — 界面语言一键切换
- **一键打包** — 打包为单文件 .exe

---

## 支持的模型 / Supported Models

### 语言模型 / Language Models

| 模型 | 类型 | 训练方式 |
|------|------|---------|
| LLM | Decoder Transformer (GPT/LLaMA/Mistral) | Full fine-tune / LoRA / QLoRA / DPO |
| GPT | GPT-2 / GPT-Neo | Causal LM fine-tune |
| BERT | Encoder Transformer | MLM / classification / NER / QA |
| T5 | Encoder-Decoder | Text-to-text fine-tune |
| Phi | Microsoft Phi-2/3/4 | Small LLM fine-tune |
| MoE | Mixtral / DeepSeek / Qwen2-MoE | Sparse expert training |
| RNN / LSTM | Sequence models | Classification / generation |

### 视觉模型 / Vision Models

| 模型 | 类型 | 训练方式 |
|------|------|---------|
| Stable Diffusion | UNet + VAE | Text-to-image LoRA / fine-tune |
| Flux | Black Forest Labs | Text-to-image |
| CNN | ResNet / Conv2D | Transfer learning |
| DETR | Detection Transformer | Object detection |
| SAM | Segment Anything | Image segmentation |

### 视频模型 / Video Models

| 模型 | 类型 | 训练方式 |
|------|------|---------|
| Stable Video Diffusion | SVD | Image-to-video |
| I2VGen-XL | Alibaba DAMO | Text/image-to-video |
| Frame Interpolation | FILM / RIFE | Frame rate upscaling |

### 多模态 & 音频 / Multimodal & Audio

| 模型 | 类型 | 训练方式 |
|------|------|---------|
| Multimodal VLM | Custom composite | Vision-language fusion |
| CLIP | ViT + GPT-2 | Contrastive learning |
| BLIP | ViT + BERT | VQA fine-tune |
| Whisper | Encoder-Decoder | Speech recognition |
| LCM | UNet + Scheduler | Distillation |

### 从零训练 / From Scratch

| 模型 | 说明 |
|------|------|
| Scratch Transformer | Decoder-only 或 Encoder-only |
| Scratch CNN | 卷积分类器 |
| Scratch LSTM | 序列分类器 |

---

## 计算平台 / Compute Platforms

| 平台 | 自动检测 |
|------|---------|
| NVIDIA CUDA (RTX 20/30/40/50, A 系列, H100) | `torch.cuda.is_available()` |
| Apple MPS (M1–M5) | `torch.backends.mps.is_available()` |
| AMD ROCm | `torch.version.hip` |
| Google TPU | `torch_xla` |
| Huawei Ascend NPU | `torch_npu` |
| Intel XPU | `intel_extension_for_pytorch` |
| CPU | Fallback |

---

## 构建安装包 / Build Installer

```bash
# 前置条件：安装 Python 3.11、Node.js 20+
pip install PyInstaller

# 一步构建
python build_installer.py

# 仅构建便携版（不含安装程序）
python build_installer.py --portable

# 快速构建（跳过 UPX 压缩）
python build_installer.py --quick
```

输出在 `dist/` 目录。

---

## 项目结构 / Project Structure

```
AITrainerUltra/
├── backend/                    # Python FastAPI 后端
│   ├── api/                    # REST API + WebSocket
│   ├── core/                   # 引擎、配置、注册表、事件、HPO、食谱
│   ├── models/                 # 25+ 模型训练器
│   ├── data/                   # 数据集、处理器、分词器
│   └── utils/                  # 设备检测、优化、检查点、日志
├── frontend/                   # React + Vite + Tailwind 前端
│   └── src/
│       ├── components/         # 11 个面板组件
│       ├── i18n/               # 中英双语支持
│       ├── store/              # Zustand 状态管理
│       └── api/                # REST + WebSocket 客户端
├── build/                      # PyInstaller 构建脚本
├── installer/                  # Inno Setup 安装程序配置
├── build_installer.py          # 一键构建脚本
├── package.py                  # 打包脚本
├── start.py                    # 启动脚本
└── README.md
```

---

## 许可证 / License

MIT License

---

*AITrainerUltra v2.1.0 — 25+ models · 20 recipes · 7 platforms*
