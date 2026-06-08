# Contributing to AITrainerUltra

Thank you for your interest in contributing! We welcome bug fixes, new model trainers, documentation improvements, and feature enhancements.

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- npm 10+

### One-Command Setup

```bash
python start.py --setup
```

This creates a virtual environment, installs backend dependencies, and prepares the project.

### Manual Setup

```bash
# Backend
pip install -r backend/requirements.txt

# Frontend
cd frontend && npm install

# Pre-commit hooks (optional but recommended)
pip install pre-commit && pre-commit install
```

## Code Style

### Python

- **Formatter**: [ruff](https://github.com/astral-sh/ruff) (v0.4+)
- **Line length**: 120 characters
- **Linting**: `ruff check backend/` — runs on CI
- **Key rule**: No top-level `import torch` outside model files; use lazy imports inside functions
- **Device handling**: Never hardcode `torch.device("cuda" if ...)`. Use `self.device` from `BaseTrainer` or `get_torch_device()` from `device.py`

### TypeScript / React

- **Type checking**: `npx tsc --noEmit` from `frontend/`
- **State management**: Use Zustand stores for shared state; local `useState` for component-private state
- **Translations**: All UI text should use the `t(key)` function from `LanguageProvider`

### Pre-commit Checks

We use pre-commit hooks that run automatically:

```bash
pre-commit run --all-files
```

Checks include:
- **ruff**: linting and formatting for Python
- **prettier**: formatting for TypeScript, CSS, JSON, and markdown
- **trailing-whitespace / end-of-file-fixer / check-yaml**: general hygiene
- **check-added-large-files**: prevents accidentally committing large files (>500KB)

## Pull Request Process

1. **Fork** the repository and create a feature branch from `main`.
2. **Make your changes** following the code style guidelines above.
3. **Test your changes**:
   - Backend: `python -m pytest tests/ -v`
   - Frontend: `npx tsc --noEmit && npm run build`
4. **Update documentation** if adding new features (README, CLAUDE.md, translations).
5. **Create a pull request** with a clear description of what your change does.

### PR Checklist

Before submitting, ensure your PR:

- [ ] Passes all existing tests
- [ ] Adds or updates tests for new functionality
- [ ] Follows the code style (ruff, prettier)
- [ ] Updates documentation if needed
- [ ] Does not introduce breaking changes without discussion
- [ ] Includes bilingual (zh/en) translations for new UI text

## Adding a New Model Type

This is one of the most common contributions. Follow these steps:

1. **Create a trainer class** in `backend/models/` extending `BaseTrainer`
   - Implement `load_model()` and `train()` using `self.device`
   - Use lazy torch imports (wrap in `try/except ImportError`)
2. **Register the trainer** in `backend/api/server.py` via `registry.register_trainer(name, module_path, info)`
3. **Add a preset** in `backend/core/config.py` → `MODEL_PRESETS`
4. **Add frontend option** in `frontend/src/components/NodeEditor/types.ts` model_type select
5. **Add pipeline template** (optional) in `backend/core/pipeline.py`
6. **Update documentation** to list the new model in README and CLAUDE.md

## Testing

### Backend Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test
python -m pytest tests/test_backend.py::test_import_routes -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=backend/ -v
```

### Frontend Type Checking

```bash
cd frontend && npx tsc --noEmit
```

### CI Pipeline

Our GitHub Actions CI runs these checks on every push/PR:
- **Backend**: ruff lint → pytest
- **Frontend**: TypeScript check → production build

## Project Structure

```
AITrainerUltra/
├── backend/          # Python FastAPI backend
│   ├── api/          # REST API routes, schemas, WebSocket, server config
│   ├── core/         # Training engine, config, registry, events, HPO, pipeline
│   ├── models/       # Model trainers (25+ types)
│   ├── data/         # Dataset classes, processors, tokenizer utilities
│   └── utils/        # Device detection, optimizations, checkpoint, logging
├── frontend/         # React + Vite + Tailwind CSS
│   └── src/
│       ├── components/   # UI panels (NodeEditor, Training, Chat, etc.)
│       ├── i18n/         # Bilingual (zh/en) support
│       ├── store/        # Zustand state management
│       └── api/          # REST + WebSocket clients
├── build/            # PyInstaller packaging scripts
├── package.py        # One-command packaging
└── start.py          # One-command launcher
```

## Questions?

If you have questions or need help, feel free to open a [Discussion](https://github.com/your-org/AITrainerUltra/discussions) or an issue.
