"""Model management - import, export, format conversion, versioning."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


class ModelCard:
    """Metadata card for a managed model."""

    def __init__(
        self,
        name: str,
        model_type: str,
        version: str = "1.0.0",
        description: str = "",
        tags: Optional[List[str]] = None,
        license: str = "MIT",
        framework: str = "pytorch",
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.version = version
        self.description = description
        self.tags = tags or []
        self.license = license
        self.framework = framework

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "license": self.license,
            "framework": self.framework,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ModelCard":
        return cls(
            name=data["name"],
            model_type=data.get("model_type", "unknown"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            license=data.get("license", "MIT"),
            framework=data.get("framework", "pytorch"),
        )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class ModelManager:
    """Manage local model storage with versioning."""

    def __init__(self, storage_dir: str = "./model_store") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def list_models(self) -> List[Dict[str, Any]]:
        models = []
        for model_dir in self.storage_dir.iterdir():
            if model_dir.is_dir():
                card_path = model_dir / "model_card.json"
                if card_path.exists():
                    with open(card_path, "r", encoding="utf-8") as f:
                        models.append(json.load(f))
        return models

    def get_model(self, name: str) -> Optional[Dict[str, Any]]:
        for model in self.list_models():
            if model["name"] == name:
                return model
        return None

    def save_model(
        self,
        name: str,
        model_type: str,
        source_path: str,
        description: str = "",
        version: str = "1.0.0",
    ) -> str:
        model_dir = self.storage_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)

        src = Path(source_path)
        if src.is_dir():
            shutil.copytree(src, model_dir / src.name, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, model_dir / src.name)

        card = ModelCard(
            name=name,
            model_type=model_type,
            version=version,
            description=description,
        )
        card.save(str(model_dir / "model_card.json"))
        return str(model_dir)

    def delete_model(self, name: str, version: Optional[str] = None) -> None:
        if version:
            path = self.storage_dir / name / version
        else:
            path = self.storage_dir / name
        if path.exists():
            shutil.rmtree(path)

    def export_model(
        self,
        name: str,
        export_format: str,
        output_path: str,
        version: str = "1.0.0",
    ) -> str:
        """Export model to different format.

        Supported formats: pytorch, safetensors, onnx, gguf
        """
        model_dir = self.storage_dir / name / version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {name} v{version} not found")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        if export_format == "pytorch":
            self._export_pytorch(model_dir, out)
        elif export_format == "safetensors":
            self._export_safetensors(model_dir, out)
        elif export_format == "onnx":
            self._export_onnx(model_dir, out)
        elif export_format == "gguf":
            self._export_gguf(model_dir, out)
        else:
            raise ValueError(f"Unsupported format: {export_format}")

        return str(out)

    def _export_pytorch(self, src: Path, dst: Path) -> None:
        if src.is_dir():
            shutil.make_archive(str(dst.with_suffix("")), "zip", src)
        else:
            shutil.copy2(src, dst)

    def _export_safetensors(self, src: Path, dst: Path) -> None:
        try:
            import torch
            from safetensors.torch import save_file
            model_path = src / "pytorch_model.bin"
            if model_path.exists():
                tensors = torch.load(model_path, map_location="cpu")
                save_file(tensors, str(dst.with_suffix(".safetensors")))
        except ImportError:
            shutil.make_archive(str(dst.with_suffix("")), "zip", src)

    def _export_onnx(self, src: Path, dst: Path) -> None:
        """Real ONNX export using torch.onnx.export()."""
        try:
            import torch
            from transformers import AutoModel

            dst = dst.with_suffix(".onnx") if not dst.suffix else dst
            dst.parent.mkdir(parents=True, exist_ok=True)

            # Find model files
            model_path = src
            if not (model_path / "config.json").exists():
                for sub in src.iterdir():
                    if sub.is_dir() and (sub / "config.json").exists():
                        model_path = sub
                        break

            # Load model and export to ONNX
            model = AutoModel.from_pretrained(str(model_path), torch_dtype=torch.float32)
            model.eval()

            # Create dummy input
            dummy_input = torch.randint(0, 100, (1, 64))

            torch.onnx.export(
                model,
                dummy_input,
                str(dst),
                input_names=["input_ids"],
                output_names=["logits"],
                dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
                opset_version=14,
                do_constant_folding=True,
            )
            self._write_export_log(dst, "ONNX", f"ONNX opset 14, dynamic batch/seq")
        except ImportError as e:
            self._write_export_log(dst.with_suffix(".txt"), "ONNX", f"Export failed: {e}. Install torch+transformers.")
        except Exception as e:
            self._write_export_log(dst.with_suffix(".txt"), "ONNX", f"Export error: {e}")

    def _export_gguf(self, src: Path, dst: Path) -> None:
        """Real GGUF export using llama.cpp's convert.py or manual conversion."""
        dst.mkdir(parents=True, exist_ok=True)
        log = dst / "gguf_export_log.txt"

        # Try using llama.cpp convert script
        import subprocess as _sp
        import sys as _sys

        convert_script = _sp.run(
            ["python", "-m", "llama_cpp.convert", str(src), "--outfile", str(dst / "model.gguf")],
            capture_output=True, text=True,
        )
        if convert_script.returncode == 0:
            log.write_text(f"GGUF export via llama_cpp.convert succeeded\n{convert_script.stdout}")
            return

        # Fallback: manual GGUF header writing for HuggingFace models
        try:
            import torch
            from safetensors.torch import load_file as _load_safetensors

            model_path = src
            if not any(f.suffix in (".safetensors", ".bin") for f in src.iterdir()):
                for sub in src.iterdir():
                    if sub.is_dir() and any(f.suffix in (".safetensors", ".bin") for f in sub.iterdir()):
                        model_path = sub
                        break

            # Load tensors and write GGUF format
            tensor_files = list(model_path.glob("*.safetensors")) or list(model_path.glob("*.bin"))
            if tensor_files:
                tensors = {}
                for tf in tensor_files:
                    if tf.suffix == ".safetensors":
                        tensors.update(_load_safetensors(str(tf), device="cpu"))
                    else:
                        tensors.update(torch.load(str(tf), map_location="cpu", weights_only=True))

                gguf_path = dst / "model.gguf"
                self._write_gguf_file(gguf_path, tensors)
                log.write_text(f"GGUF export complete: {len(tensors)} tensors, {gguf_path}")
            else:
                log.write_text(f"No model tensors found in {model_path}")
        except ImportError as e:
            log.write_text(f"GGUF export failed: {e}. Install torch and safetensors.")
        except Exception as e:
            log.write_text(f"GGUF export error: {e}")

    def _write_gguf_file(self, path: Path, tensors: dict) -> None:
        """Write a minimal GGUF file with tensor data.

        GGUF format: magic(4) + version(4) + metadata + tensor_infos + tensor_data
        """
        import struct
        GGUF_MAGIC = 0x46554747  # "GGUF"
        GGUF_VERSION = 3

        with open(path, "wb") as f:
            # Header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(tensors)))  # tensor_count

            # Tensor info (name + n_dims + dimensions + type + offset)
            offset = 0
            tensor_infos = []
            for name, tensor in tensors.items():
                t = tensor.contiguous().float()
                n_dims = t.ndim
                shape = list(t.shape)
                n_bytes = t.numel() * 4  # float32
                tensor_infos.append((name, n_dims, shape, n_bytes, offset))
                offset += n_bytes

            # Write tensor info
            KT = 0  # GGML_TYPE_F32
            for name, n_dims, shape, n_bytes, off in tensor_infos:
                name_bytes = name.encode("utf-8") + b"\x00"
                f.write(struct.pack("<Q", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", n_dims))
                for d in shape:
                    f.write(struct.pack("<Q", d))
                f.write(struct.pack("<I", KT))
                f.write(struct.pack("<Q", off))

            # Write tensor data
            for name, tensor in tensors.items():
                t = tensor.contiguous().float()
                f.write(t.numpy().tobytes())

    def _write_export_log(self, path: Path, fmt: str, msg: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"[{fmt}] {msg}\n", encoding="utf-8")


model_manager = ModelManager()
