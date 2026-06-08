"""Model serving - optimized inference API."""

from __future__ import annotations

import time
from typing import Any, AsyncGenerator, Dict, List, Optional


class InferenceServer:
    """Optimized model inference server."""

    def __init__(self) -> None:
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.configs: Dict[str, Dict[str, Any]] = {}

    def load(
        self,
        model_id: str,
        model: Any,
        tokenizer: Any = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.models[model_id] = model
        if tokenizer:
            self.tokenizers[model_id] = tokenizer
        self.configs[model_id] = config or {}
        if hasattr(model, "eval"):
            model.eval()

    def unload(self, model_id: str) -> None:
        self.models.pop(model_id, None)
        self.tokenizers.pop(model_id, None)
        self.configs.pop(model_id, None)
        import torch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def list_models(self) -> List[Dict[str, Any]]:
        return [
            {"id": mid, "config": self.configs.get(mid, {})}
            for mid in self.models
        ]

    def is_loaded(self, model_id: str) -> bool:
        return model_id in self.models

    def generate(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        stream: bool = False,
    ) -> Any:
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not loaded")

        model = self.models[model_id]
        tokenizer = self.tokenizers.get(model_id)

        if tokenizer is None:
            return {"text": prompt, "error": "No tokenizer"}

        inputs = tokenizer(prompt, return_tensors="pt")

        import torch
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        return {"text": text, "usage": {"prompt_tokens": len(inputs[0]), "completion_tokens": len(generated)}}

    async def generate_stream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        if model_id not in self.models:
            yield f"Error: Model {model_id} not loaded"
            return

        tokenizer = self.tokenizers.get(model_id)
        if not tokenizer:
            yield "Error: No tokenizer"
            return

        inputs = tokenizer(prompt, return_tensors="pt")
        model = self.models[model_id]

        import torch
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = model(**inputs)
            next_token = outputs.logits[0, -1].argmax(-1).unsqueeze(0)
            inputs["input_ids"] = torch.cat([inputs["input_ids"], next_token.unsqueeze(0)], dim=-1)
            token = tokenizer.decode(next_token, skip_special_tokens=True)
            if token:
                yield token
            if next_token.item() == tokenizer.eos_token_id:
                break

    def get_performance_stats(self, model_id: str) -> Dict[str, Any]:
        if model_id not in self.models:
            return {"error": "Model not loaded"}
        model = self.models[model_id]
        param_count = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {
            "model_id": model_id,
            "total_params": param_count,
            "trainable_params": trainable,
            "device": str(next(model.parameters()).device) if list(model.parameters()) else "unknown",
        }


# Global inference server
inference_server = InferenceServer()
