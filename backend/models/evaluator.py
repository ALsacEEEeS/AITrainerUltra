"""Model evaluation suite - benchmarks and metrics."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Dict, List, Optional


class MetricCalculator:

    @staticmethod
    def perplexity(loss: float) -> float:
        """Compute perplexity from cross-entropy loss."""
        return float(math.exp(min(loss, 20)))

    @staticmethod
    def accuracy(predictions: List[int], labels: List[int]) -> float:
        """Compute accuracy score."""
        if not predictions or len(predictions) != len(labels):
            return 0.0
        correct = sum(1 for p, l in zip(predictions, labels) if p == l)
        return correct / len(predictions)

    @staticmethod
    def f1_score(predictions: List[int], labels: List[int], average: str = "binary") -> float:
        """Compute F1 score."""
        if len(predictions) != len(labels) or not predictions:
            return 0.0
        tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)


class Evaluator:

    def __init__(self, model: Any = None, tokenizer: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.results: Dict[str, Any] = {}
        self.metrics = MetricCalculator()

    def set_model(self, model: Any, tokenizer: Any = None) -> None:
        self.model = model
        self.tokenizer = tokenizer

    async def evaluate_text_generation(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate text generation quality."""
        if not self.model:
            return {"error": "No model loaded"}

        import torch
        self.model.eval()
        total_time = 0.0
        total_tokens = 0

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            start = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            elapsed = time.time() - start
            generated = outputs[0][inputs.input_ids.shape[1]:]
            total_time += elapsed
            total_tokens += len(generated)

        tokens_per_sec = total_tokens / total_time if total_time > 0 else 0
        return {
            "total_prompts": len(prompts),
            "total_tokens": total_tokens,
            "total_time": round(total_time, 2),
            "tokens_per_sec": round(tokens_per_sec, 2),
            "avg_time_per_prompt": round(total_time / len(prompts), 3),
        }

    async def evaluate_perplexity(
        self,
        texts: List[str],
        max_length: int = 512,
    ) -> Dict[str, Any]:
        """Evaluate perplexity on a corpus."""
        if not self.model or not self.tokenizer:
            return {"error": "No model or tokenizer loaded"}

        import torch
        self.model.eval()
        total_loss = 0.0
        total_steps = 0

        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=max_length,
            )
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item()
            total_steps += 1

        avg_loss = total_loss / total_steps
        return {
            "avg_loss": round(avg_loss, 4),
            "perplexity": round(self.metrics.perplexity(avg_loss), 4),
            "num_samples": total_steps,
        }

    async def evaluate_classification(
        self,
        test_data: List[Dict[str, Any]],
        predict_fn: Callable[[Any], int],
    ) -> Dict[str, Any]:
        """Evaluate classification accuracy and F1."""
        predictions = []
        labels = []

        for item in test_data:
            pred = predict_fn(item["input"])
            predictions.append(pred)
            labels.append(item["label"])

        return {
            "accuracy": round(self.metrics.accuracy(predictions, labels), 4),
            "f1_score": round(self.metrics.f1_score(predictions, labels), 4),
            "num_samples": len(test_data),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "results": self.results,
            "num_evals": len(self.results),
        }


class BenchmarkRunner:

    BENCHMARKS = {
        "wikitext2": "perplexity",
        "glue-mrpc": "accuracy",
        "glue-sst2": "accuracy",
        "hellaswag": "accuracy",
    }

    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator
        self.results: Dict[str, Any] = {}

    async def run(self, benchmark: str) -> Dict[str, Any]:
        if benchmark == "wikitext2":
            return await self._run_wikitext2()
        elif benchmark.startswith("glue-"):
            return await self._run_glue(benchmark)
        else:
            return {"error": f"Unknown benchmark: {benchmark}"}

    async def run_all(self) -> Dict[str, Any]:
        for benchmark in self.BENCHMARKS:
            self.results[benchmark] = await self.run(benchmark)
        return self.results

    async def _run_wikitext2(self) -> Dict[str, Any]:
        try:
            from datasets import load_dataset
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            texts = dataset["text"][:50]
            return await self.evaluator.evaluate_perplexity(texts)
        except Exception as e:
            return {"error": str(e), "note": "Install datasets: pip install datasets"}

    async def _run_glue(self, benchmark: str) -> Dict[str, Any]:
        task = benchmark.replace("glue-", "")
        try:
            from datasets import load_dataset
            dataset = load_dataset("glue", task, split="validation")
            return {"benchmark": benchmark, "status": "not_implemented"}
        except Exception as e:
            return {"error": str(e)}
