"""Real data loaders from HuggingFace datasets and local files.

Replaces dummy data in trainers with actual datasets.
Auto-falls back to synthetic data when dataset is unavailable.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def load_real_dataset(
    dataset_path: str = "",
    model_type: str = "llm",
    split: str = "train",
    max_samples: int = 500,
    **kwargs: Any,
) -> Optional[torch.utils.data.Dataset]:
    """Load a real dataset based on model type.

    Tries HuggingFace datasets first, falls back to synthetic.
    """
    # Try HuggingFace datasets
    try:
        from datasets import load_dataset as hf_load

        if dataset_path:
            ds = hf_load(dataset_path, split=split)
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            return ds
    except Exception:
        pass

    # Try common default datasets per model type
    defaults = {
        "llm": "wikitext",
        "gpt": "wikitext",
        "bert": "imdb",
        "cnn": "cifar10",
        "lcm": "cifar10",
        "multimodal": "cifar10",
        "clip": "cifar10",
        "blip": "cifar10",
        "rnn": "imdb",
        "lstm": "imdb",
    }
    default = defaults.get(model_type, "")
    if default:
        try:
            from datasets import load_dataset as hf_load
            ds = hf_load(default, split=split if split != "train" else "train")
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))
            return ds
        except Exception:
            pass

    return None


def get_text_dataset(
    dataset_path: str = "",
    model_type: str = "llm",
    split: str = "train",
    max_samples: int = 500,
    text_column: str = "text",
) -> Tuple[List[str], Optional[torch.utils.data.Dataset]]:
    """Get a text dataset for LLM/GPT/BERT training.

    Returns:
        (texts, dataset) — texts is a list of strings, dataset is optional HF Dataset
    """
    ds = load_real_dataset(dataset_path, model_type, split, max_samples)

    if ds is not None:
        try:
            if text_column in ds.column_names:
                texts = [item[text_column] for item in ds if item.get(text_column)]
                if texts:
                    return texts, ds
            # Try common column names
            for col in ["text", "sentence", "content", "document", "review"]:
                if col in ds.column_names:
                    texts = [item[col] for item in ds if item.get(col)]
                    if texts:
                        return texts, ds
        except Exception:
            pass

    # Fallback: synthetic text data
    texts = [
        f"The future of artificial intelligence depends on our ability to build larger and more capable models. "
        f"Training sample {i}: deep learning has revolutionized natural language processing."
        for i in range(max_samples)
    ]
    return texts, None


def get_classification_dataset(
    dataset_path: str = "",
    model_type: str = "bert",
    split: str = "train",
    max_samples: int = 500,
    num_classes: int = 2,
) -> Tuple[List[str], List[int], Optional[Any]]:
    """Get a text classification dataset.

    Returns:
        (texts, labels, dataset)
    """
    ds = load_real_dataset(dataset_path, model_type, split, max_samples)

    if ds is not None:
        try:
            text_col = None
            label_col = None
            for c in ["text", "sentence", "review", "content"]:
                if c in ds.column_names: text_col = c; break
            for c in ["label", "labels", "sentiment", "category"]:
                if c in ds.column_names: label_col = c; break

            if text_col and label_col:
                texts = []
                labels = []
                for item in ds:
                    if item.get(text_col) is not None and item.get(label_col) is not None:
                        texts.append(str(item[text_col]))
                        labels.append(int(item[label_col]))
                if texts:
                    return texts[:max_samples], labels[:max_samples], ds
        except Exception:
            pass

    # Fallback
    texts = [f"Classification sample {i}: this is a test document." for i in range(max_samples)]
    labels = [i % num_classes for i in range(max_samples)]
    return texts, labels, None


def get_image_dataset(
    dataset_path: str = "",
    model_type: str = "cnn",
    split: str = "train",
    max_samples: int = 100,
    image_size: Tuple[int, int, int] = (3, 32, 32),
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Get an image dataset.

    Returns:
        (images_tensor, labels_tensor_or_None)
    """
    ds = load_real_dataset(dataset_path, model_type, split, max_samples)

    if ds is not None:
        try:
            import torchvision.transforms as T
            transform = T.Compose([T.Resize(image_size[1:]), T.ToTensor()])
            images = []
            labels = []
            has_label = "label" in ds.column_names

            for i, item in enumerate(ds):
                if i >= max_samples: break
                if "image" in item or "img" in item:
                    img = item.get("image") or item.get("img")
                    images.append(transform(img))
                    if has_label:
                        labels.append(int(item["label"]))

            if images:
                import torch
                img_tensor = torch.stack(images)
                lbl_tensor = torch.tensor(labels) if labels else None
                return img_tensor, lbl_tensor
        except Exception:
            pass

    # Fallback
    import torch
    images = torch.randn(min(max_samples, 64), *image_size)
    labels = torch.randint(0, 10, (min(max_samples, 64),))
    return images, labels


def create_dataloader(
    texts: List[str],
    tokenizer: Optional[Any] = None,
    batch_size: int = 8,
    max_length: int = 128,
    labels: Optional[List[int]] = None,
) -> torch.utils.data.DataLoader:
    """Create a DataLoader from text data with optional tokenization."""
    import torch
    if tokenizer is not None:
        encoded = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt",
        )
        if labels is not None:
            dataset = torch.utils.data.TensorDataset(
                encoded["input_ids"], encoded["attention_mask"],
                torch.tensor(labels),
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                encoded["input_ids"], encoded["attention_mask"],
                encoded["input_ids"].clone(),
            )
    else:
        data = torch.randint(1, 1000, (len(texts), max_length))
        if labels is not None:
            dataset = torch.utils.data.TensorDataset(data, torch.tensor(labels))
        else:
            dataset = torch.utils.data.TensorDataset(data, data.clone())

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Common dataset presets
DATASET_PRESETS = {
    "wikitext-2": {
        "path": "wikitext", "name": "wikitext-2-raw-v1",
        "description": "Wikipedia text for language modeling",
        "size": "~4MB", "samples": "~36K",
    },
    "imdb": {
        "path": "imdb",
        "description": "IMDB movie reviews for sentiment analysis",
        "size": "~80MB", "samples": "25K",
    },
    "sst2": {
        "path": "glue", "name": "sst2",
        "description": "Stanford Sentiment Treebank",
        "size": "~7MB", "samples": "67K",
    },
    "cifar10": {
        "path": "cifar10",
        "description": "CIFAR-10 image classification",
        "size": "~180MB", "samples": "60K",
    },
    "cnn_dailymail": {
        "path": "cnn_dailymail", "name": "3.0.0",
        "description": "News articles for summarization",
        "size": "~500MB", "samples": "300K",
    },
    "coco": {
        "path": "coco_captions",
        "description": "COCO captions for multimodal training",
        "size": "~25GB", "samples": "330K",
    },
}
