"""Load trusted, locally trained artifacts; never download serving weights."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import threading
from typing import Any

import torch
from backend.app.schemas import ModelInfo

CLASSES = ["fist", "open_palm", "peace", "thumbs_up"]
MODEL_NAMES = {
    "simple_cnn": "A1 SimpleCNN",
    "resnet18": "A1 ResNet18",
    "mobilenetv4": "MobileNetV4 Conv Small",
    "resnet18_finetuned": "ResNet18 · expanded-data fine-tuning",
    "mobilenetv4_finetuned": "MobileNetV4 · expanded-data fine-tuning",
}
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PREPROCESSING = {
    model_id: {"input_size": 224, "mean": MEAN, "std": STD}
    for model_id in MODEL_NAMES
}
PREPROCESSING["mobilenetv4_finetuned"] = {
    "input_size": 256, "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5],
}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    metadata: dict[str, Any]
    version: str


class ModelRegistry:
    def __init__(self, checkpoint_dir: Path, device: str = "cpu"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device
        self._cache: dict[str, LoadedModel] = {}
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()

    @contextmanager
    def inference_slot(self):
        if not self._inference_lock.acquire(timeout=20):
            raise TimeoutError("Inference service is busy. Retry after the current request finishes.")
        try:
            yield
        finally:
            self._inference_lock.release()

    def load(self, model_id: str) -> LoadedModel:
        if model_id not in MODEL_NAMES:
            raise KeyError("Unknown model ID")
        with self._lock:
            if model_id in self._cache:
                return self._cache[model_id]
            path = self.checkpoint_dir / f"{model_id}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Trained checkpoint unavailable: artifacts/checkpoints/{model_id}.pt. Run python -m ml.train --help for preparation commands.")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint must contain state_dict and metadata")
            metadata = checkpoint.get("metadata", {})
            if metadata.get("trained") is not True or metadata.get("model_id") != model_id:
                raise ValueError("Checkpoint has no matching trained-model provenance")
            if metadata.get("classes") != CLASSES:
                raise ValueError("Checkpoint class order does not match the fixed dataset mapping")
            expected = PREPROCESSING[model_id]
            if (metadata.get("input_size") != expected["input_size"]
                    or list(metadata.get("mean", [])) != expected["mean"]
                    or list(metadata.get("std", [])) != expected["std"]):
                raise ValueError(
                    f"Unsupported checkpoint preprocessing for {model_id}; "
                    f"expected RGB gray-letterbox {expected['input_size']}, "
                    f"mean={expected['mean']}, std={expected['std']}"
                )
            if not isinstance(metadata.get("best_epoch"), int) or metadata["best_epoch"] < 1:
                raise ValueError("Checkpoint is missing its selected training epoch")
            if model_id != "simple_cnn" and not metadata.get("pretrained_source"):
                raise ValueError("Transfer model is missing pretrained source provenance")
            from ml.models import create_model
            model = create_model(model_id, pretrained=False)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.to(self.device).eval()
            version = hashlib.sha256(path.read_bytes()).hexdigest()
            item = LoadedModel(model=model, metadata=metadata, version=version)
            self._cache[model_id] = item
            return item

    def describe(self) -> list[ModelInfo]:
        result = []
        for model_id, name in MODEL_NAMES.items():
            try:
                loaded = self.load(model_id)
                result.append(ModelInfo(id=model_id, name=name, available=True, version=loaded.version, metadata=loaded.metadata))
            except Exception as exc:
                result.append(ModelInfo(id=model_id, name=name, available=False, reason=str(exc)))
        return result
