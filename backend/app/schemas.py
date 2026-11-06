"""Public API response contracts. No demonstration predictions are exposed."""
from typing import Any
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    id: str
    name: str
    available: bool
    reason: str | None = None
    version: str | None = None
    metadata: dict[str, Any] | None = None


class ModelsResponse(BaseModel):
    classes: list[str]
    models: list[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    device: str
    ready: bool
    models: list[ModelInfo]


class Explanation(BaseModel):
    image_base64: str
    input_image_base64: str | None = None
    mime_type: str = "image/png"
    width: int
    height: int
    method: str = "Grad-CAM (gradient-based approximation)"


class PredictionResponse(BaseModel):
    model_id: str
    version: str
    label: str
    probabilities: dict[str, float]
    latency_ms: float = Field(ge=0)
    device: str
    preprocessing: dict[str, Any]
    explanation: Explanation | None = None


class HistoryPoint(BaseModel):
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float


class ExperimentRun(BaseModel):
    run_id: str
    model_id: str
    metadata: dict[str, Any]
    history: list[HistoryPoint]
    metrics: dict[str, Any]
    confusion_matrix: list[list[int]]
    latency: dict[str, Any]
    stress: dict[str, Any] = Field(default_factory=dict)
    # Primary metrics/confusion_matrix retain the original 60-image test set.
    # Additional holdouts remain labelled and are passed through only if saved.
    original_test: dict[str, Any] | None = None
    expanded_test: dict[str, Any] | None = None


class ExperimentsResponse(BaseModel):
    available: bool
    runs: list[ExperimentRun]
