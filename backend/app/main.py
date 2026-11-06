"""GestureLab API: local artifacts, real predictions, zero upload retention."""
import json
import logging
import math
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import torch

from backend.app.explainability import grad_cam
from backend.app.preprocessing import MAX_UPLOAD_BYTES, decode_image
from backend.app.schemas import ExperimentRun, ExperimentsResponse, HealthResponse, ModelsResponse, PredictionResponse
from backend.app.services import CLASSES, MODEL_NAMES, ModelRegistry
from backend.app.upload_limit import UploadLimitMiddleware

ROOT = Path(__file__).resolve().parents[2]
RUN_IDS = ("simple_cnn", "resnet18", "mobilenetv4", "mobilenetv4_noaug",
           "resnet18_finetuned", "mobilenetv4_finetuned")
logger = logging.getLogger(__name__)


def load_experiment(metrics_dir: Path, run_id: str) -> ExperimentRun:
    if run_id not in RUN_IDS:
        raise FileNotFoundError("Unknown experiment")
    path = metrics_dir / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError("Experiment has not been evaluated")
    raw = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Nonfinite value {value}")))
    result = ExperimentRun.model_validate(raw)
    if result.run_id != run_id or result.model_id not in MODEL_NAMES:
        raise ValueError("Experiment ID does not match its artifact")
    if len(result.confusion_matrix) != 4 or any(len(row) != 4 or any(n < 0 for n in row) for row in result.confusion_matrix):
        raise ValueError("Experiment confusion matrix is invalid")
    for name in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
        value = result.metrics.get(name)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid measured metric {name}")
    return result


def predict_image(registry: ModelRegistry, data: bytes, model_id: str, explain: bool) -> PredictionResponse:
    if model_id not in MODEL_NAMES:
        raise HTTPException(status_code=400, detail="Unknown model_id")
    try:
        image = decode_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        loaded = registry.load(model_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Trained checkpoint could not be loaded: {exc}") from exc
    from ml.preprocessing import prepare_image, image_tensor
    input_size = loaded.metadata["input_size"]
    prepared = prepare_image(image, input_size=input_size)
    inputs = image_tensor(prepared, mean=loaded.metadata["mean"],
                          std=loaded.metadata["std"], input_size=input_size).unsqueeze(0).to(registry.device)
    try:
        with registry.inference_slot():
            start = time.perf_counter()
            explanation = None
            if explain:
                try:
                    logits, explanation = grad_cam(loaded.model, inputs, prepared)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
            else:
                with torch.no_grad():
                    logits = loaded.model(inputs)
            elapsed = (time.perf_counter() - start) * 1000
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "2"}) from exc
    if logits.shape != (1, len(CLASSES)) or not torch.isfinite(logits).all():
        raise HTTPException(status_code=503, detail="Model returned invalid classification logits")
    probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().tolist()
    return PredictionResponse(
        model_id=model_id, version=loaded.version, label=CLASSES[int(logits.argmax(1).item())],
        probabilities=dict(zip(CLASSES, probabilities)), latency_ms=elapsed,
        device=registry.device, explanation=explanation,
        preprocessing={"input_size": input_size, "channel_order": "RGB", "padding": [128, 128, 128],
                       "mean": loaded.metadata["mean"], "std": loaded.metadata["std"],
                       "roi": "User-selected hand crop; no automatic hand detector",
                       "latency_scope": "Model inference including Grad-CAM when requested; excludes queue, decode and network"},
    )


def create_app(checkpoint_dir: Path | None = None, metrics_dir: Path | None = None) -> FastAPI:
    application = FastAPI(title="GestureLab", version="1.0.0")
    # CPU is an intentional reproducible default; inference never initiates downloads.
    torch.set_num_threads(min(4, torch.get_num_threads()))
    registry = ModelRegistry(checkpoint_dir or ROOT / "artifacts" / "checkpoints", device="cpu")
    metric_path = metrics_dir or ROOT / "artifacts" / "metrics"
    application.state.registry = registry
    origins = [item.strip() for item in os.getenv("GESTURELAB_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if item.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False,
                               allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

    application.add_middleware(UploadLimitMiddleware)

    @application.middleware("http")
    async def private_responses(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @application.get("/api/health", response_model=HealthResponse)
    async def health():
        models = await run_in_threadpool(registry.describe)
        ready = all(item.available for item in models)
        return HealthResponse(status="ready" if ready else "degraded", device=registry.device, ready=ready, models=models)

    @application.get("/api/models", response_model=ModelsResponse)
    async def models():
        return ModelsResponse(classes=CLASSES, models=await run_in_threadpool(registry.describe))

    @application.post("/api/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...), model_id: str = Form(...), explain: bool = Form(False)):
        try:
            data = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(data) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Image exceeds the 10 MB upload limit")
            return await run_in_threadpool(predict_image, registry, data, model_id, explain)
        finally:
            await file.close()

    @application.get("/api/experiments", response_model=ExperimentsResponse)
    async def experiments():
        def read_all():
            runs = []
            for run_id in RUN_IDS:
                try:
                    runs.append(load_experiment(metric_path, run_id))
                except FileNotFoundError:
                    pass
                except (ValueError, TypeError, KeyError) as exc:
                    logger.warning("Ignoring invalid experiment artifact %s: %s", run_id, exc)
            return ExperimentsResponse(available=bool(runs), runs=runs)
        return await run_in_threadpool(read_all)

    @application.get("/api/experiments/{run_id}", response_model=ExperimentRun)
    async def experiment(run_id: str):
        try:
            return await run_in_threadpool(load_experiment, metric_path, run_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=503, detail="Experiment artifact is invalid; regenerate evaluation") from exc

    return application


app = create_app()
