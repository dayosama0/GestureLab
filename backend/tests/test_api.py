"""API error-path tests and explicitly test-only trained checkpoint integration."""
import base64
import hashlib
import io
import json

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient
import torch

from backend.app.main import ROOT, create_app
from backend.app.preprocessing import MAX_UPLOAD_BYTES, decode_image
from backend.app.services import CLASSES, MEAN, STD, MODEL_NAMES, PREPROCESSING, ModelRegistry


def png_bytes(image=None):
    if image is None:
        values = np.random.default_rng(42).integers(0, 256, (224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(values)
    result = io.BytesIO()
    image.save(result, format="PNG")
    return result.getvalue()


@pytest.fixture
def empty_client(tmp_path):
    return TestClient(create_app(tmp_path / "checkpoints", tmp_path / "metrics"))


@pytest.fixture
def trained_client(tmp_path):
    # This fixture performs actual optimization and is strictly isolated from
    # production artifacts. It tests mechanics, never scientific performance.
    from ml.models import create_model
    torch.manual_seed(42)
    model = create_model("simple_cnn", pretrained=False)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(2):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 3, 224, 224)), torch.arange(4))
        loss.backward()
        optimizer.step()
    metadata = {"model_id": "simple_cnn", "trained": True, "classes": CLASSES,
                "input_size": 224, "mean": MEAN, "std": STD, "best_epoch": 2,
                "seed": 42, "augmentation": "none", "pretrained_source": "none",
                "provenance": "API test-only temporary synthetic training fixture; not a performance experiment"}
    path = tmp_path / "checkpoints"
    path.mkdir()
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path / "simple_cnn.pt")
    return TestClient(create_app(path, tmp_path / "metrics"))


def test_missing_models_never_predict(empty_client):
    health = empty_client.get("/api/health").json()
    assert health["status"] == "degraded"
    assert health["ready"] is False
    models = empty_client.get("/api/models").json()
    assert models["classes"] == CLASSES
    assert len(models["models"]) == 5
    assert not any(m["available"] for m in models["models"])
    response = empty_client.post("/api/predict", files={"file": ("hand.png", png_bytes(), "image/png")}, data={"model_id": "simple_cnn"})
    assert response.status_code == 503
    assert "checkpoint unavailable" in response.json()["detail"]


@pytest.mark.parametrize("payload", [b"", b"not an image", b"\xff\xd8truncated"])
def test_bad_bytes_rejected(empty_client, payload):
    response = empty_client.post("/api/predict", files={"file": ("valid.jpg", payload, "image/jpeg")}, data={"model_id": "simple_cnn"})
    assert response.status_code == 400


def test_oversized_upload(empty_client):
    response = empty_client.post("/api/predict", files={"file": ("huge.png", b"x" * (MAX_UPLOAD_BYTES + 1), "image/png")}, data={"model_id": "simple_cnn"})
    assert response.status_code == 413


def test_unknown_model_and_missing_form(empty_client):
    assert empty_client.post("/api/predict", files={"file": ("hand.png", png_bytes())}, data={"model_id": "../secret"}).status_code == 400
    assert empty_client.post("/api/predict").status_code == 422


def test_transparency_and_exif_are_normalized():
    image = Image.new("RGBA", (80, 40), (255, 0, 0, 0))
    decoded = decode_image(png_bytes(image))
    assert decoded.mode == "RGB" and decoded.getpixel((0, 0)) == (128, 128, 128)
    image = Image.new("RGB", (80, 40), "red")
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    assert decode_image(buffer.getvalue()).size == (40, 80)


def test_trained_checkpoint_prediction_and_reload(trained_client):
    data = png_bytes()
    options = {"files": {"file": ("hand.png", data)}, "data": {"model_id": "simple_cnn", "explain": "false"}}
    response = trained_client.post("/api/predict", **options)
    assert response.status_code == 200, response.text
    result = response.json()
    assert set(result["probabilities"]) == set(CLASSES)
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert result["label"] in CLASSES and result["latency_ms"] > 0
    assert result["explanation"] is None
    registry = trained_client.app.state.registry
    path = registry.checkpoint_dir / "simple_cnn.pt"
    assert result["version"] == hashlib.sha256(path.read_bytes()).hexdigest()
    registry._cache.clear()
    restored = trained_client.post("/api/predict", **options).json()
    assert restored["probabilities"] == result["probabilities"]


def test_genuine_cam_nonconstant_and_hooks_removed(trained_client):
    images = [png_bytes(), png_bytes(Image.new("RGB", (70, 160), "white"))]
    explanations = []
    for data in images:
        response = trained_client.post("/api/predict", files={"file": ("hand.png", data)}, data={"model_id": "simple_cnn", "explain": "true"})
        assert response.status_code == 200, response.text
        explanation = response.json()["explanation"]
        image = Image.open(io.BytesIO(base64.b64decode(explanation["image_base64"])))
        assert image.size == (224, 224)
        prepared = Image.open(io.BytesIO(base64.b64decode(explanation["input_image_base64"])))
        assert prepared.size == image.size
        if data == images[1]:
            assert prepared.getpixel((0, 112)) == (128, 128, 128)
            assert prepared.getpixel((112, 112)) == (255, 255, 255)
        assert np.asarray(image).std() > 0
        explanations.append(explanation["image_base64"])
    assert explanations[0] != explanations[1]
    model = trained_client.app.state.registry.load("simple_cnn").model
    assert all(not module._forward_hooks for module in model.modules())


def test_untrained_or_wrong_mapping_checkpoint_rejected(tmp_path):
    from ml.models import create_model
    metadata = {"model_id": "simple_cnn", "trained": False, "classes": CLASSES, "input_size": 224, "mean": MEAN, "std": STD, "best_epoch": 1}
    path = tmp_path / "simple_cnn.pt"
    torch.save({"state_dict": create_model("simple_cnn").state_dict(), "metadata": metadata}, path)
    with pytest.raises(ValueError, match="provenance"):
        ModelRegistry(tmp_path).load("simple_cnn")
    metadata["trained"] = True
    metadata["classes"] = list(reversed(CLASSES))
    torch.save({"state_dict": create_model("simple_cnn").state_dict(), "metadata": metadata}, path)
    with pytest.raises(ValueError, match="class order"):
        ModelRegistry(tmp_path).load("simple_cnn")


@pytest.mark.parametrize("model_id", list(MODEL_NAMES))
def test_each_model_rejects_unapproved_preprocessing_before_restore(tmp_path, model_id):
    policy = PREPROCESSING[model_id]
    metadata = {"model_id": model_id, "trained": True, "classes": CLASSES,
                "input_size": policy["input_size"], "mean": policy["mean"],
                "std": policy["std"], "best_epoch": 1, "pretrained_source": "test-only fixture"}
    path = tmp_path / f"{model_id}.pt"
    for change in ({"input_size": 225}, {"mean": [0, 0, 0]}, {"std": [1, 1, 1]}):
        torch.save({"state_dict": {}, "metadata": {**metadata, **change}}, path)
        with pytest.raises(ValueError, match="Unsupported checkpoint preprocessing"):
            ModelRegistry(tmp_path).load(model_id)


def test_mobile_finetuned_dynamic_256_norm_and_cam(tmp_path, monkeypatch):
    # A genuinely optimized tiny test-only network isolates API preprocessing
    # and dynamic CAM sizing from expensive scientific model training.
    import ml.models

    class TinySpatialClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)
            self.head = torch.nn.Linear(4, 4)
            self.last_input = None
            torch.nn.init.constant_(self.conv.weight, .1)
            torch.nn.init.constant_(self.conv.bias, 1)
            torch.nn.init.constant_(self.head.weight, .1)
            torch.nn.init.zeros_(self.head.bias)

        def forward(self, images):
            self.last_input = images.detach().clone()
            return self.head(torch.relu(self.conv(images)).mean((2, 3)))

    trained = TinySpatialClassifier()
    optimizer = torch.optim.SGD(trained.parameters(), lr=.01)
    optimizer.zero_grad()
    torch.nn.functional.cross_entropy(trained(torch.ones(1, 3, 32, 32)), torch.tensor([0])).backward()
    optimizer.step()
    monkeypatch.setattr(ml.models, "create_model", lambda model_id, pretrained=False: TinySpatialClassifier())
    monkeypatch.setattr(ml.models, "target_layer", lambda model: model.conv)
    model_id = "mobilenetv4_finetuned"
    metadata = {"model_id": model_id, "trained": True, "classes": CLASSES,
                **PREPROCESSING[model_id], "best_epoch": 1,
                "pretrained_source": "test-only optimized fixture, no scientific metric"}
    torch.save({"state_dict": trained.state_dict(), "metadata": metadata}, tmp_path / f"{model_id}.pt")
    client = TestClient(create_app(tmp_path, tmp_path / "metrics"))
    image = Image.new("RGB", (80, 80), (64, 128, 192))
    response = client.post("/api/predict", files={"file": ("hand.png", png_bytes(image))},
                           data={"model_id": model_id, "explain": "true"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["preprocessing"]["input_size"] == 256
    assert result["preprocessing"]["mean"] == [.5, .5, .5]
    assert result["preprocessing"]["std"] == [.5, .5, .5]
    overlay = Image.open(io.BytesIO(base64.b64decode(result["explanation"]["image_base64"])))
    assert overlay.size == (256, 256)
    restored = client.app.state.registry.load(model_id).model
    assert restored.last_input.shape == (1, 3, 256, 256)
    assert restored.last_input[0, :, 128, 128].tolist() == pytest.approx(
        [(channel / 255 - .5) / .5 for channel in (64, 128, 192)], abs=1e-6)


def test_experiments_empty_invalid_and_valid(empty_client, tmp_path):
    assert empty_client.get("/api/experiments").json() == {"available": False, "runs": []}
    assert empty_client.get("/api/experiments/not_a_run").status_code == 404
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    client = TestClient(create_app(tmp_path / "checkpoints", metrics))
    artifact = {"run_id": "simple_cnn", "model_id": "simple_cnn", "metadata": {"test_fixture": True},
                "history": [], "metrics": {k: .5 for k in ("accuracy", "f1_macro", "precision_macro", "recall_macro")},
                "confusion_matrix": [[1, 0, 0, 0]] * 4, "latency": {"mean_ms": 1}, "stress": {}}
    (metrics / "simple_cnn.json").write_text(json.dumps(artifact), encoding="utf-8")
    assert client.get("/api/experiments").json()["available"] is True
    artifact["metrics"]["accuracy"] = 2
    (metrics / "simple_cnn.json").write_text(json.dumps(artifact), encoding="utf-8")
    assert client.get("/api/experiments").json()["runs"] == []
    assert client.get("/api/experiments/simple_cnn").status_code == 503


@pytest.mark.parametrize("model_id", ["resnet18_finetuned", "mobilenetv4_finetuned"])
def test_expanded_experiments_preserve_labelled_holdouts(tmp_path, model_id):
    metric_values = {key: .5 for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro")}
    original = {"metrics": metric_values, "confusion_matrix": [[15, 0, 0, 0]] * 4, "count": 60}
    expanded = {"metrics": {**metric_values, "accuracy": .75}, "count": 100}
    artifact = {"run_id": model_id, "model_id": model_id, "metadata": {"test_fixture": True},
                "history": [], "metrics": metric_values,
                "confusion_matrix": original["confusion_matrix"], "latency": {}, "stress": {},
                "original_test": original, "expanded_test": expanded}
    (tmp_path / f"{model_id}.json").write_text(json.dumps(artifact), encoding="utf-8")
    client = TestClient(create_app(tmp_path / "checkpoints", tmp_path))
    response = client.get(f"/api/experiments/{model_id}")
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["original_test"] == original
    assert result["expanded_test"] == expanded
    assert result["metrics"]["accuracy"] == original["metrics"]["accuracy"]
    assert result["metrics"]["accuracy"] != result["expanded_test"]["metrics"]["accuracy"]


@pytest.mark.parametrize("model_id", list(MODEL_NAMES))
def test_real_heldout_checkpoint_integration(model_id):
    checkpoint = ROOT / "artifacts" / "checkpoints" / f"{model_id}.pt"
    candidates = sorted((ROOT / "data" / "ready_dataset" / "test").rglob("*.jpg"))
    if not checkpoint.is_file() or not candidates:
        pytest.skip("Real trained checkpoint or private held-out test image unavailable")
    client = TestClient(create_app())
    response = client.post("/api/predict", files={"file": ("heldout.jpg", candidates[0].read_bytes())}, data={"model_id": model_id, "explain": "true"})
    assert response.status_code == 200, response.text
    result = response.json()
    assert sum(result["probabilities"].values()) == pytest.approx(1, abs=1e-6)
    assert result["explanation"]["width"] == PREPROCESSING[model_id]["input_size"]
    assert result["explanation"]["height"] == PREPROCESSING[model_id]["input_size"]
