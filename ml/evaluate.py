"""Measured clean/stress classification scores and CPU timing."""
import io
import math
import platform
import time
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from .models import CLASSES


@torch.inference_mode()
def evaluate(model, dataset, batch_size=32):
    model.eval()
    truth, predictions = [], []
    for images, labels in DataLoader(dataset, batch_size=batch_size):
        predictions.extend(model(images).argmax(1).tolist())
        truth.extend(labels.tolist())
    report = classification_report(truth, predictions, labels=list(range(4)),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    macro = report["macro avg"]
    metrics = dict(accuracy=report["accuracy"], precision_macro=macro["precision"],
                   recall_macro=macro["recall"], f1_macro=macro["f1-score"],
                   per_class={label:report[label] for label in CLASSES}, sample_count=len(truth))
    n, p, z = len(truth), report["accuracy"], 1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    metrics["accuracy_wilson_95"]=[max(0,center-half),min(1,center+half)]
    metrics["interval_caveat"]="Binomial interval assumes independent samples; subject/session independence unverified"
    return metrics, confusion_matrix(truth, predictions, labels=list(range(4))).tolist(), predictions


@torch.inference_mode()
def benchmark(model, tensor, repeats=20):
    model.eval()
    images = tensor.unsqueeze(0)
    for _ in range(3): model(images)
    measurements = []
    for _ in range(repeats):
        start = time.perf_counter(); model(images)
        measurements.append((time.perf_counter()-start)*1000)
    return dict(mean_ms=float(np.mean(measurements)), p50_ms=float(np.percentile(measurements,50)),
                p95_ms=float(np.percentile(measurements,95)), device="cpu", repeats=repeats,
                warmup=3, scope="single-image forward only; preprocessing/network/explanation excluded",
                cpu=platform.processor(), torch_threads=torch.get_num_threads(), raw_ms=measurements)


def jpeg_compression(image):
    buffer = io.BytesIO(); image.save(buffer, format="JPEG", quality=30)
    buffer.seek(0)
    with Image.open(buffer) as decoded: return decoded.convert("RGB")


STRESS = {
    "dim_brightness_0.5": lambda image: ImageEnhance.Brightness(image).enhance(0.5),
    "gaussian_blur_radius_2": lambda image: image.filter(ImageFilter.GaussianBlur(2)),
    "jpeg_quality_30": jpeg_compression,
}
