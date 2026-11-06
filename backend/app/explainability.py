"""Grad-CAM from actual selected-class gradients; no decorative fallback."""
import base64
import io
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from backend.app.schemas import Explanation


def grad_cam(model, inputs: torch.Tensor, image: Image.Image):
    from ml.models import target_layer
    activations = []

    def capture(_module, _args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("Selected model has no supported spatial activation")
        output.retain_grad()
        activations.append(output)

    handle = target_layer(model).register_forward_hook(capture)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(inputs.detach().clone().requires_grad_(True))
            selected = int(logits.argmax(dim=1).item())
            logits[0, selected].backward()
            if not activations or activations[-1].grad is None:
                raise ValueError("Grad-CAM did not receive spatial feature gradients")
            activation = activations[-1]
            weights = activation.grad.mean(dim=(2, 3), keepdim=True)
            heat = torch.relu((weights * activation).sum(dim=1, keepdim=True))
            heat = F.interpolate(heat, size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0]
            heat = heat.detach().cpu().numpy()
            span = float(heat.max() - heat.min())
            if not np.isfinite(heat).all() or span <= 1e-12:
                raise ValueError("Selected-class Grad-CAM is constant; no informative explanation is available for this input")
            heat = (heat - heat.min()) / span
            # Fixed continuous colormap is applied only to real, normalized CAM.
            color = np.stack((heat, np.maximum(0, 1 - 2 * np.abs(heat - .5)), 1 - heat), axis=-1)
            overlay = (.55 * np.asarray(image.convert("RGB"), dtype=np.float32) + .45 * color * 255).clip(0, 255).astype(np.uint8)
            buffer = io.BytesIO()
            Image.fromarray(overlay).save(buffer, format="PNG")
            prepared_buffer = io.BytesIO()
            Image.fromarray(np.asarray(image.convert("RGB"))).save(prepared_buffer, format="PNG")
            return logits.detach(), Explanation(image_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                input_image_base64=base64.b64encode(prepared_buffer.getvalue()).decode("ascii"),
                width=image.width, height=image.height)
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
