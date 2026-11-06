"""Verify actual production checkpoints, input sensitivity, and spatial gradients."""
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from .dataset import ROOT, DATA_ROOT
from .models import MODEL_IDS, CLASSES, create_model, target_layer
from .preprocessing import image_tensor


def main():
    torch.set_num_threads(2)
    paths=[next((DATA_ROOT/"test"/label).glob("*.jpg")) for label in CLASSES[:2]]
    tensors=[]
    for path in paths:
        with Image.open(path) as image:tensors.append(image_tensor(image))
    results=[]
    for model_id in MODEL_IDS:
        checkpoint=ROOT/"artifacts"/"checkpoints"/f"{model_id}.pt"
        payload=torch.load(checkpoint,map_location="cpu",weights_only=True)
        assert payload["metadata"]["trained"] is True
        assert payload["metadata"]["classes"]==CLASSES
        model=create_model(model_id,pretrained=False).eval()
        model.load_state_dict(payload["state_dict"],strict=True)
        probabilities=[];cams=[];summaries=[]
        for tensor in tensors:
            activations=[]
            hook=target_layer(model).register_forward_hook(lambda m,a,o:activations.append(o))
            inputs=tensor.unsqueeze(0).requires_grad_(True)
            logits=model(inputs)
            target=int(logits.argmax(1).item())
            gradient=torch.autograd.grad(logits[0,target],activations[0])[0]
            cam=(gradient.mean((2,3),keepdim=True)*activations[0]).sum(1).relu().detach()
            hook.remove()
            assert min(cam.shape[-2:])>1
            assert torch.isfinite(cam).all() and float(cam.max()-cam.min())>0
            scores=logits.softmax(1).detach()[0]
            assert abs(float(scores.sum())-1)<1e-5
            probabilities.append(scores);cams.append(cam)
            summaries.append(dict(label=CLASSES[target],probabilities=scores.tolist(),
                                  cam_shape=list(cam.shape),cam_min=float(cam.min()),cam_max=float(cam.max())))
        assert not torch.allclose(probabilities[0],probabilities[1],atol=1e-7)
        assert not torch.allclose(cams[0],cams[1],atol=1e-7)
        results.append(dict(model_id=model_id,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            samples=[p.relative_to(DATA_ROOT).as_posix() for p in paths],
                            results=summaries,input_sensitive=True,real_spatial_gradients=True))
    output=ROOT/"artifacts"/"verification"/"ml_checkpoint_smoke.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(f"PASS: {len(results)} trained production models, strict reload, two distinct held-out inputs, normalized probabilities, nonconstant input-sensitive Grad-CAM; {output}")


if __name__=="__main__":main()
