"""Temporary synthetic-batch CUDA compatibility check; exports no trained model."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from ml.models import create_model
from ml.finetune import seed_everything, optimizer_for_stage

seed_everything()
assert torch.cuda.is_available(), 'CUDA must be available; no silent CPU fallback'
results=[]
for model_id,size in [('resnet18_finetuned',224),('mobilenetv4_finetuned',256)]:
    model=create_model(model_id,pretrained=True).cuda().train()
    optimizer=optimizer_for_stage(model,'finetune')
    first=next(m for m in model.backbone.modules() if isinstance(m,torch.nn.Conv2d))
    before=first.weight.detach().clone()
    torch.cuda.reset_peak_memory_stats()
    inputs=torch.randn(16,3,size,size,device='cuda')
    targets=torch.arange(16,device='cuda')%4
    use_amp=model_id=='resnet18_finetuned'
    scaler=torch.amp.GradScaler('cuda',init_scale=128,enabled=use_amp)
    with torch.autocast('cuda',dtype=torch.float16,enabled=use_amp):
        loss=torch.nn.functional.cross_entropy(model(inputs),targets)
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    assert torch.isfinite(first.weight.grad).all()
    scaler.step(optimizer);scaler.update();torch.cuda.synchronize()
    assert not torch.equal(first.weight.detach(),before)
    result={'model_id':model_id,'input_size':size,'batch_size':16,'amp_enabled':use_amp,'finite_backward':True,
            'backbone_updated':True,'peak_memory_bytes':torch.cuda.max_memory_allocated()}
    results.append(result);print(json.dumps(result),flush=True)
    del model,optimizer,inputs,targets,first,before,loss,scaler
    torch.cuda.empty_cache()
payload={'scope':'Synthetic-batch compatibility with actual pretrained initialization only; no task metrics or checkpoint exported',
         'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name(0),'models':results}
(ROOT/'artifacts/verification/gpu_architecture_smoke.json').write_text(json.dumps(payload,indent=2))
