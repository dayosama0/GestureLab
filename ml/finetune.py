"""Subject-disjoint expansion with real staged CUDA backbone fine-tuning.

Run only after the audited expansion manifest is complete:
    python -m ml.finetune --models all --head-epochs 2 --finetune-epochs 8

Legacy checkpoints and result files are never modified. Test sets are evaluated
only after the new-validation checkpoint has been selected and strictly reloaded.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .dataset import ROOT, DATA_ROOT
from .evaluate import evaluate, benchmark, STRESS
from .models import CLASSES, FINETUNED_IDS, TIMM_MODEL, create_model, target_layer
from .preprocessing import MEAN, STD, image_tensor
from .train import save_figures, write_csv

POLICIES = {
    "resnet18_finetuned": {"input_size":224,"mean":list(MEAN),"std":list(STD)},
    "mobilenetv4_finetuned": {"input_size":256,"mean":[0.5]*3,"std":[0.5]*3},
}


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""):digest.update(block)
    return digest.hexdigest()


def seed_everything(seed=42):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def validate_manifest(path):
    path=Path(path)
    manifest=json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("ready") is False:raise ValueError("Expansion manifest is not ready")
    records=manifest.get("records")
    if not isinstance(records,list) or not records:raise ValueError("Manifest records missing")
    groups={split:[] for split in ("train","val","test")}
    users={split:set() for split in groups};sources={};paths={};hashes={};missing_original_users=[]
    original_holdout_ids={image.stem for split in ("val","test") for image in (DATA_ROOT/split).rglob("*.jpg")}
    for record in records:
        if record.get("split") not in groups or record.get("label") not in CLASSES:
            raise ValueError("Invalid manifest split/class")
        relative=Path(record["path"])
        candidate=(ROOT/relative).resolve()
        if relative.is_absolute() or not candidate.is_relative_to(ROOT) or not candidate.is_file():
            raise ValueError(f"Missing or unsafe crop path: {relative}")
        source=str(record.get("source_id") or "")
        if not source:raise ValueError("Each crop needs a source_id")
        if source in original_holdout_ids:raise ValueError("Original held-out source ID entered expansion manifest")
        split=record["split"]
        if source in sources or candidate in paths:raise ValueError("Repeated source ID or crop path")
        sources[source]=split;paths[candidate]=split
        original_train=candidate.is_relative_to((DATA_ROOT/"train").resolve())
        original_holdout=any(candidate.is_relative_to((DATA_ROOT/s).resolve()) for s in ("val","test"))
        if original_holdout or (original_train and split!="train"):
            raise ValueError("Original holdouts cannot enter expanded training/selection")
        user=record.get("user_id")
        if not user and not original_train:raise ValueError("New HaGRID samples require upstream user_id")
        if user:users[split].add(str(user))
        elif original_train:missing_original_users.append(source)
        recorded_hash=record.get("sha256")
        if recorded_hash and sha256(candidate)!=recorded_hash:
            raise ValueError(f"Crop hash does not match manifest: {relative}")
        if recorded_hash:
            if recorded_hash in hashes and hashes[recorded_hash]!=split:
                raise ValueError("Identical crop bytes occur in different splits")
            hashes[recorded_hash]=split
        groups[split].append(record)
    for first,second in (("train","val"),("train","test"),("val","test")):
        if users[first]&users[second]:raise ValueError(f"Participant leakage between {first} and {second}")
    counts={split:dict(Counter(r["label"] for r in rows)) for split,rows in groups.items()}
    if any(set(counts[split])!=set(CLASSES) for split in groups):
        raise ValueError("Every split must contain all four classes")
    upstream=manifest.get("metadata",{})
    if upstream.get("perceptual_overlap_review_complete") is False:
        raise ValueError("Expansion perceptual-overlap review is incomplete")
    full_separation=not missing_original_users and upstream.get("all_original_participants_excluded") is True
    summary=dict(manifest_sha256=sha256(path),counts=counts,
                 user_counts={split:len(value) for split,value in users.items()},
                 known_user_sets_disjoint=True,users_disjoint=not missing_original_users,
                 full_subject_independence_verified=full_separation,
                 original_train_sources_without_user_id=missing_original_users,
                 all_original_participants_excluded=upstream.get("all_original_participants_excluded") is True,
                 source_ids_disjoint=True,
                 crop_hashes_verified=sum(bool(record.get("sha256")) for record in records),
                 original_holdouts_excluded_from_manifest=True,
                 upstream_metadata=upstream)
    return manifest,groups,summary


class ManifestDataset(Dataset):
    def __init__(self, records, policy, augment=False, stress=None):
        if augment and any(record["split"]!="train" for record in records):
            raise ValueError("Augmentation is restricted to training records")
        self.records=list(records);self.policy=policy;self.stress=stress
        self.augmentation=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=8,translate=(.03,.03),scale=(.95,1.05),
                                    interpolation=transforms.InterpolationMode.BILINEAR,fill=(128,128,128)),
            transforms.ColorJitter(brightness=.15,contrast=.1,saturation=.1),
        ]) if augment else None

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        record=self.records[index]
        with Image.open(ROOT/record["path"]) as source:
            image=source.convert("RGB")
        if self.augmentation:image=self.augmentation(image)
        if self.stress:image=self.stress(image)
        return image_tensor(image,**self.policy),CLASSES.index(record["label"])


def set_stage(model, stage):
    if stage not in ("head","finetune"):raise ValueError("Unknown training stage")
    for parameter in model.backbone.parameters():parameter.requires_grad_(stage=="finetune")
    for parameter in model.head.parameters():parameter.requires_grad_(True)


def optimizer_for_stage(model,stage):
    set_stage(model,stage)
    if stage=="head":return torch.optim.AdamW(model.head.parameters(),lr=1e-3,weight_decay=1e-4)
    return torch.optim.AdamW([
        {"params":model.backbone.parameters(),"lr":1e-5},
        {"params":model.head.parameters(),"lr":1e-4},
    ],weight_decay=1e-4)


def epoch_pass(model,batches,device,stage,optimizer=None,scaler=None,amp_enabled=False):
    training=optimizer is not None
    model.train(training)
    if training and stage=="head":model.backbone.eval()
    loss_sum=correct=count=0
    for images,targets in batches:
        images=images.to(device);targets=targets.to(device)
        if optimizer:optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp_enabled):
                logits=model(images);loss=nn.functional.cross_entropy(logits,targets)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite loss; no production checkpoint emitted")
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward();scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(),5.0)
                    scaler.step(optimizer);scaler.update()
                else:
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
        loss_sum+=float(loss.detach())*len(targets)
        correct+=(logits.argmax(1)==targets).sum().item();count+=len(targets)
    return loss_sum/count,correct/count


def original_test_records():
    return [dict(path=path.relative_to(ROOT).as_posix(),label=label,split="test",source_id=path.stem,
                 user_id=None,origin="original_test") for label in CLASSES
            for path in sorted((DATA_ROOT/"test"/label).glob("*.jpg"))]


def gradcam_smoke(model,dataset):
    model.eval();rows=[];scores=[]
    # Two distinct classes, retained only as IDs/statistics (no image exports).
    indices=[next(i for i,r in enumerate(dataset.records) if r["label"]==label) for label in CLASSES[:2]]
    for index in indices:
        activations=[]
        handle=target_layer(model).register_forward_hook(lambda m,a,o:activations.append(o))
        try:
            inputs=dataset[index][0].unsqueeze(0).requires_grad_(True)
            logits=model(inputs);label=int(logits.argmax(1))
            gradient=torch.autograd.grad(logits[0,label],activations[0])[0]
            cam=(gradient.mean((2,3),keepdim=True)*activations[0]).sum(1).relu().detach()
            if min(cam.shape[-2:])<2 or not torch.isfinite(cam).all() or float(cam.max()-cam.min())<=0:
                raise RuntimeError("Invalid or constant actual Grad-CAM")
            probability=logits.softmax(1).detach()[0]
            if abs(float(probability.sum())-1)>1e-5:raise RuntimeError("Invalid probabilities")
            scores.append(probability)
            rows.append(dict(source_id=dataset.records[index]["source_id"],predicted=CLASSES[label],
                             probabilities=probability.tolist(),cam_shape=list(cam.shape),
                             cam_min=float(cam.min()),cam_max=float(cam.max())))
        finally:handle.remove()
    if torch.allclose(scores[0],scores[1],atol=1e-7):raise RuntimeError("Model lacks input sensitivity")
    return dict(passed=True,real_gradients=True,input_sensitive=True,samples=rows)


def final_evaluation(model,groups,policy,run_id):
    original=ManifestDataset(original_test_records(),policy)
    expanded=ManifestDataset(groups["test"],policy)
    if len(original)!=60:raise ValueError("Expected exactly 60 original test images")
    output={};predictions={}
    for name,dataset in (("original_test",original),("expanded_test",expanded)):
        metrics,matrix,pred=evaluate(model,dataset,batch_size=16)
        metrics["interval_caveat"]=("Image-level Wilson interval assumes independent errors. Official user IDs verify between-split participant exclusion; repeated images within an expanded-set participant may remain correlated.")
        output[name]=dict(metrics=metrics,confusion_matrix=matrix,
                          protocol="Unaugmented held-out samples, after expanded-validation checkpoint selection")
        predictions[name]=[dict(source_id=record["source_id"],user_id=record.get("user_id"),
                               image=record["path"],actual=record["label"],predicted=CLASSES[label])
                           for record,label in zip(dataset.records,pred)]
    stress={name:evaluate(model,ManifestDataset(original.records,policy,stress=transform),16)[0]
            for name,transform in STRESS.items()}
    timing=benchmark(model,original[0][0]);smoke=gradcam_smoke(model,original)
    return output,predictions,stress,timing,smoke


def train_run(run_id,manifest_path,head_epochs=2,finetune_epochs=8,batch_size=16,patience=4,device_name="cuda"):
    if run_id not in FINETUNED_IDS:raise ValueError("Only new fine-tuned IDs are accepted")
    destination=ROOT/"artifacts"/"checkpoints"/f"{run_id}.pt"
    metric_path=ROOT/"artifacts"/"metrics"/f"{run_id}.json"
    if destination.exists() or metric_path.exists():raise FileExistsError("Immutable run already exists; use a new version instead of replacing it")
    manifest,groups,audit=validate_manifest(manifest_path)
    if manifest.get("ready") is not True:
        raise ValueError("Training requires an explicitly ready, audited expansion manifest")
    if not audit["full_subject_independence_verified"]:
        raise ValueError("Subject-disjoint training requires recovered original participant IDs and verified exclusion")
    if audit["crop_hashes_verified"]!=len(manifest["records"]):
        raise ValueError("Every production training-manifest crop must have a verified SHA256")
    seed_everything();device=torch.device(device_name)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable; no CPU fallback silently substituted")
    legacy={p.name:sha256(p) for p in (ROOT/"artifacts"/"checkpoints").glob("*.pt") if p.stem not in FINETUNED_IDS}
    policy=POLICIES[run_id];started=time.perf_counter()
    print(f"START {run_id} device={device} policy={policy} counts={audit['counts']}",flush=True)
    model=create_model(run_id,pretrained=True).to(device)
    pretrained_config=getattr(model.backbone,"pretrained_cfg",{})
    if run_id=="mobilenetv4_finetuned" and (
        list(pretrained_config.get("mean",[]))!=policy["mean"] or
        list(pretrained_config.get("std",[]))!=policy["std"] or
        list(pretrained_config.get("input_size",[]))!=[3,policy["input_size"],policy["input_size"]]
    ):
        raise RuntimeError("Installed MobileNetV4 pretrained configuration differs from approved native policy")
    first_conv=next(module for module in model.backbone.modules() if isinstance(module,nn.Conv2d))
    initial_weight=first_conv.weight.detach().cpu().clone()
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)
    datasets={split:ManifestDataset(rows,policy,augment=split=="train") for split,rows in groups.items() if split!="test"}
    loaders={split:DataLoader(dataset,batch_size=batch_size,shuffle=split=="train",num_workers=0,
                pin_memory=device.type=="cuda",generator=torch.Generator().manual_seed(42)) for split,dataset in datasets.items()}
    amp_enabled=device.type=="cuda" and run_id=="resnet18_finetuned"
    scaler=torch.amp.GradScaler("cuda",enabled=amp_enabled)
    stage_dir=ROOT/"artifacts"/"training";stage_dir.mkdir(parents=True,exist_ok=True)
    best_path=stage_dir/f"{run_id}_selected.tmp.pt"
    if best_path.exists():raise FileExistsError("Staging checkpoint exists; archive failed-run evidence first")
    history=[];best=float("inf");best_epoch=0;stale=0
    for stage,epochs in (("head",head_epochs),("finetune",finetune_epochs)):
        optimizer=optimizer_for_stage(model,stage)
        for stage_epoch in range(1,epochs+1):
            epoch=len(history)+1;epoch_start=time.perf_counter()
            train_loss,train_accuracy=epoch_pass(model,loaders["train"],device,stage,optimizer,scaler,amp_enabled=amp_enabled)
            val_loss,val_accuracy=epoch_pass(model,loaders["val"],device,stage,amp_enabled=amp_enabled)
            row=dict(epoch=epoch,stage=stage,stage_epoch=stage_epoch,train_loss=train_loss,val_loss=val_loss,
                     train_accuracy=train_accuracy,val_accuracy=val_accuracy,seconds=time.perf_counter()-epoch_start)
            history.append(row)
            print(f"{run_id} {stage} epoch{stage_epoch}/{epochs}: train={train_loss:.4f}/{train_accuracy:.4f} val={val_loss:.4f}/{val_accuracy:.4f} seconds={row['seconds']:.1f}",flush=True)
            write_csv(stage_dir/f"{run_id}_history.csv",history)
            # Only a checkpoint after actual backbone optimization is eligible.
            if stage=="finetune":
                if val_loss<best-1e-6:
                    best=val_loss;best_epoch=epoch;stale=0
                    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                    best_epoch=best_epoch,best_val_loss=best),best_path)
                else:stale+=1
                if stale>=patience:break
    if best_epoch<=head_epochs:raise RuntimeError("No selected backbone-fine-tuned checkpoint")
    selected=torch.load(best_path,map_location="cpu",weights_only=True)
    peak_memory=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None
    gpu_name=torch.cuda.get_device_name(device) if device.type=="cuda" else None
    del model,loaders,datasets,optimizer
    if device.type=="cuda":torch.cuda.empty_cache()
    restored=create_model(run_id,pretrained=False)
    restored.load_state_dict(selected["state_dict"],strict=True);restored.eval()
    selected_conv=next(module for module in restored.backbone.modules() if isinstance(module,nn.Conv2d))
    delta=float(torch.linalg.vector_norm(selected_conv.weight.detach()-initial_weight))
    if not delta>0:raise RuntimeError("Selected backbone weights did not change")
    selected_validation=evaluate(restored,ManifestDataset(groups["val"],policy),16)[0]
    selected_validation["interval_caveat"]=("Official participant IDs are excluded across splits; image-level interval may understate within-user correlation.")
    source="torchvision ResNet18_Weights.IMAGENET1K_V1" if run_id.startswith("resnet") else f"timm/{TIMM_MODEL}"
    metadata=dict(model_id=run_id,classes=CLASSES,trained=True,**policy,seed=42,
        pretrained_source=source,pretrained_weights_loaded=True,initialization="fresh original ImageNet pretrained backbone; new four-class head",
        pretrained_config={key:pretrained_config[key] for key in ("architecture","hf_hub_id","input_size","mean","std","license") if key in pretrained_config},
        backbone_frozen=False,training_method="2-stage head warmup then full-backbone gradient fine-tuning",
        finetuned_backbone_parameters=sum(parameter.numel() for parameter in restored.backbone.parameters()),
        augmentation="train_only_on_the_fly_flip_affine_color_jitter; no synthetic media",
        best_epoch=best_epoch,best_val_loss=best,actual_epochs=len(history),head_epochs=head_epochs,
        validation_metrics=selected_validation,
        finetune_epoch_budget=finetune_epochs,patience=patience,batch_size=batch_size,
        head_warmup_lr=1e-3,finetune_backbone_lr=1e-5,finetune_head_lr=1e-4,weight_decay=1e-4,
        checkpoint_selection="minimum new HaGRID validation cross-entropy; known-user-disjoint splits; only post-backbone-optimization checkpoints eligible",
        trained_at=datetime.now(timezone.utc).isoformat(),training_seconds=time.perf_counter()-started,
        train_count=len(groups["train"]),validation_count=len(groups["val"]),expanded_test_count=len(groups["test"]),
        manifest_path=Path(manifest_path).resolve().relative_to(ROOT).as_posix(),data_audit=audit,
        first_backbone_conv_weight_l2_change=delta,legacy_checkpoint_sha256=legacy,
        environment=dict(python=platform.python_version(),torch=str(torch.__version__),device=str(device),
                         gpu=gpu_name,cuda=torch.version.cuda,amp=amp_enabled,precision_policy="ResNet CUDA AMP; MobileNet CUDA float32 after observed nonfinite fp16 gradients on GTX1660Ti",peak_cuda_memory_bytes=peak_memory,
                         cpu=platform.processor(),cpu_threads=2,platform=platform.platform()),
        source_dataset_license="See expansion audit/upstream provenance; original consent remains unverified",
        normalization_policy="RGB gray128 letterbox; per-model metadata normalization; identical train/serving")
    # CPU strict-reload evaluation matches CPU serving; final metrics are never used for model selection.
    output,predictions,stress,timing,smoke=final_evaluation(restored,groups,policy,run_id)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("xb") as handle:torch.save(dict(state_dict=selected["state_dict"],metadata=metadata),handle)
    final_payload=torch.load(destination,map_location="cpu",weights_only=True)
    restored.load_state_dict(final_payload["state_dict"],strict=True)
    if final_payload["metadata"]["data_audit"]["manifest_sha256"]!=audit["manifest_sha256"]:
        raise RuntimeError("Final checkpoint metadata failed strict reload verification")
    checkpoint_hash=sha256(destination)
    result=dict(run_id=run_id,model_id=run_id,metadata=metadata,history=history,
                metrics=output["original_test"]["metrics"],confusion_matrix=output["original_test"]["confusion_matrix"],
                original_test=output["original_test"],expanded_test=output["expanded_test"],
                latency=timing,stress=stress,checkpoint_sha256=checkpoint_hash,gradcam_verification=smoke,
                test_protocol="Original60 and independent expanded test evaluated once after expanded-validation selection; stress separate",
                comparison_caveat="New data and full-backbone optimization both changed; not a single-factor ablation. Original60 was examined in the earlier phase; fresh expansion test is the new independent holdout.")
    metric_path.parent.mkdir(parents=True,exist_ok=True)
    with metric_path.open("x",encoding="utf-8") as handle:json.dump(result,handle,indent=2)
    write_csv(metric_path.parent/f"train_history_{run_id}.csv",history)
    for split,rows in predictions.items():write_csv(metric_path.parent/f"predictions_{run_id}_{split}.csv",rows)
    save_figures(run_id,history,result["confusion_matrix"])
    np.savetxt(metric_path.parent/f"confusion_{run_id}.csv",result["confusion_matrix"],delimiter=",",fmt="%d")
    for name,digest in legacy.items():
        if sha256(destination.parent/name)!=digest:raise RuntimeError("Legacy checkpoint unexpectedly changed")
    verification=ROOT/"artifacts"/"verification"/f"{run_id}_smoke.json"
    verification.parent.mkdir(exist_ok=True)
    verification.write_text(json.dumps(dict(checkpoint_sha256=checkpoint_hash,**smoke,legacy_checkpoints_unchanged=True),indent=2),encoding="utf-8")
    print(f"DONE {run_id}: original accuracy={result['metrics']['accuracy']:.6f} F1={result['metrics']['f1_macro']:.6f}; expanded accuracy={output['expanded_test']['metrics']['accuracy']:.6f} F1={output['expanded_test']['metrics']['f1_macro']:.6f}; peakVRAM={peak_memory}; checkpoint={checkpoint_hash}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,default=ROOT/"data"/"expanded"/"manifest.json")
    parser.add_argument("--models",nargs="+",choices=["all",*FINETUNED_IDS],default=["all"])
    parser.add_argument("--head-epochs",type=int,default=2)
    parser.add_argument("--finetune-epochs",type=int,default=8)
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--patience",type=int,default=4)
    parser.add_argument("--device",choices=["cuda","cpu"],default="cuda")
    args=parser.parse_args()
    if min(args.head_epochs,args.finetune_epochs,args.batch_size,args.patience)<1:parser.error("Arguments must be positive")
    for run_id in (FINETUNED_IDS if "all" in args.models else args.models):
        train_run(run_id,args.manifest,args.head_epochs,args.finetune_epochs,args.batch_size,args.patience,args.device)


if __name__=="__main__":main()
