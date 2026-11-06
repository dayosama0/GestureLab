"""Run actual fixed-split baselines and a matched MobileNetV4 augmentation ablation.

Example: python -m ml.train --epochs 15 --models all
Pretrained downloads occur explicitly here, never in serving code.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import ROOT, DATA_ROOT, GestureDataset, extract_dataset, audit_dataset
from .evaluate import evaluate, benchmark, STRESS
from .models import create_model, CLASSES, TIMM_MODEL
from .preprocessing import MEAN, STD


def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)


def loader(dataset, shuffle=False, batch_size=32):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(42))


@torch.inference_mode()
def extract_features(backbone, dataset, batch_size=32):
    backbone.eval()
    features, labels = [], []
    batches = loader(dataset, batch_size=batch_size)
    for index, (images, targets) in enumerate(batches):
        features.append(backbone(images).clone()); labels.append(targets.clone())
        if index % 10 == 0: print(f"Feature batch {index+1}/{len(batches)}", flush=True)
    # Clones outside inference_mode are made by caller for autograd-safe head learning.
    return torch.cat(features), torch.cat(labels)


def epoch_pass(model, batches, optimizer=None):
    model.train(optimizer is not None)
    loss_sum = correct = count = 0
    for images, targets in batches:
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        loss_sum += loss.item()*len(targets)
        correct += (logits.argmax(1)==targets).sum().item(); count += len(targets)
    return loss_sum/count, correct/count


def fit(model, train, val, epochs, batch_size, patience):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history, best, stale, best_epoch, state = [], float("inf"), 0, 0, None
    train_loader, val_loader = loader(train, True, batch_size), loader(val, False, batch_size)
    for epoch in range(1, epochs+1):
        train_loss, train_accuracy = epoch_pass(model, train_loader, optimizer)
        val_loss, val_accuracy = epoch_pass(model, val_loader)
        history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                            train_accuracy=train_accuracy, val_accuracy=val_accuracy))
        print(f"Epoch {epoch}/{epochs}: train={train_loss:.4f}/{train_accuracy:.3f} val={val_loss:.4f}/{val_accuracy:.3f}", flush=True)
        if val_loss < best-1e-6:
            best, stale, best_epoch, state = val_loss, 0, epoch, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience: break
    model.load_state_dict(state)
    return history, best_epoch, best


def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def save_figures(run_id, history, matrix):
    plots = ROOT/"artifacts"/"plots"; plots.mkdir(parents=True,exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(10,3.7))
    for ax,metric in zip(axes,["loss","accuracy"]):
        for split in ["train","val"]:
            ax.plot([x["epoch"] for x in history],[x[f"{split}_{metric}"] for x in history],label=split)
        ax.set(xlabel="Epoch",ylabel=metric,title=run_id); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(plots/f"curves_{run_id}.png",dpi=150); plt.close(fig)
    fig,ax = plt.subplots(figsize=(5.8,4.8)); heat=ax.imshow(matrix,cmap="Blues")
    for row in range(4):
        for column in range(4): ax.text(column,row,str(matrix[row][column]),ha="center",va="center")
    ax.set(xticks=range(4),yticks=range(4),xticklabels=CLASSES,yticklabels=CLASSES,
           xlabel="Predicted",ylabel="Actual",title=f"{run_id}: clean test (n=60)")
    plt.setp(ax.get_xticklabels(),rotation=30,ha="right")
    fig.colorbar(heat,ax=ax);fig.tight_layout();fig.savefig(plots/f"confusion_{run_id}.png",dpi=150);plt.close(fig)


def run(run_id, epochs=15, batch_size=32, patience=5):
    checkpoint = ROOT/"artifacts"/"checkpoints"/f"{run_id}.pt"
    if checkpoint.exists():
        raise FileExistsError(f"Refusing to replace existing checkpoint {checkpoint}; archive the run before retraining")
    seed_everything()
    model_id = "mobilenetv4" if run_id.startswith("mobilenetv4") else run_id
    augmentation = run_id != "mobilenetv4_noaug"
    pretrained = model_id != "simple_cnn"
    print(f"START {run_id}: pretrained={pretrained}, augmentation={augmentation}",flush=True)
    started = time.perf_counter()
    model = create_model(model_id, pretrained=pretrained)
    pretrained_config = getattr(getattr(model, "backbone", None), "pretrained_cfg", {})
    train = GestureDataset(DATA_ROOT,"train",augmentation=augmentation)
    val = GestureDataset(DATA_ROOT,"val")
    if pretrained:
        train_x, train_y = extract_features(model.backbone,train,batch_size)
        val_x, val_y = extract_features(model.backbone,val,batch_size)
        train_features = TensorDataset(train_x.clone(), train_y.clone())
        val_features = TensorDataset(val_x.clone(), val_y.clone())
        del train, val; gc.collect()
        history, best_epoch, best_loss = fit(model.head,train_features,val_features,epochs,batch_size,patience)
        del train_features,val_features,train_x,train_y,val_x,val_y
    else:
        history, best_epoch, best_loss = fit(model,train,val,epochs,batch_size,patience)
        del train,val
    training_seconds = time.perf_counter()-started
    source = "torchvision ResNet18_Weights.IMAGENET1K_V1" if model_id=="resnet18" else (
        f"timm/{TIMM_MODEL}" if model_id=="mobilenetv4" else "random initialization; trained from scratch")
    metadata = dict(model_id=model_id, classes=CLASSES, trained=True, input_size=224,
                    mean=list(MEAN), std=list(STD), seed=42,
                    augmentation="moderate_4_fixed_variants_plus_original" if augmentation else "none",
                    pretrained_source=source, backbone_frozen=pretrained, best_epoch=best_epoch,
                    pretrained_weights_loaded=pretrained,
                    pretrained_config={key:pretrained_config.get(key) for key in
                                       ["architecture","hf_hub_id","url","input_size","mean","std"]
                                       if key in pretrained_config},
                    source_dataset_license="Not supplied; do not redistribute private source photos",
                    best_val_loss=best_loss, epoch_budget=epochs, patience=patience,
                    actual_epochs=len(history), batch_size=batch_size, learning_rate=1e-3,
                    weight_decay=1e-4, checkpoint_selection="minimum validation cross-entropy",
                    trained_at=datetime.now(timezone.utc).isoformat(), training_seconds=training_seconds,
                    normalization_policy="ImageNet; 224 RGB gray-128 letterbox, user-selected hand ROI",
                    pretrained_native_input_size=256 if model_id=="mobilenetv4" else 224,
                    adaptation="MobileNetV4 native 256 weights evaluated at shared 224 ROI" if model_id=="mobilenetv4" else "A1 architecture reproduction",
                    dataset_audit_sha256=hashlib.sha256((ROOT/"artifacts"/"data_audit.json").read_bytes()).hexdigest(),
                    environment=dict(python=platform.python_version(),torch=str(torch.__version__),
                                     platform=platform.platform(),processor=platform.processor(),device="cpu",threads=2))
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(state_dict=model.state_dict(),metadata=metadata),checkpoint)
    # Evaluate a strict reload, not the in-memory training object.
    restored = create_model(model_id,pretrained=False)
    saved = torch.load(checkpoint,map_location="cpu",weights_only=True)
    restored.load_state_dict(saved["state_dict"],strict=True);restored.eval()
    del model;gc.collect()
    test = GestureDataset(DATA_ROOT,"test")
    metrics,matrix,predictions=evaluate(restored,test,batch_size)
    latency=benchmark(restored,test[0][0])
    stress={}
    for name,transform in STRESS.items():
        stress[name]=evaluate(restored,GestureDataset(DATA_ROOT,"test",stress=transform),batch_size)[0]
    result=dict(run_id=run_id,model_id=model_id,metadata=metadata,history=history,metrics=metrics,
                confusion_matrix=matrix,latency=latency,stress=stress,
                checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                test_protocol="Unaugmented source test, evaluated after validation selection; stress separate",
                uncertainty="60 test images, 15/class; subject/session independence unverified")
    output=ROOT/"artifacts"/"metrics";output.mkdir(parents=True,exist_ok=True)
    (output/f"{run_id}.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    write_csv(output/f"train_history_{run_id}.csv",history)
    np.savetxt(output/f"confusion_{run_id}.csv",matrix,delimiter=",",fmt="%d")
    (output/f"classification_report_{run_id}.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    write_csv(output/f"predictions_{run_id}.csv",[dict(image=Path(s[0]).relative_to(DATA_ROOT).as_posix(),
              actual=CLASSES[s[1]],predicted=CLASSES[p]) for s,p in zip(test.samples,predictions)])
    save_figures(run_id,history,matrix)
    print(f"DONE {run_id}: accuracy={metrics['accuracy']:.4f} macroF1={metrics['f1_macro']:.4f} {training_seconds:.1f}s",flush=True)
    del restored,test;gc.collect()
    return result


def export_comparison():
    output=ROOT/"artifacts"/"metrics"
    results=[]
    for run_id in ["simple_cnn","resnet18","mobilenetv4_noaug","mobilenetv4"]:
        path=output/f"{run_id}.json"
        if path.exists(): results.append(json.loads(path.read_text(encoding="utf-8")))
    rows=[dict(run_id=r["run_id"],accuracy=r["metrics"]["accuracy"],precision_macro=r["metrics"]["precision_macro"],
          recall_macro=r["metrics"]["recall_macro"],f1_macro=r["metrics"]["f1_macro"],
          latency_mean_ms=r["latency"]["mean_ms"],best_epoch=r["metadata"]["best_epoch"]) for r in results]
    write_csv(output/"comparison.csv",rows)
    write_csv(output/"ablation.csv",[r for r in rows if r["run_id"].startswith("mobilenetv4")])
    (output/"comparison.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (ROOT/"artifacts"/"experiment_manifest.json").write_text(json.dumps(dict(runs=[dict(run_id=r["run_id"],
        checkpoint_sha256=r["checkpoint_sha256"],metadata=r["metadata"]) for r in results]),indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",nargs="+",default=["all"],choices=["all","simple_cnn","resnet18","mobilenetv4","mobilenetv4_noaug"])
    parser.add_argument("--epochs",type=int,default=15)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--patience",type=int,default=5)
    args=parser.parse_args()
    if min(args.epochs,args.batch_size,args.patience)<1:parser.error("Training arguments must be positive")
    extract_dataset()
    audit=audit_dataset()
    (ROOT/"artifacts").mkdir(exist_ok=True)
    (ROOT/"artifacts"/"data_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    runs=["simple_cnn","resnet18","mobilenetv4_noaug","mobilenetv4"] if "all" in args.models else args.models
    for run_id in runs:
        run(run_id,args.epochs,args.batch_size,args.patience)
        export_comparison()


if __name__=="__main__": main()
