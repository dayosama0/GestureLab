import json
import pytest
import torch
from torch import nn
from PIL import Image

import ml.finetune as finetune
from ml.models import CLASSES, create_model, target_layer, TransferModel
from ml.preprocessing import image_tensor, prepare_image, MEAN, STD


def fixture_manifest(tmp_path,monkeypatch):
    monkeypatch.setattr(finetune,"ROOT",tmp_path)
    monkeypatch.setattr(finetune,"DATA_ROOT",tmp_path/"data"/"ready_dataset")
    records=[]
    for split in ("train","val","test"):
        for label in CLASSES:
            relative=f"data/expanded/crops/{split}/{label}.png"
            path=tmp_path/relative;path.parent.mkdir(parents=True,exist_ok=True)
            Image.new("RGB",(64,96),"red").save(path)
            records.append(dict(path=relative,label=label,split=split,
                                source_id=f"{split}_{label}",user_id=f"{split}_user",origin="hagrid"))
    manifest=tmp_path/"manifest.json"
    manifest.write_text(json.dumps(dict(ready=True,records=records)),encoding="utf-8")
    return manifest,records


def test_manifest_subject_disjoint_and_native_dataset(tmp_path,monkeypatch):
    path,records=fixture_manifest(tmp_path,monkeypatch)
    _,groups,audit=finetune.validate_manifest(path)
    assert audit["users_disjoint"] and len(groups["train"])==4
    dataset=finetune.ManifestDataset(groups["val"],finetune.POLICIES["mobilenetv4_finetuned"])
    tensor,label=dataset[0]
    assert tensor.shape==(3,256,256) and label==0
    with pytest.raises(ValueError):finetune.ManifestDataset(groups["val"],finetune.POLICIES["resnet18_finetuned"],augment=True)


@pytest.mark.parametrize("problem",["participant","source","path","not_ready","hash","duplicate_hash","unreviewed"])
def test_manifest_rejects_leakage_and_unsafe_paths(tmp_path,monkeypatch,problem):
    path,records=fixture_manifest(tmp_path,monkeypatch)
    manifest=dict(ready=True,records=records)
    if problem=="participant":records[4]["user_id"]=records[0]["user_id"]
    elif problem=="source":records[4]["source_id"]=records[0]["source_id"]
    elif problem=="path":records[0]["path"]="../escape.png"
    elif problem=="hash":records[0]["sha256"]="0"*64
    elif problem=="duplicate_hash":
        records[0]["sha256"]=finetune.sha256(tmp_path/records[0]["path"])
        records[4]["sha256"]=finetune.sha256(tmp_path/records[4]["path"])
    elif problem=="unreviewed":manifest["metadata"]={"perceptual_overlap_review_complete":False}
    else:manifest["ready"]=False
    path.write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError):finetune.validate_manifest(path)


def test_unknown_original_participant_is_not_claimed_independent(tmp_path,monkeypatch):
    path,records=fixture_manifest(tmp_path,monkeypatch)
    original="data/ready_dataset/train/fist/original.jpg"
    image=tmp_path/original;image.parent.mkdir(parents=True,exist_ok=True)
    Image.new("RGB",(224,224),"blue").save(image)
    records.append(dict(path=original,label="fist",split="train",source_id="original",user_id=None,origin="original"))
    path.write_text(json.dumps(dict(ready=True,records=records,metadata={"all_original_participants_excluded":True})),encoding="utf-8")
    _,_,audit=finetune.validate_manifest(path)
    assert audit["known_user_sets_disjoint"]
    assert not audit["users_disjoint"] and not audit["full_subject_independence_verified"]
    assert audit["original_train_sources_without_user_id"]==["original"]


def test_native_preprocess_preserves_legacy_default():
    image=Image.new("RGB",(90,160),(255,128,0))
    torch.testing.assert_close(image_tensor(image),image_tensor(image,mean=MEAN,std=STD,input_size=224))
    tensor=image_tensor(image,mean=[.5]*3,std=[.5]*3,input_size=256)
    assert tensor.shape==(3,256,256)
    assert float(tensor[0,128,128])==1.0
    assert prepare_image(image,input_size=256).size==(256,256)
    with pytest.raises(ValueError):image_tensor(image,std=[0,1,1])


@pytest.mark.parametrize("model_id",["resnet18_finetuned","mobilenetv4_finetuned"])
def test_new_aliases_actual_native_spatial_gradients(model_id):
    torch.set_num_threads(2)
    model=create_model(model_id,pretrained=False).eval()
    size=finetune.POLICIES[model_id]["input_size"]
    saved=[]
    handle=target_layer(model).register_forward_hook(lambda m,a,o:saved.append(o))
    try:
        output=model(torch.randn(1,3,size,size,requires_grad=True))
        gradient=torch.autograd.grad(output[0,0],saved[0])[0]
        assert output.shape==(1,4) and min(gradient.shape[-2:])>=7
    finally:handle.remove()


def test_actual_backbone_optimization_and_head_warmup_bn():
    torch.set_num_threads(2)
    backbone=nn.Sequential(nn.Conv2d(3,4,3,padding=1),nn.BatchNorm2d(4),nn.ReLU(),nn.AdaptiveAvgPool2d(1),nn.Flatten())
    model=TransferModel(backbone,nn.Linear(4,4))
    batches=[(torch.randn(4,3,16,16),torch.tensor([0,1,2,3]))]
    initial=backbone[0].weight.detach().clone();bn_initial=backbone[1].running_mean.clone()
    head_optimizer=finetune.optimizer_for_stage(model,"head")
    finetune.epoch_pass(model,batches,torch.device("cpu"),"head",head_optimizer)
    torch.testing.assert_close(initial,backbone[0].weight)
    torch.testing.assert_close(bn_initial,backbone[1].running_mean)
    optimizer=finetune.optimizer_for_stage(model,"finetune")
    finetune.epoch_pass(model,batches,torch.device("cpu"),"finetune",optimizer)
    assert not torch.equal(initial,backbone[0].weight)
    assert all(parameter.requires_grad for parameter in backbone.parameters())
