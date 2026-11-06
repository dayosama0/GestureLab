import zipfile
import pytest
import torch
from PIL import Image
from ml.dataset import GestureDataset, extract_dataset, audit_dataset, DATA_ROOT
from ml.models import create_model, CLASSES, target_layer
from ml.preprocessing import prepare_image, image_tensor


def test_letterbox_transparency_and_rgb():
    output=prepare_image(Image.new("RGBA",(100,200),(255,0,0,0)))
    assert output.size==(224,224) and output.mode=="RGB"
    assert output.getpixel((0,0))==(128,128,128)
    assert image_tensor(output).shape==(3,224,224)


def test_exif_orientation():
    image=Image.new("RGB",(100,200),"red")
    image.getexif()[274]=6
    output=prepare_image(image)
    assert output.getpixel((112,0))==(128,128,128)
    assert output.getpixel((112,112))==(255,0,0)


def test_zip_traversal_blocked(tmp_path):
    archive=tmp_path/"bad.zip"
    with zipfile.ZipFile(archive,"w") as zipped:zipped.writestr("../escape.jpg",b"bad")
    with pytest.raises(ValueError):extract_dataset(archive,tmp_path/"target")
    assert not (tmp_path/"escape.jpg").exists()


def test_clean_splits_cannot_be_augmented():
    with pytest.raises(ValueError):GestureDataset(DATA_ROOT,"test",augmentation=True)
    with pytest.raises(ValueError):GestureDataset(DATA_ROOT,"val",augmentation=True)


def test_source_dataset_audit():
    if not DATA_ROOT.exists():pytest.skip("private source dataset unavailable")
    audit=audit_dataset()
    assert audit["total"]==400
    assert audit["exact_duplicates"]==0
    assert list(audit["counts"]["train"])==CLASSES


@pytest.mark.parametrize("model_id",["simple_cnn","resnet18","mobilenetv4"])
def test_architecture_forward_restore_and_gradient(model_id):
    torch.set_num_threads(2)
    model=create_model(model_id,pretrained=False).eval()
    image=torch.randn(1,3,224,224,requires_grad=True)
    saved=[]
    hook=target_layer(model).register_forward_hook(lambda module,args,output:saved.append(output))
    logits=model(image)
    assert logits.shape==(1,4)
    gradient=torch.autograd.grad(logits[0,0],saved[0])[0]
    assert gradient.ndim==4 and torch.isfinite(gradient).all()
    assert min(gradient.shape[-2:]) > 1
    hook.remove()
    restored=create_model(model_id,pretrained=False).eval()
    restored.load_state_dict(model.state_dict(),strict=True)
    with torch.no_grad():torch.testing.assert_close(restored(image),logits)
