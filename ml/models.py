"""A1-compatible baselines and the MobileNetV4 frozen-feature classifier."""
from torch import nn
from torchvision import models

CLASSES = ["fist", "open_palm", "peace", "thumbs_up"]
MODEL_IDS = ("simple_cnn", "resnet18", "mobilenetv4")
FINETUNED_IDS = ("resnet18_finetuned", "mobilenetv4_finetuned")
TIMM_MODEL = "mobilenetv4_conv_small.e3600_r256_in1k"


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*4*4, 128),
                                        nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, images):
        return self.classifier(self.features(images))


class TransferModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone, self.head = backbone, head

    def forward(self, images):
        return self.head(self.backbone(images))


def create_model(model_id, pretrained=False):
    """Construction never downloads weights unless explicitly requested by training."""
    model_id = {"resnet18_finetuned": "resnet18", "mobilenetv4_finetuned": "mobilenetv4"}.get(model_id, model_id)
    if model_id == "simple_cnn":
        return SimpleCNN()
    if model_id == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        width = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif model_id == "mobilenetv4":
        import timm
        backbone = timm.create_model(TIMM_MODEL, pretrained=pretrained, num_classes=0)
        width = backbone.conv_head.out_channels
    else:
        raise ValueError(f"Unknown model: {model_id}")
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    return TransferModel(backbone, nn.Sequential(nn.Dropout(0.2), nn.Linear(width, 4)))


def target_layer(model):
    """Spatial pre-pooling features, with input gradients for frozen models.

    timm MobileNet's conv_head follows global pooling and is 1x1; using that
    convolution would destroy localization. The final blocks stage is spatial.
    """
    if isinstance(model, TransferModel) and hasattr(model.backbone, "blocks"):
        return model.backbone.blocks[-1]
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    if not convolutions:
        raise ValueError("No convolutional Grad-CAM target")
    return convolutions[-1]
