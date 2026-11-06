"""Shared training and inference hand-ROI preprocessing."""
from PIL import Image, ImageOps
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


def prepare_image(image, input_size=INPUT_SIZE):
    if not isinstance(input_size, int) or not 32 <= input_size <= 1024:
        raise ValueError("input_size must be an integer between 32 and 1024")
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        base = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
        image = Image.alpha_composite(base, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return ImageOps.pad(image, (input_size, input_size), method=Image.Resampling.LANCZOS,
                        color=(128, 128, 128))


def image_tensor(image, mean=MEAN, std=STD, input_size=INPUT_SIZE):
    if len(mean) != 3 or len(std) != 3 or any(value <= 0 for value in std):
        raise ValueError("Three RGB means and positive standard deviations required")
    return transforms.functional.normalize(
        transforms.functional.to_tensor(prepare_image(image, input_size=input_size)), mean, std)
