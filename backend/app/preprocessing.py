"""In-memory image decoding. Upload names and image bytes are never retained."""
import io
import warnings
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000


def decode_image(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("The uploaded file is empty")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                if probe.format not in {"JPEG", "PNG", "WEBP"}:
                    raise ValueError("Upload an actual JPEG, PNG, or WebP image")
                if probe.width * probe.height > MAX_PIXELS:
                    raise ValueError("Image exceeds the 16 megapixel decoding limit")
                if getattr(probe, "n_frames", 1) != 1:
                    raise ValueError("Animated images are not supported; upload a single frame")
                probe.verify()
            with Image.open(io.BytesIO(data)) as source:
                source.load()
                oriented = ImageOps.exif_transpose(source)
                if "A" in oriented.getbands() or "transparency" in oriented.info:
                    rgba = oriented.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (128, 128, 128, 255))
                    return Image.alpha_composite(background, rgba).convert("RGB")
                return oriented.convert("RGB").copy()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("The uploaded bytes are not a valid, safely decodable image") from exc
