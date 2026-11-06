"""Private dataset extraction, fixed-split audit, and notebook moderate variants."""
import hashlib
import json
import random
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset
from .models import CLASSES
from .preprocessing import image_tensor

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "ready_dataset"


def extract_dataset(archive=ROOT / "data" / "ready_dataset.zip", destination=ROOT / "data"):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination) or "\\" in info.filename:
                raise ValueError("Unsafe archive path")
            if info.file_size > 12_000_000 or info.external_attr >> 16 & 0o170000 == 0o120000:
                raise ValueError("Unsafe archive entry")
        for info in zipped.infolist():
            target = destination / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            elif target.exists():
                if target.read_bytes() != zipped.read(info):
                    raise ValueError(f"Refusing to overwrite changed file {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zipped.read(info))
    return destination / "ready_dataset"


def audit_dataset(root=DATA_ROOT):
    root = Path(root)
    counts, rows, seen_hash, seen_id = {}, [], {}, {}
    for split, expected in (("train", 70), ("val", 15), ("test", 15)):
        if sorted(p.name for p in (root/split).iterdir() if p.is_dir()) != CLASSES:
            raise ValueError(f"Incorrect class directories in {split}")
        counts[split] = {}
        for label in CLASSES:
            photos = sorted((root/split/label).glob("*.jpg"))
            counts[split][label] = len(photos)
            if len(photos) != expected:
                raise ValueError(f"Unexpected count: {split}/{label}: {len(photos)}")
            for photo in photos:
                digest = hashlib.sha256(photo.read_bytes()).hexdigest()
                source_id = re.sub(r"__aug\d+$", "", photo.stem)
                if "__aug" in photo.stem or digest in seen_hash or source_id in seen_id:
                    raise ValueError(f"Duplicate or augmented source image: {photo}")
                seen_hash[digest] = seen_id[source_id] = split
                with Image.open(photo) as im:
                    im.load()
                    if im.size != (224, 224) or im.mode != "RGB":
                        raise ValueError(f"Unexpected image format {photo}")
                    small = np.array(im.convert("L").resize((9, 8)))
                    bits = small[:, 1:] > small[:, :-1]
                    dhash = sum(int(v) << i for i, v in enumerate(bits.flat))
                rows.append(dict(path=photo.relative_to(root).as_posix(), sha256=digest,
                                 split=split, label=label, dhash=dhash))
    near = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a["split"] != b["split"]:
                distance = (a["dhash"] ^ b["dhash"]).bit_count()
                if distance <= 3:
                    near.append(dict(a=a["path"], b=b["path"], dhash_distance=distance))
    return dict(counts=counts, total=len(rows), exact_duplicates=0, repeated_source_ids=0,
                near_duplicate_method="64-bit difference hash; cross-split Hamming distance <=3; screening only",
                cross_split_near_duplicates=near,
                subject_independence="Unverified: no participant/session manifest supplied",
                consent_status="Unverified: keep dataset private until consent confirmed",
                files=[{k:v for k,v in row.items() if k != "dhash"} for row in rows])


def moderate_variants(image, rng):
    """Notebook's four variants: mirror, rotation, color, affine shift/scale."""
    flipped = ImageOps.mirror(image)
    rotated = image.rotate(rng.choice([-8, -5, 5, 8]), resample=Image.Resampling.BICUBIC,
                           fillcolor=(128,128,128))
    colored = ImageEnhance.Brightness(image).enhance(rng.choice([0.85,1.15]))
    colored = ImageEnhance.Contrast(colored).enhance(rng.choice([0.90,1.10]))
    colored = ImageEnhance.Color(colored).enhance(rng.choice([0.90,1.10]))
    width, height = image.size
    scale = rng.choice([0.95,1.05]); dx,dy = rng.choice([-4,4]),rng.choice([-4,4])
    a = 1/scale; cx,cy = width/2,height/2
    shifted = image.transform(image.size, Image.Transform.AFFINE,
                              (a,0,cx-a*(cx+dx),0,a,cy-a*(cy+dy)),
                              resample=Image.Resampling.BICUBIC, fillcolor=(128,128,128))
    return [flipped, rotated, colored, shifted]


class GestureDataset(Dataset):
    def __init__(self, root, split, augmentation=False, seed=42, stress=None):
        if augmentation and split != "train":
            raise ValueError("Validation and test augmentation prohibited")
        self.samples, self.images, self.targets = [], [], []
        rng = random.Random(seed)
        for index, label in enumerate(CLASSES):
            for path in sorted((Path(root)/split/label).glob("*.jpg")):
                with Image.open(path) as im:
                    image = im.convert("RGB")
                variants = [image] + (moderate_variants(image, rng) if augmentation else [])
                for variant, item in enumerate(variants):
                    if stress:
                        item = stress(item)
                    self.samples.append((str(path), index, variant))
                    self.images.append(image_tensor(item))
                    self.targets.append(index)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


if __name__ == "__main__":
    root = extract_dataset()
    audit = audit_dataset(root)
    destination = ROOT/"artifacts"/"data_audit.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k:v for k,v in audit.items() if k != "files"}, indent=2))
