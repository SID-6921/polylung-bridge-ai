import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import swin_t


def load_model(checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", []) if isinstance(checkpoint, dict) else []
    model = swin_t(weights=None)
    model.head = torch.nn.Linear(model.head.in_features, len(class_names))
    state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model, class_names


def build_dataset(data_dir: Path):
    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return datasets.ImageFolder(str(data_dir / "val"), transform=tf)


def predict(model, loader, device):
    preds = []
    labels = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            labels.extend(targets.tolist())
    return labels, preds


def make_loader(dataset, transform_fn, batch_size=8):
    class WrappedDataset(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform
            self.classes = base.classes

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            path, label = self.base.samples[idx]
            img = Image.open(path).convert("L")
            img = self.transform(img)
            return img, label

    wrapped = WrappedDataset(dataset, transform_fn)
    return DataLoader(wrapped, batch_size=batch_size, shuffle=False, num_workers=0)


def transform_identity(img: Image.Image) -> Image.Image:
    return img


def transform_flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


def transform_brightness(img: Image.Image) -> Image.Image:
    return ImageEnhance.Brightness(img).enhance(1.25)


def transform_rotation(img: Image.Image) -> Image.Image:
    return img.rotate(10, resample=Image.Resampling.BICUBIC, expand=False)


def transform_contrast(img: Image.Image) -> Image.Image:
    return ImageEnhance.Contrast(img).enhance(1.25)


def evaluate_variant(model, dataset, device, transform_fn) -> Dict:
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class VariantDataset(torch.utils.data.Dataset):
        def __init__(self, base, transform):
            self.base = base
            self.transform = transform
            self.classes = base.classes

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            path, label = self.base.samples[idx]
            img = Image.open(path).convert("L")
            img = self.transform(img)
            img = preprocess(img)
            return img, label

    loader = DataLoader(VariantDataset(dataset, transform_fn), batch_size=8, shuffle=False, num_workers=0)
    labels, preds = predict(model, loader, device)
    return {
        "accuracy": round(float(accuracy_score(labels, preds)), 4),
        "macro_f1": round(float(f1_score(labels, preds, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(labels, preds, average="weighted", zero_division=0)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess simple robustness of the Microplastics_SEM classifier")
    parser.add_argument("--data-dir", default="data/microplastics_sem_split")
    parser.add_argument("--checkpoint", default="evidence/public/pd1/microplastics_sem_swin_t.pt")
    parser.add_argument("--output", default="evidence/public/pd1/microplastics_sem_robustness.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names = load_model(checkpoint, device)
    dataset = build_dataset(data_dir)

    variants: Dict[str, Tuple] = {
        "clean": transform_identity,
        "horizontal_flip": transform_flip,
        "brightness_up": transform_brightness,
        "rotation_10deg": transform_rotation,
        "contrast_up": transform_contrast,
    }

    results = {
        "device": device,
        "class_names": class_names,
        "variants": {},
    }
    for name, transform_fn in variants.items():
        results["variants"][name] = evaluate_variant(model, dataset, device, transform_fn)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote robustness results to {out_path}")


if __name__ == "__main__":
    main()