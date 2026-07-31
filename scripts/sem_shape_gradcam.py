"""
Grad-CAM visualizations for the Swin-T SEM shape classifier.
Picks a few correctly-classified and misclassified validation images per
class and saves overlay PNGs. Uses pytorch-grad-cam if available.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import swin_t

CLASSES = ["bead", "fibre", "fragment"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def reshape_transform(tensor):
    # swin_t feature maps come out as (B, H, W, C); grad-cam expects (B, C, H, W)
    result = tensor.transpose(2, 3).transpose(1, 2)
    return result


def load_model(ckpt_path, device):
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, len(CLASSES))
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-class", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)
    target_layer = model.features[-1][-1].norm2
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    rows = []
    with open(args.val_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["path"], r["label"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    correct = {c: [] for c in CLASSES}
    wrong = {c: [] for c in CLASSES}

    for path, label in rows:
        img = Image.open(path).convert("RGB")
        x = eval_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            pred_idx = int(logits.argmax(dim=1).item())
        true_idx = CLASS_TO_IDX[label]
        pred_name = CLASSES[pred_idx]
        if pred_idx == true_idx and len(correct[label]) < args.per_class:
            correct[label].append((path, pred_name))
        elif pred_idx != true_idx and len(wrong[label]) < args.per_class:
            wrong[label].append((path, pred_name))

    saved = []
    for bucket_name, bucket in [("correct", correct), ("misclassified", wrong)]:
        for label, items in bucket.items():
            for path, pred_name in items:
                img = Image.open(path).convert("RGB").resize((224, 224))
                rgb = np.float32(img) / 255.0
                x = eval_tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                target_idx = CLASS_TO_IDX[label] if bucket_name == "correct" else CLASS_TO_IDX[pred_name]
                grayscale_cam = cam(input_tensor=x, targets=None)[0]
                overlay = show_cam_on_image(rgb, grayscale_cam, use_rgb=True)
                fname = f"{bucket_name}_true-{label}_pred-{pred_name}_{Path(path).stem}.png"
                out_path = out_dir / fname
                Image.fromarray(overlay).save(out_path)
                saved.append(str(out_path))
                print(f"saved {out_path}")

    print(f"Total Grad-CAM images saved: {len(saved)}")


if __name__ == "__main__":
    main()
