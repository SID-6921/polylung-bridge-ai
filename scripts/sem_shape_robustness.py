"""
Robustness check: evaluate the trained Swin-T SEM shape classifier on the
test set under simple perturbations vs the clean baseline.
"""
import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import transforms
from torchvision.models import swin_t

CLASSES = ["bead", "fibre", "fragment"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

NORM = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

PERTURBATIONS = {
    "clean": transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM]),
    "horizontal_flip": transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=1.0), transforms.ToTensor(), NORM]),
    "brightness_jitter": transforms.Compose([
        transforms.Resize((224, 224)), transforms.ColorJitter(brightness=0.5), transforms.ToTensor(), NORM]),
    "rotation_15": transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomRotation((15, 15)), transforms.ToTensor(), NORM]),
    "rotation_neg15": transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomRotation((-15, -15)), transforms.ToTensor(), NORM]),
    "contrast_jitter": transforms.Compose([
        transforms.Resize((224, 224)), transforms.ColorJitter(contrast=0.5), transforms.ToTensor(), NORM]),
}


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
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device)

    rows = []
    with open(args.test_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append((r["path"], CLASS_TO_IDX[r["label"]]))

    results = {}
    for name, tf in PERTURBATIONS.items():
        preds, labels = [], []
        with torch.no_grad():
            for path, label in rows:
                img = Image.open(path).convert("RGB")
                x = tf(img).unsqueeze(0).to(device)
                logits = model(x)
                pred = int(logits.argmax(dim=1).item())
                preds.append(pred)
                labels.append(label)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        results[name] = {"accuracy": round(float(acc), 4), "macro_f1": round(float(macro_f1), 4)}
        print(f"{name}: acc={acc:.4f} macro_f1={macro_f1:.4f}")

    clean_acc = results["clean"]["accuracy"]
    for name in results:
        results[name]["accuracy_delta_vs_clean"] = round(results[name]["accuracy"] - clean_acc, 4)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"model": "swin_t", "n_test_images": len(rows), "results": results}, indent=2), encoding="utf-8")
    print(f"Wrote robustness results to {out_path}")


if __name__ == "__main__":
    main()
