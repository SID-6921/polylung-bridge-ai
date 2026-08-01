"""
External validation of the SEM shape classifier (fibre/bead/fragment) on an
independent dataset: CUNY microFTIR biosolids study camera images, with SHAPE
labels sourced from All-Data-Spreadsheet.xlsx (see
evidence/public/sem_shape_external_validation/README.md for methodology).

Inference-only. Reuses build_model/build_transforms/CLASSES from
scripts/sem_shape_train.py so preprocessing and architecture exactly match
training.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sem_shape_train import CLASSES, CLASS_TO_IDX, build_model, build_transforms  # noqa: E402


class ManifestDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append((r["image_path"], CLASS_TO_IDX[r["label"]]))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, path


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, targets, _ in loader:
        images = images.to(device)
        logits = model(images)
        preds = torch.softmax(logits, dim=1).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(targets.tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(
        all_labels, all_preds, target_names=CLASSES,
        labels=list(range(len(CLASSES))), output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES)))).tolist()
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "classification_report": report,
        "confusion_matrix": {"labels": CLASSES, "matrix": cm},
        "n_samples": len(all_labels),
    }


def plot_confusion_matrix(cm, labels, model_name, out_path: Path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"External validation confusion matrix - {model_name}\n(CUNY biosolids, out-of-distribution)")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["swin_t", "resnet50", "efficientnet_b0"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cm-output", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, eval_tf = build_transforms()
    ds = ManifestDataset(Path(args.manifest), eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, len(CLASSES), pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    metrics = evaluate(model, loader, device)
    metrics["model"] = args.model
    metrics["checkpoint"] = args.checkpoint
    metrics["device"] = device
    metrics["classes"] = CLASSES

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[{args.model}] acc={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f} n={metrics['n_samples']}")
    print(f"Wrote {out_path}")

    if args.cm_output:
        plot_confusion_matrix(metrics["confusion_matrix"]["matrix"], CLASSES, args.model, Path(args.cm_output))
        print(f"Wrote {args.cm_output}")


if __name__ == "__main__":
    main()
