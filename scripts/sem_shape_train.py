"""
Baseline 3-class SHAPE (fibre/bead/fragment) classifier training + eval.
Preliminary pilot evidence, NOT a validated/final classifier (mirrors the
repo's existing PD-1 framing). Structural reference: pd1_train_swin_lunghist700.py.
Supports --model swin_t | resnet50 | efficientnet_b0 so the same script can
produce the Swin-T, ResNet50 and EfficientNet-B0 baselines on the identical
train/val/test split (from sem_shape_prepare.py) for a fair comparison.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet50_Weights,
    Swin_T_Weights,
    efficientnet_b0,
    resnet50,
    swin_t,
)

CLASSES = ["bead", "fibre", "fragment"]  # fixed order for reproducibility
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class CsvImageDataset(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.rows = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append((r["path"], CLASS_TO_IDX[r["label"]]))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        path, label = self.rows[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label, path


def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_model(name: str, num_classes: int, pretrained: bool):
    if name == "swin_t":
        weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)
    elif name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b0(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"unknown model {name}")
    return model


def compute_class_weights(dataset: CsvImageDataset, device):
    counts = [0] * len(CLASSES)
    for _, label in dataset.rows:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (len(CLASSES) * c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, targets, _ in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_preds, all_labels, all_paths = [], [], [], []
    for images, targets, paths in loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(targets.tolist())
        all_paths.extend(paths)
    all_probs = np.concatenate(all_probs, axis=0)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(len(CLASSES))))
    report = classification_report(all_labels, all_preds, target_names=CLASSES, labels=list(range(len(CLASSES))), output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASSES)))).tolist()

    y_bin = label_binarize(all_labels, classes=list(range(len(CLASSES))))
    auc_per_class = {}
    roc_data = {}
    for i, c in enumerate(CLASSES):
        if y_bin[:, i].sum() == 0 or y_bin[:, i].sum() == len(y_bin):
            auc_per_class[c] = None
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
        auc = roc_auc_score(y_bin[:, i], all_probs[:, i])
        auc_per_class[c] = round(float(auc), 4)
        roc_data[c] = (fpr, tpr, auc)
    valid_aucs = [v for v in auc_per_class.values() if v is not None]
    mean_auc = round(float(np.mean(valid_aucs)), 4) if valid_aucs else None

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class_f1": {c: round(float(f), 4) for c, f in zip(CLASSES, per_class_f1)},
        "classification_report": report,
        "confusion_matrix": {"labels": CLASSES, "matrix": cm},
        "auc_per_class": auc_per_class,
        "mean_auc": mean_auc,
        "roc_data": roc_data,
        "preds": all_preds,
        "labels": all_labels,
        "paths": all_paths,
        "probs": all_probs,
    }


def plot_roc(roc_data, model_name, out_path: Path):
    plt.figure(figsize=(6, 6))
    for c, (fpr, tpr, auc) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{c} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"SEM shape classification ROC (one-vs-rest) - {model_name}")
    plt.legend(loc="lower right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with train.csv/val.csv/test.csv")
    ap.add_argument("--model", required=True, choices=["swin_t", "resnet50", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--output", required=True, help="metrics json output path")
    ap.add_argument("--roc-output", required=True, help="roc curve png output path")
    ap.add_argument("--model-output", default=None)
    ap.add_argument("--pretrained", action="store_true", default=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tf, eval_tf = build_transforms()
    train_ds = CsvImageDataset(data_dir / "train.csv", train_tf)
    val_ds = CsvImageDataset(data_dir / "val.csv", eval_tf)
    test_ds = CsvImageDataset(data_dir / "test.csv", eval_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, len(CLASSES), args.pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        })
        print(f"[{args.model}] epoch {epoch} train_loss={train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")

    final_val = evaluate(model, val_loader, device)
    final_test = evaluate(model, test_loader, device)

    roc_path = Path(args.roc_output)
    plot_roc(final_test["roc_data"], args.model, roc_path)

    output = {
        "model": args.model,
        "task": "SEM particle SHAPE classification (fibre/bead/fragment) - preliminary pilot, not a validated classifier",
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "class_balanced_loss": True,
        "classes": CLASSES,
        "history": history,
        "val_metrics": {k: v for k, v in final_val.items() if k not in ("roc_data", "preds", "labels", "paths", "probs")},
        "test_metrics": {k: v for k, v in final_test.items() if k not in ("roc_data", "preds", "labels", "paths", "probs")},
        "roc_curve_png": str(roc_path),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {out_path}")

    if args.model_output:
        model_out = Path(args.model_output)
        model_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "classes": CLASSES, "model_name": args.model}, model_out)
        print(f"Saved model checkpoint to {model_out}")


if __name__ == "__main__":
    main()
