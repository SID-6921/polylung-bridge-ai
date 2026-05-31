import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import Swin_T_Weights, swin_t


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def to_binary_targets(targets: torch.Tensor, class_names: List[str]) -> torch.Tensor:
    labels = [class_names[idx] for idx in targets.tolist()]
    # Normal stays 0; all pathology classes map to 1.
    binary = [0 if name.lower() == "normal" else 1 for name in labels]
    return torch.tensor(binary, dtype=torch.long, device=targets.device)


def train_one_epoch(model, loader, optimizer, criterion, class_names, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        binary_targets = to_binary_targets(targets, class_names)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, binary_targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, class_names, device) -> Dict:
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            binary_targets = to_binary_targets(targets, class_names)

            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().tolist())
            labels.extend(binary_targets.cpu().tolist())

    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=["normal", "pathological"], output_dict=True)
    return {
        "binary_accuracy": round(float(acc), 4),
        "classification_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PD-1: Fine-tune Swin-T on LungHist700 binary objective")
    parser.add_argument("--data-dir", required=True, help="Dataset root with train/val folders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="reports/pd1/pd1_metrics.json")
    parser.add_argument("--published-baseline", type=float, default=0.0)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        raise FileNotFoundError("Expected train/ and val/ under --data-dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_dataloaders(data_dir, args.batch_size, args.num_workers)
    class_names = train_loader.dataset.classes

    weights = None
    if args.pretrained:
        try:
            weights = Swin_T_Weights.IMAGENET1K_V1
        except Exception:
            weights = None

    model = swin_t(weights=weights)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, 2)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, class_names, device)
        eval_metrics = evaluate(model, val_loader, class_names, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 4),
                "val_binary_accuracy": eval_metrics["binary_accuracy"],
            }
        )

    final = evaluate(model, val_loader, class_names, device)
    baseline_delta = round(final["binary_accuracy"] - args.published_baseline, 4)
    output = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "binary_accuracy": final["binary_accuracy"],
        "published_baseline": args.published_baseline,
        "baseline_delta": baseline_delta,
        "history": history,
        "classification_report": final["classification_report"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote PD-1 metrics to {out_path}")


if __name__ == "__main__":
    main()
