import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import Swin_T_Weights, swin_t


def build_dataloaders(data_dir: Path, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(str(data_dir / "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(str(data_dir / "val"), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def compute_class_weights(dataset, class_names: List[str], device: str) -> torch.Tensor:
    counts = [0 for _ in class_names]
    for target in dataset.targets:
        counts[target] += 1

    if any(count == 0 for count in counts):
        missing = [name for name, count in zip(class_names, counts) if count == 0]
        raise ValueError(f"Each class must be present in training data: missing {missing}")

    total = sum(counts)
    weights = [total / (len(counts) * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
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

            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().tolist())
            labels.extend(targets.cpu().tolist())

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    report = classification_report(labels, preds, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names)))).tolist()
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm,
        },
        "classification_report": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Swin-T on an ImageFolder-style SEM dataset")
    parser.add_argument("--data-dir", required=True, help="Dataset root with train/val folders")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", default="evidence/public/pd1/microplastics_sem_metrics.json")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained weights")
    parser.add_argument(
        "--disable-class-balanced-loss",
        action="store_true",
        help="Disable inverse-frequency class weights in CrossEntropyLoss",
    )
    parser.add_argument("--model-output", default="evidence/public/pd1/microplastics_sem_swin_t.pt")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        raise FileNotFoundError("Expected train/ and val/ under --data-dir")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader = build_dataloaders(data_dir, args.batch_size, args.num_workers)
    class_names = train_loader.dataset.classes
    if len(class_names) < 2:
        raise ValueError("Need at least two classes for classification training")

    weights = None
    if args.pretrained:
        try:
            weights = Swin_T_Weights.IMAGENET1K_V1
        except Exception:
            weights = None

    pretrained_loaded = False
    try:
        model = swin_t(weights=weights)
        pretrained_loaded = weights is not None
    except Exception as exc:
        print(f"Pretrained weights unavailable, training from scratch: {exc}")
        model = swin_t(weights=None)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, len(class_names))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.disable_class_balanced_loss:
        class_weights = None
    else:
        class_weights = compute_class_weights(train_loader.dataset, class_names, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_metrics = evaluate(model, val_loader, class_names, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 4),
                "val_accuracy": eval_metrics["accuracy"],
                "val_macro_f1": eval_metrics["macro_f1"],
                "val_weighted_f1": eval_metrics["weighted_f1"],
            }
        )

    final = evaluate(model, val_loader, class_names, device)
    output = {
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "class_balanced_loss": not args.disable_class_balanced_loss,
        "pretrained_loaded": pretrained_loaded,
        "accuracy": final["accuracy"],
        "macro_f1": final["macro_f1"],
        "weighted_f1": final["weighted_f1"],
        "history": history,
        "confusion_matrix": final["confusion_matrix"],
        "classification_report": final["classification_report"],
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote SEM metrics to {out_path}")

    model_output = Path(args.model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    save_state_dict = {}
    for key, value in model.state_dict().items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            save_state_dict[key] = value.detach().cpu().half()
        else:
            save_state_dict[key] = value.detach().cpu() if torch.is_tensor(value) else value
    torch.save(
        {
            "model_state_dict": save_state_dict,
            "class_names": class_names,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
        },
        model_output,
    )
    print(f"Saved model checkpoint to: {model_output}")


if __name__ == "__main__":
    main()