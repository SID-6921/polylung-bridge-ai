"""
Improved Swin-T training: small LR sweep with early stopping on val macro-F1,
same train/val/test split and data as scripts/sem_shape_train.py. Reuses all
data loading / model / eval code from sem_shape_train.py.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sem_shape_train import (
    CLASSES,
    CsvImageDataset,
    build_model,
    build_transforms,
    compute_class_weights,
    evaluate,
    plot_roc,
    train_one_epoch,
)

DATA_DIR = Path("evidence/public/sem_shape")
OUT_DIR = Path("evidence/public/sem_shape_improved")
MAX_EPOCHS = 30
PATIENCE = 5
LRS = [3e-5, 1e-4, 3e-4]
BATCH_SIZE = 16
NUM_WORKERS = 2


def run_config(lr, device, train_ds, val_ds, test_ds):
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = build_model("swin_t", len(CLASSES), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    class_weights = compute_class_weights(train_ds, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    epochs_since_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["macro_f1"]
        history.append({
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_f1,
        })
        print(f"[lr={lr}] epoch {epoch} train_loss={train_loss:.4f} "
              f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_f1:.4f}", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= PATIENCE:
                print(f"[lr={lr}] early stopping at epoch {epoch} (best epoch {best_epoch}, "
                      f"best val_macro_f1={best_val_f1:.4f})", flush=True)
                break

    model.load_state_dict(best_state)
    return {
        "lr": lr,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "best_val_macro_f1": best_val_f1,
        "history": history,
        "model": model,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_tf, eval_tf = build_transforms()
    train_ds = CsvImageDataset(DATA_DIR / "train.csv", train_tf)
    val_ds = CsvImageDataset(DATA_DIR / "val.csv", eval_tf)
    test_ds = CsvImageDataset(DATA_DIR / "test.csv", eval_tf)

    results = []
    best_overall = None
    for lr in LRS:
        r = run_config(lr, device, train_ds, val_ds, test_ds)
        results.append(r)
        if best_overall is None or r["best_val_macro_f1"] > best_overall["best_val_macro_f1"]:
            best_overall = r

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    final_val = evaluate(best_overall["model"], val_loader, device)
    final_test = evaluate(best_overall["model"], test_loader, device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    roc_path = OUT_DIR / "roc_curve_swin_t_improved.png"
    plot_roc(final_test["roc_data"], "swin_t_improved", roc_path)

    model_out = OUT_DIR / "sem_shape_swin_t_improved_model.pt"
    torch.save(
        {"model_state_dict": best_overall["model"].state_dict(), "classes": CLASSES, "model_name": "swin_t"},
        model_out,
    )
    print(f"Saved best checkpoint (lr={best_overall['lr']}, epoch={best_overall['best_epoch']}) to {model_out}")

    log = {
        "task": "Improved Swin-T SEM shape classifier: small LR sweep + early stopping on val macro-F1",
        "device": device,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "lrs_tried": LRS,
        "sweep_results": [
            {
                "lr": r["lr"],
                "best_epoch": r["best_epoch"],
                "epochs_run": r["epochs_run"],
                "best_val_macro_f1": r["best_val_macro_f1"],
                "history": r["history"],
            }
            for r in results
        ],
        "selected_lr": best_overall["lr"],
        "selected_best_epoch": best_overall["best_epoch"],
        "selection_criterion": "best val macro-F1 across sweep (NOT external test data)",
        "final_val_metrics": {k: v for k, v in final_val.items() if k not in ("roc_data", "preds", "labels", "paths", "probs")},
        "final_test_metrics": {k: v for k, v in final_test.items() if k not in ("roc_data", "preds", "labels", "paths", "probs")},
        "checkpoint": str(model_out),
        "roc_curve_png": str(roc_path),
    }
    (OUT_DIR / "training_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"Wrote training log to {OUT_DIR / 'training_log.json'}")


if __name__ == "__main__":
    main()
