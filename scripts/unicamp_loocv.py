"""
Leave-one-out cross-validation of a Swin-T (ImageNet-pretrained) classifier on the
n=14 UNICAMP SEM-image / polymer-ID labeled set.

Honest small-n proof-of-concept, not a validated accuracy figure. See README.md
in evidence/public/unicamp_classification/ for full discussion.
"""
import os, csv, json, random
import torch, torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.metrics import f1_score

SEED = 42
random.seed(SEED); torch.manual_seed(SEED)

ROOT = os.path.expanduser("~/polylung-bridge-ai")
MANIFEST = os.path.join(ROOT, "evidence/public/unicamp_classification/manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
LR = 1e-4
BATCH = 4

with open(MANIFEST) as f:
    rows = list(csv.DictReader(f))

samples = [(r["sample_id"], r["polymer_label"], os.path.join(ROOT, r["image_path"])) for r in rows]
print("Loaded", len(samples), "samples:", Counter([s[1] for s in samples]))

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_img(path, tf):
    im = Image.open(path).convert("L")
    return tf(im)


def make_model(n_classes):
    m = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
    in_f = m.head.in_features
    m.head = nn.Linear(in_f, n_classes)
    return m.to(DEVICE)


def run_loocv(subset_samples, tag):
    classes = sorted(set(s[1] for s in subset_samples))
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n = len(subset_samples)
    fold_results = []

    for i in range(n):
        test_sid, test_label, test_path = subset_samples[i]
        train_items = [subset_samples[j] for j in range(n) if j != i]

        model = make_model(len(classes))
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(EPOCHS):
            random.shuffle(train_items)
            for b0 in range(0, len(train_items), BATCH):
                batch = train_items[b0:b0 + BATCH]
                imgs = torch.stack([load_img(p, train_tf) for _, _, p in batch]).to(DEVICE)
                labels = torch.tensor([cls_to_idx[l] for _, l, _ in batch]).to(DEVICE)
                opt.zero_grad()
                out = model(imgs)
                loss = crit(out, labels)
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            img = load_img(test_path, eval_tf).unsqueeze(0).to(DEVICE)
            logits = model(img)
            probs = torch.softmax(logits, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            pred_label = classes[pred_idx]
            confidence = float(probs[pred_idx].item())
            correct = (pred_label == test_label)

        print(f"[{tag}] fold {i+1}/{n} held out={test_sid} ({test_label}) -> pred={pred_label} conf={confidence:.3f} correct={correct}")
        fold_results.append({
            "sample_id": test_sid,
            "true_label": test_label,
            "predicted_label": pred_label,
            "confidence": round(confidence, 4),
            "correct": bool(correct),
        })
        del model
        torch.cuda.empty_cache()

    acc = sum(r["correct"] for r in fold_results) / n
    y_true = [r["true_label"] for r in fold_results]
    y_pred = [r["predicted_label"] for r in fold_results]
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)

    majority_class = Counter(s[1] for s in subset_samples).most_common(1)[0][0]
    majority_acc = sum(1 for s in subset_samples if s[1] == majority_class) / n

    return {
        "tag": tag,
        "n": n,
        "classes": classes,
        "class_counts": dict(Counter(s[1] for s in subset_samples)),
        "fold_results": fold_results,
        "loocv_accuracy": round(acc, 4),
        "loocv_macro_f1": round(macro_f1, 4),
        "majority_baseline_class": majority_class,
        "majority_baseline_accuracy": round(majority_acc, 4),
    }


# Full n=14 set
full_results = run_loocv(samples, "full_n14")

# >=2-examples-only subset (PE, PP, PS) -- excludes singleton PA, PVC
subset_2plus = [s for s in samples if s[1] in ("PE", "PP", "PS")]
subset_results = run_loocv(subset_2plus, "pe_pp_ps_n%d" % len(subset_2plus))

# Panel overlap: target 4-polymer panel PE/PP/PS/PVC
panel_samples = [s for s in samples if s[1] in ("PE", "PP", "PS", "PVC")]
out_of_panel = [s for s in samples if s[1] not in ("PE", "PP", "PS", "PVC")]

output = {
    "note": "n=14 proof-of-concept signal, not a validated accuracy figure. Swin-T ImageNet-pretrained, fine-tuned per-fold, 15 epochs, heavy augmentation, no internal validation split given n this small.",
    "dataset_n": len(samples),
    "dataset_class_counts": dict(Counter(s[1] for s in samples)),
    "panel_overlap": {
        "panel_classes": ["PE", "PP", "PS", "PVC"],
        "in_panel_n": len(panel_samples),
        "in_panel_class_counts": dict(Counter(s[1] for s in panel_samples)),
        "out_of_panel_n": len(out_of_panel),
        "out_of_panel_samples": [{"sample_id": s[0], "label": s[1]} for s in out_of_panel],
    },
    "loocv_full_n14": full_results,
    "loocv_pe_pp_ps_only": subset_results,
    "singleton_class_limitation": (
        "PA (n=1) and PVC (n=1) each have exactly one example in the full n=14 set. "
        "In true LOOCV, when a singleton class's only example is held out, the training set "
        "contains zero examples of that class -- the model cannot have learned that class's "
        "visual signature and structurally cannot predict it correctly on that fold, regardless "
        "of model quality. This is a fundamental small-n limitation, not a bug. The full n=14 "
        "LOOCV figure includes these 2 folds (which fail by construction); the PE/PP/PS-only "
        "n=12 LOOCV figure excludes them and is the more interpretable accuracy estimate."
    ),
}

out_dir = os.path.join(ROOT, "evidence/public/unicamp_classification")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "loocv_results.json"), "w") as f:
    json.dump(output, f, indent=2)

print("\n=== SUMMARY ===")
print("Full n=%d LOOCV accuracy: %.3f (majority baseline %.3f)" % (
    full_results["n"], full_results["loocv_accuracy"], full_results["majority_baseline_accuracy"]))
print("PE/PP/PS n=%d LOOCV accuracy: %.3f (majority baseline %.3f)" % (
    subset_results["n"], subset_results["loocv_accuracy"], subset_results["majority_baseline_accuracy"]))
print("Panel overlap: in-panel n=%d %s | out-of-panel n=%d %s" % (
    len(panel_samples), dict(Counter(s[1] for s in panel_samples)),
    len(out_of_panel), [s[0] for s in out_of_panel]))
print("\nWrote:", os.path.join(out_dir, "loocv_results.json"))
