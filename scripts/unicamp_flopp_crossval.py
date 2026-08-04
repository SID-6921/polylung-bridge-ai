"""
Test the EXISTING FLOPP-trained spectral classifier on UNICAMP's 14 FTIR spectra.
No new training -- pure inference, sidestepping the small-n training problem
entirely, analogous to the existing FLOPP-e weathered-spectra external validation.

n=14 proof-of-concept, not a validated accuracy figure.
"""
import glob, os, re, csv, json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from collections import Counter

ROOT = os.path.expanduser("~/polylung-bridge-ai")
FLOPP_BASE = os.path.join(ROOT, "data/microplastics_datasets/flopp_spectral_library/extracted")
MANIFEST = os.path.join(ROOT, "evidence/public/unicamp_classification/manifest.csv")
COMMON_GRID = np.linspace(400, 4000, 900)


def load_flopp_split(folder):
    X, y = [], []
    for f in glob.glob(os.path.join(FLOPP_BASE, folder, "*.CSV")):
        base = os.path.basename(f)
        m = re.match(r"^([A-Za-z]+)\s", base)
        if not m:
            continue
        label = m.group(1)
        try:
            data = np.loadtxt(f, delimiter=",")
        except Exception:
            continue
        if data.ndim != 2 or data.shape[0] < 10:
            continue
        wn, ab = data[:, 0], data[:, 1]
        order = np.argsort(wn)
        wn, ab = wn[order], ab[order]
        interp = np.interp(COMMON_GRID, wn, ab)
        X.append(interp)
        y.append(label)
    return np.array(X), np.array(y)


def load_unicamp_spectrum(path):
    wn, ab = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                wn.append(float(row[0]))
                ab.append(float(row[1]))
            except ValueError:
                continue
    wn = np.array(wn); ab = np.array(ab)
    order = np.argsort(wn)
    wn, ab = wn[order], ab[order]
    return np.interp(COMMON_GRID, wn, ab)


# --- Train FLOPP classifier on ALL of FLOPP (no held-out split needed -- we're
# testing generalization to UNICAMP, not re-measuring FLOPP's own test performance) ---
Xf, yf = load_flopp_split("FLOPP .csv")
cnt = Counter(yf)
keep = {k for k, v in cnt.items() if v >= 2}
mask = np.array([lab in keep for lab in yf])
Xf, yf = Xf[mask], yf[mask]
flopp_classes = sorted(set(yf))
print("FLOPP training set:", Xf.shape, "classes:", flopp_classes)

clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
clf.fit(Xf, yf)

# --- Load UNICAMP manifest and spectra ---
with open(MANIFEST) as f:
    rows = list(csv.DictReader(f))

results = []
for r in rows:
    sid, label = r["sample_id"], r["polymer_label"]
    csv_path = os.path.join(ROOT, r["ftir_csv_path"])
    if not os.path.exists(csv_path):
        print(f"MISSING: {csv_path}")
        continue
    spec = load_unicamp_spectrum(csv_path)
    pred = clf.predict([spec])[0]
    proba = clf.predict_proba([spec])[0]
    confidence = float(max(proba))
    in_flopp_classes = label in flopp_classes
    correct = bool(pred == label)
    print(f"{sid} true={label} (in_flopp_classes={in_flopp_classes}) pred={pred} conf={confidence:.3f} correct={correct}")
    results.append({
        "sample_id": sid, "true_label": label, "predicted_label": pred,
        "confidence": round(confidence, 4), "correct": correct,
        "true_label_in_flopp_training_classes": in_flopp_classes,
    })

n = len(results)
acc = sum(r["correct"] for r in results) / n
y_true = [r["true_label"] for r in results]
y_pred = [r["predicted_label"] for r in results]
all_labels = sorted(set(y_true) | set(y_pred))
macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average="macro", zero_division=0)
cm = confusion_matrix(y_true, y_pred, labels=all_labels).tolist()

majority_class = Counter(y_true).most_common(1)[0][0]
majority_acc = sum(1 for r in results if r["true_label"] == majority_class) / n

# in-panel (all UNICAMP labels covered by FLOPP training classes) subset
in_panel = [r for r in results if r["true_label_in_flopp_training_classes"]]
in_panel_acc = sum(r["correct"] for r in in_panel) / len(in_panel) if in_panel else None

output = {
    "note": (
        "n=14 proof-of-concept, not a validated accuracy figure. The EXISTING FLOPP-trained "
        "RandomForest spectral classifier (trained fresh here on all of FLOPP, same code/params as "
        "scripts/spectral_polymer_train.py) is evaluated via pure inference on UNICAMP's 14 FTIR "
        "spectra (Brazil, different lab/instrument than FLOPP). NO retraining on UNICAMP data -- "
        "this sidesteps the small-n training problem entirely, analogous to the existing FLOPP-e "
        "weathered-spectra external validation."
    ),
    "flopp_training_classes": flopp_classes,
    "n_unicamp_samples": n,
    "unicamp_class_counts": dict(Counter(y_true)),
    "all_labels_in_confusion_matrix": all_labels,
    "confusion_matrix": cm,
    "fold_results": results,
    "accuracy": round(acc, 4),
    "macro_f1": round(macro_f1, 4),
    "majority_baseline_class": majority_class,
    "majority_baseline_accuracy": round(majority_acc, 4),
    "in_panel_only": {
        "n": len(in_panel),
        "accuracy": round(in_panel_acc, 4) if in_panel_acc is not None else None,
        "note": "restricted to UNICAMP samples whose true label is present in the FLOPP training class set",
    },
}

out_path = os.path.join(ROOT, "evidence/public/unicamp_classification/flopp_model_on_unicamp_ftir_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print("\n=== SUMMARY ===")
print(f"accuracy={acc:.3f} macro_f1={macro_f1:.3f} vs majority_baseline={majority_acc:.3f} (n={n})")
print(f"in-panel-only accuracy: {in_panel_acc}")
print("Wrote:", out_path)
