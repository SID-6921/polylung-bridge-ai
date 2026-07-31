import glob, os, re, json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

BASE = os.path.expanduser("~/polylung-bridge-ai/data/microplastics_datasets/flopp_spectral_library/extracted")

def load_split(folder):
    X, y, names = [], [], []
    common_grid = np.linspace(400, 4000, 900)
    for f in glob.glob(os.path.join(BASE, folder, "*.CSV")):
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
        interp = np.interp(common_grid, wn, ab)
        X.append(interp)
        y.append(label)
        names.append(base)
    return np.array(X), np.array(y), names

Xf, yf, nf = load_split("FLOPP .csv")
Xe, ye, ne = load_split("FLOPP-e .csv")

print("FLOPP:", Xf.shape, "classes:", sorted(set(yf)))
print("FLOPP-e:", Xe.shape, "classes:", sorted(set(ye)))

# keep only classes with >=2 samples in FLOPP for stratified split
from collections import Counter
cnt = Counter(yf)
keep = {k for k, v in cnt.items() if v >= 2}
mask = np.array([lab in keep for lab in yf])
Xf, yf = Xf[mask], yf[mask]
print("FLOPP after filtering classes<2:", Xf.shape, "classes:", sorted(set(yf)))

Xtr, Xte, ytr, yte = train_test_split(Xf, yf, test_size=0.2, random_state=42, stratify=yf)

clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
clf.fit(Xtr, ytr)

def evaluate(clf, X, y, classes):
    pred = clf.predict(X)
    proba = clf.predict_proba(X)
    acc = accuracy_score(y, pred)
    macro_f1 = f1_score(y, pred, average="macro", zero_division=0)
    report = classification_report(y, pred, labels=classes, zero_division=0, output_dict=True)
    cm = confusion_matrix(y, pred, labels=classes).tolist()
    # AUC one-vs-rest, only for classes present in y
    try:
        yb = label_binarize(y, classes=classes)
        auc = roc_auc_score(yb, proba, average="macro", multi_class="ovr")
    except Exception as e:
        auc = None
    return {"accuracy": acc, "macro_f1": macro_f1, "confusion_matrix": {"labels": classes, "matrix": cm},
            "classification_report": report, "macro_auc_ovr": auc, "n": len(y)}

classes = sorted(set(yf))
results = {}
results["flopp_internal_test"] = evaluate(clf, Xte, yte, classes)

# external validation on FLOPP-e (weathered) restricted to overlapping classes
overlap = sorted(set(ye) & set(classes))
mask_e = np.array([lab in overlap for lab in ye])
Xe2, ye2 = Xe[mask_e], ye[mask_e]
print("FLOPP-e overlap eval set:", Xe2.shape, "classes:", overlap)
if len(Xe2) > 0:
    pred_e = clf.predict(Xe2)
    proba_e = clf.predict_proba(Xe2)
    acc_e = accuracy_score(ye2, pred_e)
    macro_f1_e = f1_score(ye2, pred_e, average="macro", zero_division=0)
    cm_e = confusion_matrix(ye2, pred_e, labels=classes).tolist()
    results["flopp_e_external_validation"] = {
        "accuracy": acc_e, "macro_f1": macro_f1_e, "n": len(ye2),
        "confusion_matrix": {"labels": classes, "matrix": cm_e},
        "note": "Model trained ONLY on fresh FLOPP spectra, evaluated on FLOPP-e (environmentally weathered) spectra for the classes present in both -- this is a genuine external/domain-shift validation, not a random split."
    }

results["dataset_info"] = {
    "flopp_n_total": int(Xf.shape[0]) if hasattr(Xf, 'shape') else None,
    "flopp_classes_used": classes,
    "flopp_e_n_total": int(Xe.shape[0]),
    "model": "RandomForestClassifier(n_estimators=300, class_weight=balanced)",
    "features": "FTIR absorbance interpolated onto common 900-point grid, 400-4000 cm^-1",
    "split": "80/20 stratified train/test on FLOPP; FLOPP-e used separately as external weathered-sample validation"
}

out_dir = os.path.expanduser("~/polylung-bridge-ai/evidence/public/spectral_polymer")
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "spectral_classifier_metrics.json"), "w") as fh:
    json.dump(results, fh, indent=2, default=str)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
from itertools import cycle

# confusion matrix plot (internal FLOPP test set)
cm = np.array(results["flopp_internal_test"]["confusion_matrix"]["matrix"])
cm_labels = results["flopp_internal_test"]["confusion_matrix"]["labels"]
fig, ax = plt.subplots(figsize=(9, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_labels)
disp.plot(ax=ax, cmap="Blues", colorbar=True, xticks_rotation=45)
ax.set_title(f"FLOPP spectral classifier — confusion matrix (held-out test, n={sum(sum(r) for r in cm)})")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "confusion_matrix_flopp_test.png"), dpi=150)
plt.close()

# ROC curves, one-vs-rest, internal FLOPP test set
proba_te = clf.predict_proba(Xte)
yte_bin = label_binarize(yte, classes=classes)
fig, ax = plt.subplots(figsize=(8, 7))
colors = cycle(plt.cm.tab20.colors)
for i, (cls, color) in enumerate(zip(classes, colors)):
    if yte_bin[:, i].sum() == 0:
        continue
    fpr, tpr, _ = roc_curve(yte_bin[:, i], proba_te[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=1.5, label=f"{cls} (AUC={roc_auc:.2f})")
ax.plot([0, 1], [0, 1], "k--", lw=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"FLOPP spectral classifier — ROC (one-vs-rest, held-out test, macro AUC={results['flopp_internal_test']['macro_auc_ovr']:.3f})")
ax.legend(loc="lower right", fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "roc_flopp_test.png"), dpi=150)
plt.close()

# confusion matrix for FLOPP-e external validation, if present
if "flopp_e_external_validation" in results:
    cm_e = np.array(results["flopp_e_external_validation"]["confusion_matrix"]["matrix"])
    cm_e_labels = results["flopp_e_external_validation"]["confusion_matrix"]["labels"]
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_e, display_labels=cm_e_labels)
    disp.plot(ax=ax, cmap="Oranges", colorbar=True, xticks_rotation=45)
    ax.set_title("FLOPP-trained model — confusion matrix on FLOPP-e (weathered, external validation)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix_floppe_external.png"), dpi=150)
    plt.close()

print("PLOTS_SAVED")

print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("accuracy","macro_f1","macro_auc_ovr","n")} for k, v in results.items() if isinstance(v, dict) and "accuracy" in v}, indent=2))
print("DONE")
