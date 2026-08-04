"""
LOOCV on the n=14 UNICAMP samples using FUSED features: frozen ImageNet-pretrained
Swin-T image embedding + FTIR spectral features (PCA-reduced), concatenated.
This is the most on-target technical approach: it's literally the combined
SEM+FTIR technology the project proposes, tested for the first time as one
fused classifier rather than two separate modality-only attempts.

Given UNICAMP's raw FTIR intensity scale differs drastically from FLOPP's
(diagnosed in unicamp_flopp_crossval.py), spectral features here are used only
in their OWN right distribution (not cross-instrument transfer) -- standardized
per-fold like the image features, so the scale mismatch issue does not apply
within this LOOCV (train and test spectra are all UNICAMP, same instrument).

n=14 proof-of-concept, not a validated accuracy figure.
"""
import os, csv, json
import numpy as np
import torch, torch.nn as nn
import torchvision
from torchvision import transforms
from PIL import Image
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

SEED = 42
torch.manual_seed(SEED)
ROOT = os.path.expanduser("~/polylung-bridge-ai")
MANIFEST = os.path.join(ROOT, "evidence/public/unicamp_classification/manifest.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COMMON_GRID = np.linspace(400, 4000, 900)

with open(MANIFEST) as f:
    rows = list(csv.DictReader(f))
samples = [(r["sample_id"], r["polymer_label"], os.path.join(ROOT, r["image_path"]),
            os.path.join(ROOT, r["ftir_csv_path"])) for r in rows]
print("Loaded", len(samples), "samples:", Counter([s[1] for s in samples]))

eval_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_unicamp_spectrum(path):
    wn, ab = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=";")
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            try:
                wn.append(float(row[0])); ab.append(float(row[1]))
            except ValueError:
                continue
    wn = np.array(wn); ab = np.array(ab)
    order = np.argsort(wn)
    return np.interp(COMMON_GRID, wn[order], ab[order])


# --- Extract image features (frozen Swin-T) ---
backbone = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.IMAGENET1K_V1)
backbone.head = nn.Identity()
backbone = backbone.to(DEVICE)
backbone.eval()

img_features = {}
with torch.no_grad():
    for sid, label_, img_path, ftir_path in samples:
        im = Image.open(img_path).convert("L")
        t = eval_tf(im).unsqueeze(0).to(DEVICE)
        feat = backbone(t)[0].cpu().numpy()
        img_features[sid] = feat
del backbone
torch.cuda.empty_cache()
print("Image feature dim:", next(iter(img_features.values())).shape[0])

# --- Extract spectral features, then PCA-reduce (n=14 -> at most 13 components meaningful) ---
spec_raw = {}
for sid, label_, img_path, ftir_path in samples:
    spec_raw[sid] = load_unicamp_spectrum(ftir_path)

spec_matrix = np.array([spec_raw[s[0]] for s in samples])
n_components = min(8, len(samples) - 2)  # keep modest relative to n=14
pca = PCA(n_components=n_components, random_state=SEED)
spec_pca = pca.fit_transform(spec_matrix)
print(f"Spectral PCA: {spec_matrix.shape} -> {spec_pca.shape}, explained_var_ratio_sum={pca.explained_variance_ratio_.sum():.3f}")
spec_features = {samples[i][0]: spec_pca[i] for i in range(len(samples))}

# --- Fuse: concatenate image embedding + spectral PCA features ---
fused_features = {}
for sid, label_, _, _ in samples:
    fused_features[sid] = np.concatenate([img_features[sid], spec_features[sid]])
print("Fused feature dim:", next(iter(fused_features.values())).shape[0])


def make_clf(kind):
    if kind == "svm":
        return LinearSVC(C=1.0, max_iter=10000, random_state=SEED)
    return LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)


def run_loocv(subset_samples, tag, clf_kind, feat_dict):
    classes = sorted(set(s[1] for s in subset_samples))
    n = len(subset_samples)
    fold_results = []
    for i in range(n):
        test_sid, test_label = subset_samples[i][0], subset_samples[i][1]
        train_items = [subset_samples[j] for j in range(n) if j != i]
        X_train = [feat_dict[s[0]] for s in train_items]
        y_train = [s[1] for s in train_items]
        X_test = feat_dict[test_sid]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform([X_test])

        clf = make_clf(clf_kind)
        clf.fit(X_train_s, y_train)
        pred_label = clf.predict(X_test_s)[0]
        correct = (pred_label == test_label)
        print(f"[{tag}/{clf_kind}] held out={test_sid} ({test_label}) -> pred={pred_label} correct={correct}")
        fold_results.append({"sample_id": test_sid, "true_label": test_label,
                              "predicted_label": pred_label, "correct": bool(correct)})

    acc = sum(r["correct"] for r in fold_results) / n
    y_true = [r["true_label"] for r in fold_results]
    y_pred = [r["predicted_label"] for r in fold_results]
    macro_f1 = f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)
    majority_class = Counter(s[1] for s in subset_samples).most_common(1)[0][0]
    majority_acc = sum(1 for s in subset_samples if s[1] == majority_class) / n
    return {"tag": tag, "classifier": clf_kind, "n": n, "classes": classes,
            "class_counts": dict(Counter(s[1] for s in subset_samples)),
            "fold_results": fold_results, "loocv_accuracy": round(acc, 4),
            "loocv_macro_f1": round(macro_f1, 4), "majority_baseline_class": majority_class,
            "majority_baseline_accuracy": round(majority_acc, 4)}


subset_2plus = [s for s in samples if s[1] in ("PE", "PP", "PS")]
results = {}
for clf_kind in ("svm", "logreg"):
    results[f"loocv_full_n14_{clf_kind}"] = run_loocv(samples, "full_n14", clf_kind, fused_features)
    results[f"loocv_pe_pp_ps_only_{clf_kind}"] = run_loocv(subset_2plus, "pe_pp_ps_n%d" % len(subset_2plus), clf_kind, fused_features)

output = {
    "note": ("n=14 proof-of-concept, not a validated accuracy figure. Fused features: frozen "
              "ImageNet-pretrained Swin-T image embedding (no fine-tuning) concatenated with "
              "PCA-reduced UNICAMP FTIR spectral features (8 components, within-UNICAMP PCA, no "
              "cross-instrument transfer -- avoids the raw-intensity-scale mismatch diagnosed "
              "against FLOPP). Simple linear classifier per LOOCV fold, same protocol as prior "
              "scripts for direct comparability. This is the most technically on-target attempt: "
              "the fused SEM+FTIR representation is what the project's actual technology proposes."),
    "dataset_n": len(samples),
    "dataset_class_counts": dict(Counter(s[1] for s in samples)),
    "spectral_pca_components": n_components,
    "spectral_pca_explained_variance_ratio_sum": round(float(pca.explained_variance_ratio_.sum()), 4),
    **results,
}

out_path = os.path.join(ROOT, "evidence/public/unicamp_classification/loocv_fused_features_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print("\n=== SUMMARY ===")
for key, r in results.items():
    print(f"{key}: acc={r['loocv_accuracy']:.3f} macro_f1={r['loocv_macro_f1']:.3f} (majority {r['majority_baseline_accuracy']:.3f}), n={r['n']}")
print("Wrote:", out_path)
