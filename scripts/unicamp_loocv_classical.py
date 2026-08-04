"""
LOOCV on the n=14 UNICAMP SEM images using classical (non-deep) texture/shape
features -- GLCM texture descriptors + basic region shape descriptors -- instead
of deep ImageNet features. Rationale: generic ImageNet features may not transfer
well to SEM micrograph texture; hand-crafted descriptors sometimes do better in
very-low-data regimes.

Same LOOCV protocol as scripts/unicamp_loocv.py and unicamp_loocv_frozen_features.py
(full n=14, and PE/PP/PS-only n=12) for direct comparability.

n=14 proof-of-concept, not a validated accuracy figure.
"""
import os, csv, json
import numpy as np
from PIL import Image
from collections import Counter
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

SEED = 42
ROOT = os.path.expanduser("~/polylung-bridge-ai")
MANIFEST = os.path.join(ROOT, "evidence/public/unicamp_classification/manifest.csv")

with open(MANIFEST) as f:
    rows = list(csv.DictReader(f))
samples = [(r["sample_id"], r["polymer_label"], os.path.join(ROOT, r["image_path"])) for r in rows]
print("Loaded", len(samples), "samples:", Counter([s[1] for s in samples]))


def extract_classical_features(path):
    im = Image.open(path).convert("L").resize((256, 256))
    arr = np.array(im)

    # GLCM texture features
    glcm = graycomatrix(arr, distances=[1, 3], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    correlation = graycoprops(glcm, "correlation").mean()
    dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    asm = graycoprops(glcm, "ASM").mean()

    # simple intensity stats
    mean_int, std_int = arr.mean(), arr.std()

    # basic shape descriptors via Otsu threshold + largest region
    try:
        thresh = threshold_otsu(arr)
        binary = arr > thresh
        lbl = label(binary)
        regions = regionprops(lbl)
        if regions:
            largest = max(regions, key=lambda r: r.area)
            aspect_ratio = largest.major_axis_length / (largest.minor_axis_length + 1e-6)
            extent = largest.extent
            solidity = largest.solidity
            eccentricity = largest.eccentricity
        else:
            aspect_ratio = extent = solidity = eccentricity = 0.0
    except Exception:
        aspect_ratio = extent = solidity = eccentricity = 0.0

    return np.array([contrast, homogeneity, energy, correlation, dissimilarity, asm,
                      mean_int, std_int, aspect_ratio, extent, solidity, eccentricity])


features = {}
for sid, label_, path in samples:
    feat = extract_classical_features(path)
    features[sid] = feat
    print(f"extracted classical features for {sid} ({label_}): dim={feat.shape[0]}")


def make_clf(kind):
    if kind == "svm":
        return LinearSVC(C=1.0, max_iter=10000, random_state=SEED)
    return LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)


def run_loocv(subset_samples, tag, clf_kind):
    classes = sorted(set(s[1] for s in subset_samples))
    n = len(subset_samples)
    fold_results = []
    for i in range(n):
        test_sid, test_label, _ = subset_samples[i]
        train_items = [subset_samples[j] for j in range(n) if j != i]
        X_train = [features[sid] for sid, _, _ in train_items]
        y_train = [lbl for _, lbl, _ in train_items]
        X_test = features[test_sid]

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
    results[f"loocv_full_n14_{clf_kind}"] = run_loocv(samples, "full_n14", clf_kind)
    results[f"loocv_pe_pp_ps_only_{clf_kind}"] = run_loocv(subset_2plus, "pe_pp_ps_n%d" % len(subset_2plus), clf_kind)

output = {
    "note": ("n=14 proof-of-concept, not a validated accuracy figure. Classical (non-deep) GLCM "
              "texture descriptors (contrast, homogeneity, energy, correlation, dissimilarity, ASM "
              "at distances 1,3 and 4 angles, averaged) + basic Otsu-threshold region shape "
              "descriptors (aspect ratio, extent, solidity, eccentricity) + basic intensity stats, "
              "12-dim feature vector per image, fed to a simple linear classifier per LOOCV fold. "
              "Same protocol as the deep-feature scripts for direct comparability."),
    "dataset_n": len(samples),
    "dataset_class_counts": dict(Counter(s[1] for s in samples)),
    **results,
}

out_path = os.path.join(ROOT, "evidence/public/unicamp_classification/loocv_classical_features_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)

print("\n=== SUMMARY ===")
for key, r in results.items():
    print(f"{key}: acc={r['loocv_accuracy']:.3f} macro_f1={r['loocv_macro_f1']:.3f} (majority {r['majority_baseline_accuracy']:.3f}), n={r['n']}")
print("Wrote:", out_path)
