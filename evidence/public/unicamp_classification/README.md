# UNICAMP SEM-image / polymer-ID classification: proof-of-concept LOOCV

**Status: n=14 proof-of-concept signal, not a validated accuracy figure.** This is the first SEM-image-to-polymer-identity classification evidence in the project (prior work was spectral or shape-only, not polymer identity).

## Why n=14, not the expected n=15

The original diagnostic identified 15 spreadsheet rows in `Dataset.xlsx` with a confirmed `Polymer` label, a matching FTIR CSV, and a matching SEM-EDS PDF. On re-derivation, two of those 15 rows are **duplicate entries sharing the same `Sample_ID`, `RC_MP_56`** (different depth/size measurements — likely two distinct physical particles that were assigned the same ID during data entry). Only one SEM-EDS PDF (`RC_MP_56.pdf`) and one FTIR CSV (`RC_MP_56.csv`) exist on disk for that ID, so only one image is physically available — the two spreadsheet rows cannot be resolved into two separate images. Both rows are labeled PE, so the ambiguity does not create a labeling conflict, but it does mean only **one** usable image exists where the row count implied two. This drops the usable set from 15 to **14 unique image-labeled particles**, and the PE count from the previously assumed 7 to the actual **6**.

Confirmed final breakdown (n=14): **PE=6, PP=3, PS=3, PVC=1, PA=1**.

## Panel overlap

The project's four-polymer target panel is PE/PP/PS/PVC. Of the 14 samples, **13 fall inside the panel** (PE=6, PP=3, PS=3, PVC=1) and **1 falls outside it**: `RC_MP_01` (PA / polyamide), as expected.

## LOOCV setup

Swin-T (torchvision, ImageNet-pretrained), full fine-tuning, one model trained from scratch per fold on the remaining n-1 images, 15 epochs, heavy augmentation (random crop/flip/rotation/color jitter), no internal validation split (not enough data to spare any). Two versions run:
- **Full n=14** (all 5 classes)
- **PE/PP/PS only, n=12** (excludes the two singleton classes, PA and PVC)

## Headline result: negative

| Version | LOOCV accuracy | Macro F1 | Majority baseline |
|---|---|---|---|
| Full n=14 | **7.1%** (1/14 correct) | 0.057 | 42.9% (always predict PE) |
| PE/PP/PS only, n=12 | **33.3%** (4/12 correct) | 0.238 | 50.0% (always predict PE) |

**Both LOOCV accuracies are worse than trivially guessing the majority class.** Full per-fold fine-tuning of a Swin-T backbone on only ~11–13 training images catastrophically overfits: the model latches onto fold-specific noise rather than any transferable polymer-visual signature, and generalizes worse than a constant-output baseline. This is a genuine negative result, not a modeling bug, and per the PI's explicit instruction it is reported plainly rather than softened.

This result is concrete, direct evidence for why Aim 1 needs a properly sized paired SEM+polymer-ID benchmark (the project's n=120 target): n=14 is empirically insufficient for even simple transfer-learning fine-tuning to beat a trivial baseline here, let alone support a production-grade classifier.

## Singleton-class limitation

PA (n=1) and PVC (n=1) each have exactly one example in the full n=14 set. In true LOOCV, when a singleton class's only example is held out, its training set contains zero examples of that class — the model cannot have learned that class's visual signature and structurally cannot predict it correctly on that fold, regardless of model quality. This is a fundamental small-n limitation, not a bug. Both singleton folds (RC_MP_01/PA and RC_MP_09/PVC) fail in the full n=14 run by construction. The PE/PP/PS-only n=12 result excludes them and is the more interpretable of the two accuracy figures, though it is itself also below baseline.

## Files

- `evidence/public/unicamp_classification/manifest.csv` — sample_id, polymer_label, image_path, ftir_csv_path for all 14 samples
- `evidence/public/unicamp_classification/loocv_results.json` — per-fold results, aggregate accuracy/macro-F1 for both versions, majority baseline, panel-overlap breakdown
- `evidence/public/unicamp_classification/README.md` — this file
- `data/unicamp_labeled_images/` — extracted SEM micrographs, named `{Sample_ID}_{PolymerLabel}.png`
- `scripts/step1_derive_15.py` — spreadsheet/file cross-reference derivation
- `scripts/step2_extract_images.py` — SEM image extraction (pdfimages) and manifest generation
- `scripts/unicamp_loocv.py` — LOOCV training/evaluation script

---

## Follow-up: frozen pretrained features + linear classifier (SVM / logistic regression)

Suggested by Selina Park (ML co-researcher) as the more appropriate approach for n this small: full fine-tuning of a large transformer has far more parameters than training examples per fold and overfits easily, whereas a linear classifier trained on frozen, high-quality pretrained features has orders of magnitude fewer free parameters. This is a legitimate hypothesis worth testing directly, so we ran it under the exact same LOOCV protocol as the fine-tuning result above, decided before looking at any per-fold outcome, for direct comparability.

**Setup.** Same Swin-T backbone (torchvision, ImageNet-pretrained), but this time frozen entirely: classification head removed, `eval()` mode, no gradient updates, a single forward pass per image to get the 768-dim penultimate-layer embedding. Features were extracted once for all 14 images. For each LOOCV fold, a linear classifier (`sklearn.svm.LinearSVC` or `sklearn.linear_model.LogisticRegression`, default `C=1.0`, features standardized with `StandardScaler` fit on the fold's training data only) was trained on the frozen embeddings of the other n-1 samples and used to predict the held-out sample. Same two versions as before: full n=14 (5 classes) and PE/PP/PS-only n=12 (excludes singleton classes PA, PVC).

**Result: also negative, and worse than the fine-tuning result on the full n=14 set.**

| Version | Classifier | LOOCV accuracy | Macro F1 | Majority baseline |
|---|---|---|---|---|
| Full n=14 | Linear SVM | **0.0%** (0/14) | 0.000 | 42.9% |
| Full n=14 | Logistic regression | **7.1%** (1/14) | 0.027 | 42.9% |
| PE/PP/PS only, n=12 | Linear SVM | **0.0%** (0/12) | 0.000 | 50.0% |
| PE/PP/PS only, n=12 | Logistic regression | **8.3%** (1/12) | 0.051 | 50.0% |

Every configuration falls well below its majority baseline. Logistic regression matches the fine-tuning approach's 7.1% on the full n=14 set (both get exactly 1/14 folds right, though not the same fold) but does not beat it, and is still far below baseline. Linear SVM performs worse than fine-tuning in both versions (0% vs. 7.1%/33.3%). Freezing the backbone did not rescue the signal.

**Interpretation.** The overfitting-from-fine-tuning hypothesis does not appear to be the dominant failure mode here — removing the fine-tuning parameters entirely still leaves the classifier unable to beat trivial guessing. The more likely explanation is that generic ImageNet features (whether fine-tuned or used frozen) do not capture whatever visual signal, if any, distinguishes these five polymer classes in this particular SEM image set, and/or that n=14 (11-13 per fold) is simply too small for any classifier, however parameter-light, to learn a reliable class boundary here. This is consistent with — and reinforces, rather than resolves — the project's existing conclusion that Aim 1 needs a properly sized paired SEM+polymer-ID benchmark; a linear classifier's inability to do better than fine-tuning on the same 14 images is itself evidence that the bottleneck is data quantity/quality, not model capacity.

**Files.**
- `evidence/public/unicamp_classification/loocv_frozen_features_results.json` — per-fold results (predictions, correctness) for all four configurations (SVM/logreg × full/PE-PP-PS-only), aggregate accuracy/macro-F1, majority baselines
- `scripts/unicamp_loocv_frozen_features.py` — frozen-feature extraction + per-fold linear classifier LOOCV script

---

## Third-pass attempts: spectral cross-validation, fused features, classical features

Three further, independent methods were tried on the n=14 UNICAMP set, none of which involve full fine-tuning (to rule out overfitting-from-fine-tuning as the sole cause of the earlier negative results).

| Method | n=14 accuracy | n=14 majority baseline | n=12 (PE/PP/PS) accuracy | n=12 majority baseline |
|---|---|---|---|---|
| FLOPP-trained spectral model, pure inference on UNICAMP spectra (no training on UNICAMP at all) | 0.0% | 42.9% | n/a (single run, not split by panel) | n/a |
| Classical GLCM texture + shape descriptors, linear SVM/logreg | 28.6% | 42.9% | 25.0% | 50.0% |
| Fused frozen image embedding + spectral PCA features, linear SVM/logreg | 0.0–7.1% | 42.9% | 0.0–16.7% | 50.0% |

**FLOPP-model-on-UNICAMP-spectra finding (important, separate from the accuracy result):** diagnostic inspection of raw spectral intensity values revealed a large scale mismatch between the two instruments/labs -- FLOPP absorbance values range roughly 0-113 (mean ~93), UNICAMP's range roughly -4 to 6.6 (mean ~1.3). The trained model sees UNICAMP spectra as far outside its training distribution, collapsing predictions to a single class. This is a genuine, actionable methodological finding (cross-instrument spectral normalization would need to be solved before any cross-lab spectral transfer is attempted), not evidence that no polymer signal exists in the UNICAMP spectra -- but as tested, this approach produced 0% and no usable classification result.

## Overall verdict across all five methods tried

Combined with the two earlier attempts (full fine-tuning: 7.1% n=14 / 33.3% n=12; frozen-feature SVM/logreg: 0.0-8.3%), **five independent methods have now been tried on the UNICAMP n=14 set, and all five perform at or below the majority-class baseline.** No configuration of architecture (Swin-T fine-tuned, Swin-T frozen), classifier (deep, linear SVM, logistic regression), or feature type (deep image, classical image, spectral, fused image+spectral) produced a result that beats trivially guessing the most common class.

This is a consistent, honest, and now well-triangulated conclusion: **n=14 cannot support a classification claim by any method attempted so far.** The bottleneck is data quantity, not model choice, feature type, or evaluation protocol. This is not a negative reflection on the underlying feasibility of SEM+FTIR polymer classification -- it is direct, first-hand evidence for exactly the claim Aim 1's Significance section makes: a benchmark at this scale (n=14) is fundamentally insufficient, and the field-wide absence of a larger public paired dataset is a real, well-characterized gap, not a hypothetical one. Per PI instruction, this data and every attempt made on it are excluded from the grant application's Preliminary Studies section, but are preserved here as an honest, reproducible lab record.
