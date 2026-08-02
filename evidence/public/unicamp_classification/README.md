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
