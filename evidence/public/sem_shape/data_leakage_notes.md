# Data leakage / triviality investigation — SEM shape classifier

**Bottom line: the 100% accuracy / 1.0 macro-F1 / 1.0 AUC results from all three
models (Swin-T, ResNet50, EfficientNet-B0) should NOT be presented as evidence
that the models learned to classify particle morphology. The task is confounded
by an acquisition-batch effect: each shape class in this split is drawn from a
small, near-disjoint set of source products/imaging sessions that is shared
identically across train/val/test, so any nuisance feature tied to that session
(background, lighting, scale-bar style, sensor/compression signature, or raw
image resolution) is a usable shortcut for the label, independent of particle
shape.**

## 1. Image dimensions / aspect ratio per class

| class | n | widths seen | heights seen | aspect ratios seen |
|---|---|---|---|---|
| bead | 51 | {1280} | {900} | {1.422} |
| fibre | 90 | {1279, 1280, 1536, 2560} | {850, 899, 900, 1080, 1800} | {1.422, 1.423, 1.506} |
| fragment | 96 | {800, 1280, 2560} | {561, 898, 1794, 1800} | {1.422, 1.425, 1.426, 1.427} |

Aspect ratio is essentially constant (~1.42) across all three classes, so
aspect ratio alone is not a usable shortcut. However, **bead is the only class
with a single, exact, constant raw resolution (1280x900) for every one of its
51 images** — a direct fingerprint of "this came from the Dove-microbeads
imaging session," not "this is round." fibre and fragment each mix 3-4 raw
resolutions tied to their respective source products.

## 2. Trivial-feature sanity check (requested diagnostic)

Trained a RandomForestClassifier on **only** `[width, height, width/height]`
per image (no pixel data, no CNN) on the same train split, evaluated on the
same val/test split as the real models:

- val: accuracy = 0.778, macro-F1 = 0.593
- test: accuracy = 0.771, macro-F1 = 0.573

This is well above the majority-class baseline (~0.40, fragment) but **not**
100% — so raw image dimensions alone do not fully explain the CNNs' perfect
scores. Dimensions are a real but partial confound; the dominant shortcut is
almost certainly session/background/texture signal from the source-product
confound described below (available to a CNN operating on pixels, not to a
3-feature model).

## 3. Duplicate / leakage check across splits

- **Exact byte-identical duplicates across train/val/test: 0** (confirmed via
  MD5 hash of every file; the dedup step in `sem_shape_prepare.py` also found
  0 duplicates before splitting). So there is no literal copy-paste leakage.
- **Source-product / imaging-session overlap across splits (the real issue):**
  grouping filenames by source-product prefix shows each class draws from a
  very small, fixed set of products/sessions that appears **in all three
  splits**:
  - `bead`: **1 product total** ("Dove_men_acrylates_copolymer_microbeads"),
    present in train, val, and test. Every bead image in every split is the
    same product/session.
  - `fragment`: **4 products total** ("Foam_No", "Particle_No",
    "Particle_2_No", "PS_Isopropyl_Alcohol_Suspension"), and all 4 appear in
    train, val, and test.
  - `fibre`: more diverse (PAN/PE/PP/polyester at several kV settings), but
    still drawn from the same overall pool split at the file level, not held
    out by material.

Because held-out splits reuse the *same* source sessions as training, the
val/test evaluation is not testing generalization to new material/imaging
conditions — it is largely testing "which of these already-seen sessions does
this image belong to," which is a much easier (and different) problem than
morphology classification, and is consistent with an image classifier reaching
saturation performance quickly.

## 4. Grad-CAM cross-check

See `evidence/public/sem_shape/gradcam/`. Overlays were inspected for whether
attention concentrates on particle edges/texture vs. diffuse/background
regions — see the training report for the qualitative call; if attention is
diffuse or background-dominated rather than particle-focused, that further
supports the shortcut-learning explanation above.

## Recommendation

Do not report these metrics as validated shape-classification performance.
To get a meaningful signal, the dataset needs a **product/session-level
(grouped) split** — holding out entire source products, not just individual
images, from train — and ideally more distinct products per class so shape
is not confounded 1:1 with acquisition session. This pilot's honest
conclusion is: *current results are not evidence the model learned particle
morphology; they mostly reflect that the three classes come from
near-disjoint, non-held-out imaging sessions.*

## 5. Robustness/perturbation results as further evidence

`robustness_results.json` shows Swin-T at **100% accuracy / 1.0 macro-F1 under
every perturbation tested** (horizontal flip, brightness jitter, rotation
+-15 degrees, contrast jitter) with zero degradation vs. the clean baseline.
This total invariance is further evidence *for* the shortcut/session-confound
hypothesis above, not against it: a model reading genuine particle
morphology would typically show at least some sensitivity to rotation or
brightness changes, whereas perfect robustness across all these perturbations
is more consistent with the model keying on a coarse, perturbation-robust
signal (background composition, scale-bar/session artifact) rather than
fine-grained particle shape/texture.
