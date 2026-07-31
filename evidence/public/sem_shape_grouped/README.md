# SEM shape classifier — product/session-grouped split (honest re-run)

## What changed vs. the original (flawed) run

The original split (`scripts/sem_shape_prepare.py`, stratified by class, seed=42)
let images from the same source product/imaging session appear in train, val
AND test simultaneously (documented in
`evidence/public/sem_shape/data_leakage_notes.md`). All three models
(Swin-T, ResNet50, EfficientNet-B0) hit 100% accuracy / 1.0 macro-F1 / 1.0 AUC
on that split, but this was diagnosed as invalid evaluation, not evidence of
learned shape morphology.

This run uses `scripts/sem_shape_prepare_grouped.py`, which assigns whole
source products/imaging sessions to exactly one split (train, val, or test)
so no session leaks across splits. See
`data/sem_shape_grouped_split/dataset_split_summary_grouped.json` for the
full group->split assignment.

**Group assignment (manual, chosen to land close to 70/15/15):**

| class | train groups | val groups | test groups |
|---|---|---|---|
| fibre | polyester (48), PP (10) | PE (12) | PAN (20) |
| fragment | PS_Isopropyl_Alcohol_Suspension (41), Particle_No (28) | Particle_2_No (15) | Foam_No (12) |
| bead | Dove_men_acrylates_copolymer_microbeads (51) | — | — |

Actual overall split ratio: train 75.1% (178 imgs) / val 11.4% (27 imgs) /
test 13.5% (32 imgs) — close to but not exactly 70/15/15, because group
integrity took priority over hitting the exact ratio (only 4 groups exist
for fibre and for fragment, so ratios are constrained by group sizes).

## The `bead` class holdout problem

**`bead` has exactly 1 source product in the entire dataset**
(`Dove_men_acrylates_copolymer_microbeads`, 51 images). A true grouped split
cannot put any bead images in val or test without reusing the same
session in both train and eval — i.e. reintroducing the original leakage.
So all 51 bead images went to train, and **bead cannot be evaluated on
held-out data with this dataset as it currently exists.** This is not a bug
to patch around; it is itself evidence that more source diversity (more
bead products/sessions) is needed before bead shape classification can be
validated at all.

Because bead is absent from val/test, **the held-out evaluation below is
effectively a 2-class (fibre vs. fragment) evaluation** — both of those
classes do have >=1 held-out group per split. That is the honest way to
read the numbers below; the "bead" row in the confusion matrix and
per-class F1 is 0/undefined by construction (no bead support in test), not
a genuine failure to classify beads.

## Honest results — grouped split (test set, n=32: 20 fibre + 12 fragment, 0 bead)

| model | accuracy | macro-F1* | fibre F1 | fragment F1 | bead F1 |
|---|---|---|---|---|---|
| Swin-T | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0 (no support) |
| ResNet50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0 (no support) |
| EfficientNet-B0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0 (no support) |

\* macro-F1 computed over classes present in y_true/y_pred (fibre, fragment
only, since bead has zero test support) — i.e. this is a 2-class macro-F1,
not a 3-class one.

**Explicit comparison:** flawed same-session split: 100% accuracy (invalid,
confounded by session leakage across train/val/test); session-grouped
split: 100% accuracy on the fibre-vs-fragment held-out evaluation (valid
w.r.t. group leakage, honest result) — the split fix did **not** lower
accuracy in this instance.

## Why 100% again, and what it does and doesn't mean

This is a real, group-disjoint result: no fibre or fragment image in
val/test comes from the same source product as any train image. Unlike the
original run, this evaluation is testing generalization to a genuinely
unseen product/session, not "which known session is this."

However, it should **not** be over-interpreted as strong validated evidence
either, for two reasons that are structural to this dataset, not
choices made in this run:

1. **Each held-out split (val, test) is still built from a single source
   product/session** (val = PE only, test = PAN only, for fibre; similarly
   fragment's val/test are single-product). A model could still be
   keying on session-level nuisance cues specific to that one held-out
   session (background, lighting, resolution) rather than fibre/fragment
   shape per se — we just can no longer distinguish "learned shape" from
   "learned this one new session's signature" with only 4 groups per
   class. Confirming genuine shape generalization would need >=2 held-out
   groups per class per split so a model can't trivially fit a single
   session's fingerprint.
2. **`bead` could not be evaluated at all** (see above) — this pilot
   provides no valid held-out evidence about bead classification.

**Bottom line:** the grouped split removes the specific leakage mechanism
documented in `data_leakage_notes.md` (same session in train AND eval), and
the fibre-vs-fragment result stands up to that fix. But the dataset's small
number of distinct products per class (1 for bead, 4 for fibre, 4 for
fragment) means this remains a preliminary pilot, not proof of
morphology-based generalization — more source products per class are
needed for a fully conclusive test.

## Files

- `data/sem_shape_grouped_split/dataset_split_summary_grouped.json` — full
  group->split assignment and counts
- `data/sem_shape_grouped_split/{train,val,test}.csv` — image path/label CSVs
- `evidence/public/sem_shape_grouped/sem_shape_grouped_metrics_{swin_t,resnet50,efficientnet}.json`
  — full metrics (accuracy, precision/recall, macro-F1, per-class F1,
  confusion matrix, ROC/AUC) for each model
- `evidence/public/sem_shape_grouped/roc_{swin_t,resnet50,efficientnet}.png`
  — ROC curves
