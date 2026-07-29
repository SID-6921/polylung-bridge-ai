# PD-1 Swin-T on LungHist700

## Overview

PD-1 evaluates the feasibility of a Swin Transformer (Swin-T) pipeline for
binary classification of normal and pathological lung H&E images from
LungHist700.

The results in this folder are preliminary feasibility results and should not
be interpreted as independent clinical validation or as a final
polymer-specific classifier.

## Deliverables

- `runbook.md`: training and evaluation workflow
- `dataset_split_summary.json`: generated train and validation split counts
- `baseline_comparison.csv`: comparison between the initial CPU pilot and GPU run
- `pd1_metrics.json`: original 1-epoch CPU pilot results
- `pd1_metrics_gpu_local.json`: final 8-epoch NVIDIA L40S GPU results
- `pd1_metrics_gpu_local_33654808.json`: archived copy associated with SLURM job 33654808
- `gradcam_pathological_panel.png`: representative Grad-CAM visualization
- `representative_histology_images.png`: representative normal and pathological H&E images

## Current Status

The initial 1-epoch CPU pilot validated the reproducibility of the training and
evaluation pipeline but showed majority-class collapse:

- Accuracy: 0.7826
- Macro F1: 0.439
- Normal-class F1: 0.000

An 8-epoch GPU training run was subsequently completed on an NVIDIA L40S using
class-balanced loss and ImageNet-pretrained Swin-T weights.

## Final L40S GPU Results

- Validation accuracy: **0.9638**
- Macro F1: **0.9461**
- Weighted F1: **0.9635**
- Normal-class F1: **0.9153**
- Pathological-class F1: **0.9770**

Confusion matrix:

| Actual class | Predicted normal | Predicted pathological |
|---|---:|---:|
| Normal | 27 | 3 |
| Pathological | 2 | 106 |

The GPU run resolved the majority-class collapse observed in the CPU pilot and
demonstrated strong classification performance across both classes.

These findings remain preliminary because evaluation was performed using an
internal validation split rather than an independent external dataset.

## Compute Environment

- Platform: USF CIRCE / Helios
- GPU: NVIDIA L40S
- Epochs: 8
- Batch size: 8
- Learning rate: 1e-4
- Class-balanced loss: enabled
- Runtime: approximately 4 minutes 59 seconds
- SLURM job ID: 33654808

## Reproducing the Dataset Split

```bash
python scripts/pd1_prepare_lunghist700_binary.py \
  --source data/raw/LungHist700/data/images \
  --output data/lunghist700_binary \
  --summary evidence/public/pd1/dataset_split_summary.json
```

## Reproducing GPU Training

```bash
python scripts/pd1_train_swin_lunghist700.py \
  --data-dir data/lunghist700_binary \
  --epochs 8 \
  --batch-size 8 \
  --num-workers 0 \
  --lr 1e-4 \
  --published-baseline 0.0000 \
  --pretrained \
  --output evidence/public/pd1/pd1_metrics_gpu_local.json
```

PD-1 reporting should prioritize macro F1, per-class F1, and the confusion
matrix over accuracy alone.

## Model Checkpoint

The trained model checkpoint is retained on the institutional server:

```text
evidence/public/pd1/pd1_swin_l40s_model.pt
```

The checkpoint is not committed to GitHub because its size exceeds the standard
GitHub file limit.

## Interpretation

The current results support the feasibility of the GPU-based Swin-T training
and evaluation pipeline. They should not be presented as externally validated
classifier performance or as a final clinical model.

## Next Steps

- Validate performance using an independent dataset or patient-level split
- Save image filenames and prediction probabilities
- Review misclassified images
- Generate additional Grad-CAM examples
- Confirm that no patient- or slide-level leakage exists in the data split
