# PD-1 Runbook (Swin on LungHist700)

## 1) Extract and prepare dataset
If you have `LungHist700.rar`, extract to:

- `data/raw/LungHist700/data/images`

Then run:
```bash
python scripts/pd1_prepare_lunghist700_binary.py --source data/raw/LungHist700/data/images --output data/lunghist700_binary --summary evidence/public/pd1/dataset_split_summary.json
```

Prepared structure:
- `data/lunghist700_binary/train/normal`
- `data/lunghist700_binary/train/pathological`
- `data/lunghist700_binary/val/normal`
- `data/lunghist700_binary/val/pathological`

## 2) Install dependencies
```bash
pip install -r requirements-pd1.txt
```

## 3) Train and evaluate
```bash
python scripts/pd1_train_swin_lunghist700.py \
  --data-dir data/lunghist700_binary \
  --epochs 8 \
  --batch-size 8 \
  --num-workers 0 \
  --lr 1e-4 \
  --published-baseline 0.0000 \
  --pretrained \
  --output evidence/public/pd1/pd1_metrics.json
```

Notes:
- Class-balanced loss is enabled by default to reduce majority-class collapse risk.
- Use `--disable-class-balanced-loss` only for ablation.
- Primary report metrics are `macro_f1`, per-class F1, and confusion matrix.

## 4) Fill baseline comparison
Update `baseline_comparison.csv` with published baseline and generated model metric.
