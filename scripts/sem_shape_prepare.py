"""
Standardize the Toronto microplastics SEM dataset for a 3-class SHAPE
(fibre / bead / fragment) classification pilot -- NOT a polymer classifier.

Only the annotated *full-frame micrograph* image folders are used (the
"image" subfolder next to each shape's segmentation "label" folder). The
raw "original_image_*" folders are excluded (no processed annotation, and
largely redundant with the "image" folders). fibre additionally has a
separate per-particle crop dataset (fibre_1crop / fibre_2crop, 512x512
single-particle crops) that bead/fragment do NOT have an equivalent for --
mixing crop-level fibre samples with frame-level bead/fragment samples
would confound shape with image granularity, so those crops are
deliberately NOT used here. This asymmetry is recorded in the summary JSON.
"""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}

SHAPE_DIRS = {
    "fibre": ["fibre/fibre_1/image", "fibre/fibre_2/image"],
    "bead": ["bead/bead/image"],
    "fragment": ["fragment/fragment_1/image", "fragment/fragment_2/image"],
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def stratified_split(items, seed, ratios=(0.70, 0.15, 0.15)):
    by_class = {}
    for it in items:
        by_class.setdefault(it["label"], []).append(it)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for label, group in by_class.items():
        rng.shuffle(group)
        n = len(group)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to dataset3/dataset3")
    ap.add_argument("--out-dir", required=True, help="Where to write train/val/test csv + summary json")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    seen_hashes = {}
    dup_count = 0
    per_class_raw = {}
    for label, subdirs in SHAPE_DIRS.items():
        for sub in subdirs:
            d = data_root / sub
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
            per_class_raw.setdefault(label, 0)
            per_class_raw[label] += len(files)
            for p in files:
                h = md5_of(p)
                if h in seen_hashes:
                    dup_count += 1
                    continue
                seen_hashes[h] = str(p)
                items.append({"path": str(p), "label": label, "source_subdir": sub, "md5": h})

    train, val, test = stratified_split(items, args.seed)

    def write_csv(name, rows):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            for r in rows:
                w.writerow([r["path"], r["label"]])

    write_csv("train.csv", train)
    write_csv("val.csv", val)
    write_csv("test.csv", test)

    def counts(rows):
        c = {}
        for r in rows:
            c[r["label"]] = c.get(r["label"], 0) + 1
        return c

    summary = {
        "task": "SEM particle-morphology (SHAPE) classification pilot: fibre vs bead vs fragment. "
                "NOT polymer/chemical classification -- Toronto dataset has no FTIR/chemical labels.",
        "seed": args.seed,
        "split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "source_dirs_used": SHAPE_DIRS,
        "excluded": [
            "original_image_* raw folders (unprocessed full micrographs, no usable per-image annotation used here)",
            "fibre_1crop / fibre_2crop per-particle 512x512 crops (no equivalent crop set exists for bead/fragment; "
            "mixing crop-level fibre samples with frame-level bead/fragment samples would confound shape with "
            "image granularity, so all 3 classes use full-frame annotated micrographs for consistency)",
        ],
        "raw_file_counts_before_dedup": per_class_raw,
        "duplicate_files_removed": dup_count,
        "total_after_dedup": len(items),
        "counts_per_class_per_split": {
            "train": counts(train),
            "val": counts(val),
            "test": counts(test),
        },
        "total_per_split": {"train": len(train), "val": len(val), "test": len(test)},
    }
    (out_dir / "dataset_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
