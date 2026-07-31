"""
Product/session-GROUPED split for the Toronto SEM shape (fibre/bead/fragment)
classifier. Fixes the leakage documented in
evidence/public/sem_shape/data_leakage_notes.md: the original stratified
per-image split (sem_shape_prepare.py) let images from the same source
product/imaging-session appear in train, val AND test simultaneously, so
100% accuracy mostly reflected "which known session is this" rather than
genuine shape generalization.

This script assigns whole GROUPS (source products/imaging sessions,
identified via filename prefix) to exactly one split. No group ever appears
in more than one split.

Groups identified by inspecting actual filenames in dataset3/dataset3:
  - bead:     1 group total  -> "Dove_men_acrylates_copolymer_microbeads"
              Only 1 product exists for this class. A true held-out
              evaluation of bead is therefore impossible without violating
              group integrity -- ALL bead images go to train, none to
              val/test. This is reported honestly (see summary JSON) rather
              than fudged. Because bead never appears in val/test, the
              held-out evaluation of this classifier is effectively a
              2-class (fibre vs fragment) evaluation, which is the more
              honest way to read the val/test numbers here.
  - fragment: 4 groups -> Foam_No, Particle_No, Particle_2_No (fragment_1),
              PS_Isopropyl_Alcohol_Suspension (fragment_2)
  - fibre:    4 groups (by material) -> polyester (fibre_1 "polyester_*" +
              fibre_2 "p<N>polyester_fibre.png"), PAN, PE, PP (fibre_2,
              various kV settings / specimens of the same material)

Manual group->split assignment (whole groups, chosen to land close to
70/15/15 while keeping every split non-trivial):
  fibre:    train={polyester(48),PP(10)}=58  val={PE(12)}=12  test={PAN(20)}=20
  fragment: train={PS_Isopropyl(41),Particle_No(28)}=69  val={Particle_2_No(15)}=15  test={Foam_No(12)}=12
  bead:     train={Dove(51)}=51  val={}=0  test={}=0
"""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

IMG_EXTS = {".png", ".jpg", ".jpeg"}

SHAPE_DIRS = {
    "fibre": ["fibre/fibre_1/image", "fibre/fibre_2/image"],
    "bead": ["bead/bead/image"],
    "fragment": ["fragment/fragment_1/image", "fragment/fragment_2/image"],
}


def group_of(label: str, filename: str) -> str:
    if label == "bead":
        return "Dove_men_acrylates_copolymer_microbeads"
    if label == "fragment":
        for prefix in ("Foam_No", "Particle_2_No", "Particle_No", "PS_Isopropyl_Alcohol_Suspension"):
            if filename.startswith(prefix):
                return prefix
        raise ValueError(f"unrecognized fragment filename: {filename}")
    if label == "fibre":
        if filename.startswith("polyester_") or re.match(r"^p\d+polyester_fibre", filename):
            return "polyester"
        m = re.match(r"^(PAN|PE|PP)_", filename)
        if m:
            return m.group(1)
        raise ValueError(f"unrecognized fibre filename: {filename}")
    raise ValueError(f"unrecognized label: {label}")


# Manual whole-group -> split assignment (see module docstring for rationale)
GROUP_SPLIT = {
    ("fibre", "polyester"): "train",
    ("fibre", "PP"): "train",
    ("fibre", "PE"): "val",
    ("fibre", "PAN"): "test",
    ("fragment", "PS_Isopropyl_Alcohol_Suspension"): "train",
    ("fragment", "Particle_No"): "train",
    ("fragment", "Particle_2_No"): "val",
    ("fragment", "Foam_No"): "test",
    ("bead", "Dove_men_acrylates_copolymer_microbeads"): "train",
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to dataset3/dataset3")
    ap.add_argument("--out-dir", required=True)
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
                grp = group_of(label, p.name)
                split = GROUP_SPLIT[(label, grp)]
                items.append({"path": str(p), "label": label, "group": grp, "split": split, "md5": h})

    splits = {"train": [], "val": [], "test": []}
    for it in items:
        splits[it["split"]].append(it)

    def write_csv(name, rows):
        with open(out_dir / name, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["path", "label"])
            for r in rows:
                w.writerow([r["path"], r["label"]])

    for name in ("train", "val", "test"):
        write_csv(f"{name}.csv", splits[name])

    def counts(rows):
        c = {}
        for r in rows:
            c[r["label"]] = c.get(r["label"], 0) + 1
        return c

    def groups_in(rows):
        return sorted({r["group"] for r in rows})

    summary = {
        "task": "SEM particle-morphology (SHAPE) classification -- PRODUCT/SESSION-GROUPED split. "
                "Fixes leakage in the original stratified split; see evidence/public/sem_shape/data_leakage_notes.md.",
        "split_method": "manual whole-group assignment; no source product/imaging-session appears in more than one split",
        "source_dirs_used": SHAPE_DIRS,
        "group_split_assignment": {f"{k[0]}:{k[1]}": v for k, v in GROUP_SPLIT.items()},
        "groups_per_class_per_split": {
            "train": {c: sorted({r["group"] for r in splits["train"] if r["label"] == c}) for c in SHAPE_DIRS},
            "val": {c: sorted({r["group"] for r in splits["val"] if r["label"] == c}) for c in SHAPE_DIRS},
            "test": {c: sorted({r["group"] for r in splits["test"] if r["label"] == c}) for c in SHAPE_DIRS},
        },
        "bead_holdout_note": (
            "bead has exactly 1 source product (Dove_men_acrylates_copolymer_microbeads) in the entire "
            "dataset. A grouped/held-out split cannot put any bead images in val or test without "
            "violating group integrity (reusing the same session in train and eval, i.e. reintroducing "
            "the original leakage). All 51 bead images are therefore in train only; bead CANNOT be "
            "evaluated on held-out data with this dataset as-is. This is itself evidence that more "
            "source diversity is needed before shape classification (specifically for beads) can be "
            "validated. Because of this, val/test in this split are effectively a 2-class (fibre vs "
            "fragment) held-out evaluation -- both of those classes have >=1 group held out per split -- "
            "and should be read that way."
        ),
        "duplicate_files_removed": dup_count,
        "total_after_dedup": len(items),
        "raw_file_counts_before_dedup": per_class_raw,
        "counts_per_class_per_split": {
            "train": counts(splits["train"]),
            "val": counts(splits["val"]),
            "test": counts(splits["test"]),
        },
        "total_per_split": {k: len(v) for k, v in splits.items()},
        "actual_split_ratio": {
            k: round(len(v) / len(items), 4) for k, v in splits.items()
        },
    }
    (out_dir / "dataset_split_summary_grouped.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
