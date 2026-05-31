import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
PATHOLOGY_CLASSES = {"aca_bd", "aca_md", "aca_pd", "scc_bd", "scc_md", "scc_pd"}
NORMAL_CLASSES = {"nor"}


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def split_items(items: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items, val_items


def copy_group(files: List[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, out_dir / src.name)


def gather_sources(images_root: Path) -> Dict[str, List[Path]]:
    out = {"normal": [], "pathological": []}

    for class_dir in sorted(images_root.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name.lower()
        imgs = list_images(class_dir)

        if class_name in NORMAL_CLASSES:
            out["normal"].extend(imgs)
        elif class_name in PATHOLOGY_CLASSES:
            out["pathological"].extend(imgs)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LungHist700 binary train/val split")
    parser.add_argument("--source", default="data/raw/LungHist700/data/images")
    parser.add_argument("--output", default="data/lunghist700_binary")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summary", default="reports/pd1/dataset_split_summary.json")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    if output.exists():
        shutil.rmtree(output)

    grouped = gather_sources(source)
    summary: Dict[str, Dict[str, int]] = {}

    for label, files in grouped.items():
        train_files, val_files = split_items(files, args.val_ratio, args.seed)
        copy_group(train_files, output / "train" / label)
        copy_group(val_files, output / "val" / label)
        summary[label] = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files),
        }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote split summary to {summary_path}")


if __name__ == "__main__":
    main()
