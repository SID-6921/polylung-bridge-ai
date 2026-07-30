import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def list_images(folder: Path) -> List[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]


def split_items(items: List[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled))
    val_items = shuffled[:val_count]
    train_items = shuffled[val_count:]
    return train_items, val_items


def copy_group(files: List[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, out_dir / src.name)


def collect_class_dirs(root: Path) -> List[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir() and list_images(p)]


def prepare_existing_splits(source: Path, output: Path) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    for split_name in ("train", "val"):
        split_root = source / split_name
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_root}")

        for class_dir in collect_class_dirs(split_root):
            files = list_images(class_dir)
            copy_group(files, output / split_name / class_dir.name)
            class_summary = summary.setdefault(class_dir.name, {"train": 0, "val": 0, "total": 0})
            class_summary[split_name] = len(files)
            class_summary["total"] += len(files)

    return summary


def prepare_split_from_class_dirs(source: Path, output: Path, val_ratio: float, seed: int) -> Dict[str, Dict[str, int]]:
    summary: Dict[str, Dict[str, int]] = {}
    class_dirs = collect_class_dirs(source)
    if not class_dirs:
        raise ValueError(
            "No class folders with images were found. Point --source to a folder that contains class subdirectories."
        )

    for class_dir in class_dirs:
        files = list_images(class_dir)
        train_files, val_files = split_items(files, val_ratio, seed)
        copy_group(train_files, output / "train" / class_dir.name)
        copy_group(val_files, output / "val" / class_dir.name)
        summary[class_dir.name] = {
            "total": len(files),
            "train": len(train_files),
            "val": len(val_files),
        }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an ImageFolder-style train/val split for SEM or similar datasets")
    parser.add_argument("--source", default="data/raw/Microplastics_SEM", help="Dataset root")
    parser.add_argument("--output", default="data/microplastics_sem_split", help="Output train/val root")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary",
        default="evidence/public/pd1/microplastics_sem_split_summary.json",
        help="JSON summary output",
    )
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.exists():
        raise FileNotFoundError(f"Source folder not found: {source}")

    if output.exists():
        shutil.rmtree(output)

    if (source / "train").is_dir() and (source / "val").is_dir():
        summary = prepare_existing_splits(source, output)
        layout = "existing_splits"
    else:
        summary = prepare_split_from_class_dirs(source, output, args.val_ratio, args.seed)
        layout = "split_from_class_dirs"

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"layout": layout, "classes": summary}, indent=2), encoding="utf-8")
    print(f"Wrote split summary to {summary_path}")


if __name__ == "__main__":
    main()