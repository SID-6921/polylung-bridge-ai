import argparse
import json
import random
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def normalize_stem(stem: str) -> str:
    stem = re.sub(r"^\._", "", stem)
    stem = re.sub(r"^\d+-\d+_", "", stem)
    stem = re.sub(r"_No\d+(?:_\d+)?$", "", stem)
    return stem


def label_from_stem(stem: str) -> Optional[str]:
    normalized = normalize_stem(stem)
    if normalized.startswith("PS_Isopropyl_Alcohol_Suspension"):
        return "PS"
    if normalized.startswith("Polyester_Fibres") or normalized.startswith("Polyester_Fibre_On_Carbon_Tape"):
        return "Polyester"
    if normalized.startswith("Dove_men_acrylates_copolymer_microbeads_new"):
        return "Acrylates"
    if normalized.startswith("PAN"):
        return "PAN"
    if normalized.startswith("PE"):
        return "PE"
    if normalized.startswith("PP"):
        return "PP"
    return None


def split_items(items: List[str], val_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio))
    val_count = min(val_count, len(shuffled))
    return shuffled[val_count:], shuffled[:val_count]


def copy_member(archive: zipfile.ZipFile, member: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(member) as src, out_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def iter_image_members(archive: zipfile.ZipFile) -> Iterable[str]:
    for member in archive.namelist():
        if member.startswith("__MACOSX/"):
            continue
        if not member.lower().endswith(tuple(IMAGE_EXTS)):
            continue
        if "original_image_" not in member:
            continue
        yield member


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardize Microplastics_SEM into train/val ImageFolder splits")
    parser.add_argument("--archive", default=".tmp/dataset3.zip", help="Path to dataset3.zip")
    parser.add_argument("--output", default="data/microplastics_sem_standardized", help="Output ImageFolder root")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--summary",
        default="evidence/public/pd1/microplastics_sem_polymer_summary.json",
        help="JSON summary output",
    )
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    output = Path(args.output)
    if output.exists():
        shutil.rmtree(output)

    grouped: Dict[str, List[str]] = {}
    with zipfile.ZipFile(archive_path) as zf:
        for member in iter_image_members(zf):
            label = label_from_stem(Path(member).stem)
            if label is None:
                continue
            grouped.setdefault(label, []).append(member)

        if not grouped:
            raise ValueError("No confirmed polymer labels found in the archive")

        summary: Dict[str, Dict[str, int]] = {}
        for label, members in sorted(grouped.items()):
            train_members, val_members = split_items(members, args.val_ratio, args.seed)
            for member in train_members:
                copy_member(zf, member, output / "train" / label / Path(member).name)
            for member in val_members:
                copy_member(zf, member, output / "val" / label / Path(member).name)
            summary[label] = {
                "total": len(members),
                "train": len(train_members),
                "val": len(val_members),
            }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"archive": str(archive_path), "classes": summary}, indent=2), encoding="utf-8")
    print(f"Wrote standardized dataset summary to {summary_path}")


if __name__ == "__main__":
    main()