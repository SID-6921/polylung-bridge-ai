from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

NORMAL_DIR = Path("data/lunghist700_binary/val/normal")
PATHOLOGICAL_DIR = Path("data/lunghist700_binary/val/pathological")
OUTPUT = Path("evidence/public/pd1/representative_histology_images.png")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def get_images(folder: Path, n: int = 3):
    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    if len(files) < n:
        raise ValueError(
            f"Not enough images in {folder}. "
            f"Found {len(files)}, need at least {n}."
        )

    return files[:n]


def main():
    if not NORMAL_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {NORMAL_DIR}")

    if not PATHOLOGICAL_DIR.exists():
        raise FileNotFoundError(f"Missing folder: {PATHOLOGICAL_DIR}")

    normal_files = get_images(NORMAL_DIR, 3)
    pathological_files = get_images(PATHOLOGICAL_DIR, 3)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6))

    for i, image_path in enumerate(normal_files):
        image = Image.open(image_path).convert("RGB")
        axes[0, i].imshow(image)
        axes[0, i].set_title(f"Normal {i + 1}")
        axes[0, i].axis("off")

    for i, image_path in enumerate(pathological_files):
        image = Image.open(image_path).convert("RGB")
        axes[1, i].imshow(image)
        axes[1, i].set_title(f"Pathological {i + 1}")
        axes[1, i].axis("off")

    fig.suptitle(
        "Representative LungHist700 H&E Images",
        fontsize=16
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(
        OUTPUT,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Saved figure to: {OUTPUT}")
    print("Normal images:")
    for path in normal_files:
        print(f"  {path}")

    print("Pathological images:")
    for path in pathological_files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
