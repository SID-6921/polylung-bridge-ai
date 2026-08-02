import os, subprocess, glob, csv
from PIL import Image

BASE = os.path.expanduser("~/polylung-bridge-ai/data/microplastics_datasets/unicamp_estuary_sem_ftir")
OUT = os.path.expanduser("~/polylung-bridge-ai/data/unicamp_labeled_images")
os.makedirs(OUT, exist_ok=True)
TMP = "/tmp/pdfimg_extract"
os.makedirs(TMP, exist_ok=True)

# Unique 14 samples (RC_MP_56 dedup'd - two spreadsheet rows collide on this Sample_ID,
# only one PDF/CSV exists, both rows are labeled PE anyway so class is unambiguous)
samples = [
    ("RC_MP_01", "PA"),
    ("RC_MP_02", "PP"),
    ("RC_MP_04", "PE"),
    ("RC_MP_09", "PVC"),
    ("RC_MP_13", "PE"),
    ("RC_MP_18", "PS"),
    ("RC_MP_25", "PP"),
    ("RC_MP_38", "PE"),
    ("RC_MP_44", "PE"),
    ("RC_MP_46", "PS"),
    ("RC_MP_51", "PE"),
    ("RC_MP_56", "PE"),
    ("RC_MP_59", "PP"),
    ("RC_MP_60", "PS"),
]

manifest_rows = []
choice_notes = []

for sid, label in samples:
    pdfs = sorted(glob.glob(os.path.join(BASE, "SEM-EDS", "SEM-EDS", f"{sid}*.pdf")))
    ftir_csv = os.path.join(BASE, "FTIR", "FTIR", f"{sid}.csv")
    assert os.path.exists(ftir_csv), f"missing ftir {sid}"
    assert len(pdfs) >= 1, f"no pdf for {sid}"

    # extract images from each candidate pdf, pick the largest-area image (fuller frame,
    # not a zoomed-in crop) among all extracted images across all split files
    best_img = None
    best_area = -1
    best_src = None
    for pdf in pdfs:
        prefix = os.path.join(TMP, os.path.basename(pdf).replace(".pdf", ""))
        subprocess.run(["pdfimages", "-png", pdf, prefix], check=True)
        for img_path in sorted(glob.glob(prefix + "-*.png")):
            with Image.open(img_path) as im:
                area = im.width * im.height
                # prefer grayscale SEM micrographs (mode L) roughly matching known ~688x516 dims
                if area > best_area:
                    best_area = area
                    best_img = img_path
                    best_src = os.path.basename(pdf)

    assert best_img is not None, f"no image extracted for {sid}"
    out_name = f"{sid}_{label}.png"
    out_path = os.path.join(OUT, out_name)
    with Image.open(best_img) as im:
        im.save(out_path)

    choice_notes.append(f"{sid}: chose image from {best_src} (largest extracted image, {best_area}px area) among {len(pdfs)} candidate PDF(s)")
    manifest_rows.append({
        "sample_id": sid,
        "polymer_label": label,
        "image_path": os.path.relpath(out_path, os.path.expanduser("~/polylung-bridge-ai")),
        "ftir_csv_path": os.path.relpath(ftir_csv, os.path.expanduser("~/polylung-bridge-ai")),
    })

manifest_dir = os.path.expanduser("~/polylung-bridge-ai/evidence/public/unicamp_classification")
os.makedirs(manifest_dir, exist_ok=True)
with open(os.path.join(manifest_dir, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["sample_id", "polymer_label", "image_path", "ftir_csv_path"])
    w.writeheader()
    w.writerows(manifest_rows)

print("Image selection notes:")
for n in choice_notes:
    print(" ", n)

print("\nManifest written:", os.path.join(manifest_dir, "manifest.csv"))
from collections import Counter
print("Class breakdown (n=%d):" % len(manifest_rows), Counter([r["polymer_label"] for r in manifest_rows]))
