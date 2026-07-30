from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import swin_t


DEFAULT_IMAGE_PATH = Path("data/microplastics_sem_split/val/PS")
DEFAULT_CHECKPOINT_PATH = Path("evidence/public/pd1/microplastics_sem_swin_t.pt")
DEFAULT_OUTPUT_PATH = Path("evidence/public/pd1/microplastics_sem_explainability_panel.png")


def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            class_names = checkpoint.get("class_names", [])
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            class_names = checkpoint.get("class_names", [])
        else:
            state_dict = checkpoint
            class_names = checkpoint.get("class_names", [])
    else:
        state_dict = checkpoint
        class_names = []

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned_state_dict[key] = value.float() if torch.is_tensor(value) and torch.is_floating_point(value) else value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model, class_names


def resolve_image_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(
            [
                p
                for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
            ]
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No image found at or under {path}")


def normalize_map(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values, 0)
    values = values - values.min()
    if values.max() > 0:
        values = values / values.max()
    return values


def predict_class_probabilities(model, input_tensor):
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())
    pred_prob = float(probs[0, pred_idx].item())
    return pred_idx, pred_prob


def build_occlusion_map(model, input_tensor, pred_idx, patch_size=32, stride=16):
    _, _, height, width = input_tensor.shape

    with torch.no_grad():
        base_prob = float(torch.softmax(model(input_tensor), dim=1)[0, pred_idx].item())
        occlusion = np.zeros((height, width), dtype=np.float32)
        counts = np.zeros((height, width), dtype=np.float32)
        occlusion_value = input_tensor.mean(dim=(2, 3), keepdim=True)

        for top in range(0, height, stride):
            bottom = min(top + patch_size, height)
            for left in range(0, width, stride):
                right = min(left + patch_size, width)
                perturbed = input_tensor.clone()
                perturbed[:, :, top:bottom, left:right] = occlusion_value
                prob = float(torch.softmax(model(perturbed), dim=1)[0, pred_idx].item())
                drop = max(0.0, base_prob - prob)
                occlusion[top:bottom, left:right] += drop
                counts[top:bottom, left:right] += 1.0

    counts[counts == 0] = 1.0
    saliency = normalize_map(occlusion / counts)
    return saliency, base_prob


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create an occlusion-based explainability panel for the Microplastics_SEM Swin-T model")
    parser.add_argument("--image-path", default=str(DEFAULT_IMAGE_PATH), help="Image file or directory")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    image_path = resolve_image_path(Path(args.image_path))
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = swin_t(weights=None)
    model.head = torch.nn.Linear(model.head.in_features, 6)
    model, class_names = load_checkpoint(model, checkpoint_path, device)
    model = model.to(device)
    model.eval()

    pil_image = Image.open(image_path).convert("L")
    rgb_image = np.asarray(pil_image.resize((224, 224)).convert("RGB")).astype(np.float32) / 255.0

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(pil_image).unsqueeze(0).to(device)

    pred_idx, pred_prob = predict_class_probabilities(model, input_tensor)
    saliency, _ = build_occlusion_map(model, input_tensor, pred_idx)
    saliency_img = Image.fromarray((saliency * 255).astype(np.uint8)).resize((224, 224), resample=Image.Resampling.BILINEAR)
    saliency_np = np.asarray(saliency_img).astype(np.float32) / 255.0

    overlay = np.clip(0.72 * rgb_image + 0.28 * plt.get_cmap("magma")(saliency_np)[..., :3], 0, 1)
    class_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.02, "hspace": 0.0})
    axes[0].imshow(rgb_image)
    axes[0].set_title(f"Input\n{image_path.name}")
    axes[1].imshow(saliency_np, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title(f"Occlusion saliency\nPred: {class_name}")
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay\nP={pred_prob:.3f}")

    for axis in axes:
        axis.axis("off")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Saved explainability panel to: {output_path}")


if __name__ == "__main__":
    main()