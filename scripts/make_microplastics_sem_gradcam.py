from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import swin_t


DEFAULT_IMAGE_PATH = Path("data/microplastics_sem_split/val/PS")
DEFAULT_CHECKPOINT_PATH = Path("evidence/public/pd1/microplastics_sem_swin_t.pt")
DEFAULT_OUTPUT_PATH = Path("evidence/public/pd1/microplastics_sem_gradcam_panel.png")


def load_checkpoint(model, checkpoint_path, device):
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
            class_names = checkpoint.get("class_names", []) if isinstance(checkpoint, dict) else []
    else:
        state_dict = checkpoint
        class_names = []

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)
    return model, class_names


def resolve_image_path(path: Path) -> Path:
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(
            [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}]
        )
        if candidates:
            return candidates[0]
    raise FileNotFoundError(f"No image found at or under {path}")


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    denom = cam.max() if cam.max() > 0 else 1.0
    return cam / denom


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create a Grad-CAM-style explanation for the Microplastics_SEM Swin-T model")
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

    activations = {}
    gradients = {}

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    target_layer = model.features[-1][-1].norm2
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        logits = model(input_tensor)
        pred_idx = int(torch.argmax(logits, dim=1).item())
        score = logits[0, pred_idx]
        model.zero_grad(set_to_none=True)
        score.backward()

        activation = activations["value"]
        grad = gradients["value"]
        if activation.ndim != 4:
            raise ValueError(f"Expected 4D activation map, got {tuple(activation.shape)}")

        if grad.ndim != 4:
            raise ValueError(f"Expected 4D gradient map, got {tuple(grad.shape)}")

        weights = grad.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activation).sum(dim=1)).squeeze(0)
        cam = cam.cpu().numpy()
        cam = normalize_cam(cam)
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224))
        cam_np = np.asarray(cam_img).astype(np.float32) / 255.0

        overlay = np.clip(0.65 * rgb_image + 0.35 * plt.get_cmap("jet")(cam_np)[..., :3], 0, 1)

        class_name = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4), gridspec_kw={"wspace": 0.02, "hspace": 0.0})
        axes[0].imshow(rgb_image)
        axes[0].set_title(f"Input\n{image_path.name}")
        axes[1].imshow(cam_np, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(f"Heatmap\nPred: {class_name}")
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")

        for axis in axes:
            axis.axis("off")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved Grad-CAM panel to: {output_path}")
    finally:
        forward_handle.remove()
        backward_handle.remove()


if __name__ == "__main__":
    main()