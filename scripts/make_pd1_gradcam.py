from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import swin_t


IMAGE_PATH = Path(
    "data/lunghist700_binary/val/pathological/aca_bd_20x_13.jpg"
)

CHECKPOINT_PATH = Path(
    "evidence/public/pd1/pd1_swin_l40s_model.pt"
)

OUTPUT_PATH = Path(
    "evidence/public/pd1/gradcam_pathological_panel.png"
)

# SWIN TRANSFORMER RESHAPE
def reshape_transform(tensor):
    """
    Convert Swin Transformer activation output into the format
    expected by Grad-CAM: [batch, channels, height, width].
    """

    if tensor.ndim != 4:
        raise ValueError(
            f"Expected a 4D activation tensor, got shape {tensor.shape}"
        )

    # torchvision Swin commonly returns:
    # [batch, height, width, channels]
    tensor = tensor.permute(0, 3, 1, 2)

    return tensor


# LOAD CHECKPOINT
def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Handle several common checkpoint structures
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove "module." prefix from DataParallel checkpoints
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    return model


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found: {IMAGE_PATH}"
        )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            "\nTrained model checkpoint was not found:\n"
            f"{CHECKPOINT_PATH}\n\n"
            "Grad-CAM requires the trained model weights. "
            "Check the checkpoint filename and path."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Image: {IMAGE_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    # Swin-T model
    model = swin_t(weights=None)

    # Binary classification head
    in_features = model.head.in_features
    model.head = torch.nn.Linear(
        in_features,
        2,
    )

    model = load_checkpoint(
        model,
        CHECKPOINT_PATH,
        device,
    )

    model = model.to(device)
    model.eval()

    # Input image
    pil_image = Image.open(
        IMAGE_PATH
    ).convert("RGB")

    rgb_image = np.asarray(
        pil_image.resize((224, 224))
    ).astype(np.float32) / 255.0

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    input_tensor = preprocess(
        pil_image
    ).unsqueeze(0).to(device)

    # Last Swin normalization layer
    target_layers = [
        model.features[-1][-1].norm2
    ]

    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        reshape_transform=reshape_transform,
    )

    # target=None means use the model's predicted class
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=None,
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
        image_weight=0.7,
    )

    # Three images with no text, margins, or axes
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12, 4),
        gridspec_kw={
            "wspace": 0,
            "hspace": 0,
        },
    )

    axes[0].imshow(rgb_image)
    axes[1].imshow(
        grayscale_cam,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[2].imshow(visualization)

    for axis in axes:
        axis.axis("off")

    plt.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
        wspace=0,
        hspace=0,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.savefig(
        OUTPUT_PATH,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)

    print(
        f"Saved Grad-CAM panel to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
