import os
import sys
# Ensure project root is in sys.path so we can import from models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import cv2
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.clip_wrapper import CLIPWrapper


def reshape_transform(tensor, height=7, width=7):
    """Reshape ViT activations from 3D (batch, seq, dim) → 4D (batch, channels, h, w)"""
    # Remove cls token
    result = tensor[:, 1:, :]
    # Reshape to (batch, height, width, channels)
    result = result.reshape(result.shape[0], height, width, result.shape[-1])
    # Channels first: (batch, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="CLIP Grad-CAM visualization example")
    parser.add_argument("--image-path", required=True, help="Path to input image")
    parser.add_argument("--labels", nargs="+", required=True, 
                        help='Text labels, e.g. "a cat" "a dog"')
    parser.add_argument("--method", default="gradcam++", 
                        choices=["gradcam", "gradcam++", "scorecam", "eigencam"],
                        help="CAM method to use")
    parser.add_argument("--device", default="cuda", 
                        help="Device: cuda, cuda:0, cpu")
    parser.add_argument("--output-dir", default=".", 
                        help="Directory to save output images (default: current dir)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if "cuda" in args.device and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_name = "openai/clip-vit-base-patch32"
    print(f"Loading CLIP model: {model_name} (may download weights, please wait...)")

    wrapper = CLIPWrapper(model_name, args.labels, device=device).to(device).eval()

    # Load and preprocess image
    processor = wrapper.get_processor()
    img = Image.open(args.image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # (1,3,H,W)

    # Choose target layer
    target_layers = wrapper.find_target_layers()
    if target_layers is None:
        print("Warning: Could not auto-find target layer. Using fallback.")
        if hasattr(wrapper.clip, "visual") and hasattr(wrapper.clip.visual, "ln_post"):
            target_layers = [wrapper.clip.visual.ln_post]
            print("Fallback target layer: clip.visual.ln_post")
        else:
            raise RuntimeError("Cannot find suitable target layer. Please specify manually.")

    # Select CAM method
    methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM,
    }
    cam_cls = methods[args.method]

    cam = cam_cls(
        model=wrapper,
        target_layers=target_layers,
        reshape_transform=reshape_transform
    )
    cam.batch_size = 32  # for ScoreCAM memory efficiency

    # Prepare original image for visualization (resized to match model input size)
    H, W = pixel_values.shape[-2], pixel_values.shape[-1]
    img_rgb = np.array(img.resize((W, H))) / 255.0  # float [0,1]

    # Generate heatmap for EACH label
    for idx, label in enumerate(args.labels):
        targets = [ClassifierOutputTarget(idx)]
        print(f"\nGenerating heatmap for: '{label}' (index {idx})")

        grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # take first (and only) item in batch

        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

        # Safe filename
        safe_label = label.replace(" ", "_").replace("/", "_").replace(":", "_")
        out_filename = f"{args.method}_{safe_label}.jpg"
        out_path = os.path.join(args.output_dir, out_filename)

        cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()