#!/usr/bin/env python3
# 例子脚本：使用 CLIPWrapper + pytorch-grad-cam 为每个文本标签生成热力图并保存
# 已修正：移除 use_cuda、處理 heatmap 與原圖尺寸不匹配問題

import os
import sys
# 確保項目根目錄在 sys.path 中，這樣可以導入 models 包
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
    """將 ViT 的 3D activations 轉為 4D (batch, channels, height, width)"""
    # 移除 cls token
    result = tensor[:, 1:, :]
    # reshape 到 (batch, height, width, channels)
    result = result.reshape(result.shape[0], height, width, result.shape[-1])
    # channels first: (batch, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for CLIP with text labels")
    parser.add_argument("--image-path", required=True, help="輸入圖像路徑")
    parser.add_argument("--labels", nargs="+", required=True, help='文本標籤，例如 "a cat" "a dog"')
    parser.add_argument("--method", default="gradcam", 
                        choices=["gradcam", "scorecam", "eigencam", "gradcam++"], 
                        help="CAM 方法")
    parser.add_argument("--device", default="cuda", help="設備：cuda 或 cpu 或 cuda:0")
    parser.add_argument("--output-dir", default="outputs", help="輸出目錄")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", 
                        help="CLIP 模型名稱或路徑")
    return parser.parse_args()


def main():
    args = parse_args()

    # 處理 device
    requested = args.device
    device = torch.device(requested if "cuda" in requested and torch.cuda.is_available() else "cpu")
    print(f"使用裝置：{device}")

    os.makedirs(args.output_dir, exist_ok=True)

    print("加載 CLIP 包裝器並生成 text embeddings（labels）：", args.labels)

    # 初始化 wrapper 並移到指定裝置
    wrapper = CLIPWrapper(args.model_name, args.labels, device=device)
    wrapper = wrapper.to(device).eval()

    # 讀取原始圖像
    img = Image.open(args.image_path).convert("RGB")
    original_size = img.size  # (width, height)

    # 預處理輸入給 CLIP（會 resize 到 224x224）
    processor = wrapper.get_processor()
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # (1,3,224,224)

    # 用於疊加的原始圖像 (H, W, 3) float 0-1
    img_rgb = np.array(img).astype(np.float32) / 255.0
    print(f"原始圖像尺寸 (H,W): {img_rgb.shape[:2]}")

    # 選擇 target layer
    target_layers = wrapper.find_target_layers()
    if target_layers is None:
        print("警告：未能自動找到 target layer，使用 fallback")
        if hasattr(wrapper.clip, "visual") and hasattr(wrapper.clip.visual, "ln_post"):
            target_layers = [wrapper.clip.visual.ln_post]
            print("使用 fallback target layer: clip.visual.ln_post")
        else:
            raise RuntimeError("無法定位 target layer，請手動指定")

    # 選擇 CAM 方法
    methods = {
        "gradcam": GradCAM,
        "scorecam": ScoreCAM,
        "eigencam": EigenCAM,
        "gradcam++": GradCAMPlusPlus
    }
    cam_cls = methods.get(args.method.lower())
    if cam_cls is None:
        raise ValueError(f"不支援的方法: {args.method}")

    print(f"使用 CAM 方法: {args.method} | target_layers: {target_layers}")

    # 建立 CAM 物件
    cam = cam_cls(
        model=wrapper,
        target_layers=target_layers,
        reshape_transform=reshape_transform  # 對 CLIP ViT 必要
    )

    if args.method.lower() == "scorecam":
        cam.batch_size = 32

    # 為每個 label 生成熱力圖
    for idx, label in enumerate(args.labels):
        targets = [ClassifierOutputTarget(idx)]
        print(f"\n生成熱力圖 for: {label} (index {idx})")

        # 產生 CAM (通常得到 (1, 224, 224) 或 list)
        grayscale_cam = cam(input_tensor=pixel_values, targets=targets)

        # 處理回傳格式
        if isinstance(grayscale_cam, (list, tuple)):
            grayscale_cam = grayscale_cam[0]
        grayscale_cam = np.array(grayscale_cam)

        if grayscale_cam.ndim == 3:
            cam_map = grayscale_cam[0]          # (224, 224)
        elif grayscale_cam.ndim == 4:
            cam_map = grayscale_cam[0, 0]       # (224, 224)
        else:
            raise RuntimeError(f"不支援的 CAM shape: {grayscale_cam.shape}")

        print(f"CAM map 原始形狀: {cam_map.shape}")

        # 上採樣到原始圖像尺寸
        original_h, original_w = img_rgb.shape[0], img_rgb.shape[1]
        cam_map_resized = cv2.resize(
            cam_map,
            (original_w, original_h),           # cv2.resize 用 (width, height)
            interpolation=cv2.INTER_LINEAR
        )
        cam_map_resized = np.clip(cam_map_resized, 0.0, 1.0)

        # 疊加到原始圖像
        visualization = show_cam_on_image(img_rgb, cam_map_resized, use_rgb=True)

        # 保存（轉為 BGR + uint8）
        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(args.output_dir, f"{args.method}_{safe_label}.jpg")
        
        vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, (vis_bgr * 255).astype(np.uint8))
        print(f"已保存：{out_path}")


if __name__ == "__main__":
    main()