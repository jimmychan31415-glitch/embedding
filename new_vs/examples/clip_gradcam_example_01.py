import os
import sys
# 确保项目根目录在 sys.path 中，这样可以导入 models 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def reshape_transform(tensor, height=7, width=7):
    """将 ViT 的 3D activations 转为 4D (batch, channels, height, width)"""
    # 移除 cls token
    result = tensor[:, 1:, :]
    # reshape 到 (batch, height, width, channels)
    result = result.reshape(result.shape[0], height, width, result.shape[-1])
    # channels first: (batch, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    return result


import argparse
import os
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
# GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.clip_wrapper import CLIPWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, help="输入图像路径")
    parser.add_argument("--labels", nargs="+", required=True, help='文本标签，例如 "a cat" "a dog"')
    parser.add_argument("--method", default="gradcam", choices=["gradcam", "scorecam", "eigencam"], help="CAM 方法")
    parser.add_argument("--device", default="cuda", help="设备：cuda 或 cpu 或 cuda:0")
    parser.add_argument("--output", default="cam_out.jpg", help="输出文件名")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu")
    model_name = "openai/clip-vit-base-patch32"

    print("加载 CLIP 模型：", model_name, "，这会下载/加载权重，请耐心等待（可能较慢）。")
    wrapper = CLIPWrapper(model_name, args.labels, device=device).to(device).eval()

    # 读取并预处理图像（使用 wrapper.processor）
    processor = wrapper.get_processor()
    from PIL import Image
    img = Image.open(args.image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # (1,3,H,W)

    # 选择 target layer
    target_layers = wrapper.find_target_layers()
    if target_layers is None:
        print("警告：未能自动找到 target layer。请手动修改脚本选择合适的层（参考 wrapper.clip 的模块结构）。")
        if hasattr(wrapper.clip, "visual") and hasattr(wrapper.clip.visual, "ln_post"):
            target_layers = [wrapper.clip.visual.ln_post]
            print("使用 fallback target layer: clip.visual.ln_post")
        else:
            raise RuntimeError("无法自动定位 target layer，请手动指定。")

    methods = {"gradcam": GradCAM, "scorecam": ScoreCAM, "eigencam": EigenCAM,
    "gradcam++": GradCAMPlusPlus}
    cam_cls = methods[args.method]

    use_cuda = device.type == "cuda"
    cam = cam_cls(model=wrapper, target_layers=target_layers, reshape_transform=reshape_transform)
    cam.batch_size = 32


    # 为每个 label 分别生成热力图
    for idx, label in enumerate(args.labels):
        targets = [ClassifierOutputTarget(idx)]
        print(f"\n生成热力图 for: {label} (index {idx})")
        grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = f"{args.method}_{safe_label}.jpg"
        cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"保存结果： {out_path}")
    


if __name__ == "__main__":
    main()