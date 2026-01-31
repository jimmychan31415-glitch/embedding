#!/usr/bin/env python3
# 例子脚本：使用 CLIPWrapper + pytorch-grad-cam 为每个文本标签生成热力图并保存
# 参考：@jacobgil/pytorch-grad-cam (https://github.com/jacobgil/pytorch-grad-cam)

import os
import sys
# 确保项目根目录在 sys.path 中，这样可以导入 models 包
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def reshape_transform(tensor, height=7, width=7):
    """将 ViT 的 3D activations 转为 4D (batch, channels, height, width)
    注意：height/width 需要与模型的 patch grid 大小匹配（示例值为 7x7，按需修改）
    """
    # 移除 cls token
    result = tensor[:, 1:, :]
    # reshape 到 (batch, height, width, channels)
    result = result.reshape(result.shape[0], height, width, result.shape[-1])
    # channels first: (batch, channels, height, width)
    result = result.permute(0, 3, 1, 2)
    return result


import argparse
import cv2
import numpy as np
import torch

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.clip_wrapper import CLIPWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, help="输入图像路径")
    parser.add_argument("--labels", nargs="+", required=True, help='文本标签，例如 "a cat" "a dog"')
    parser.add_argument("--method", default="gradcam", choices=["gradcam", "scorecam", "eigencam", "gradcam++"], help="CAM 方法")
    parser.add_argument("--device", default="cuda", help="设备：cuda 或 cpu 或 cuda:0")
    parser.add_argument("--output-dir", default="outputs", help="输出目录")
    parser.add_argument("--model-name", default="openai/clip-vit-base-patch32", help="CLIP 模型名称或路径")
    return parser.parse_args()


def try_create_cam(cam_cls, **kwargs):
    """
    尝试用若干不同参数签名创建 cam 对象以兼容不同版本的 pytorch-grad-cam。
    返回创建的 cam 对象或抛出最后的异常。
    """
    # 构造候选参数列表（按优先级）
    candidates = []
    # 优先尝试常见的带 reshape_transform 的签名（如果提供）
    if 'reshape_transform' in kwargs:
        candidates.append({k: v for k, v in kwargs.items() if k != 'use_cuda'})
        # 兼容有 use_cuda 的老版本
        cand_with_usecuda = {k: v for k, v in kwargs.items()}
        if 'use_cuda' in kwargs:
            candidates.append(cand_with_usecuda)
    # 然后尝试不带 reshape_transform 的签名
    cand_basic = {k: v for k, v in kwargs.items() if k not in ['reshape_transform', 'use_cuda']}
    candidates.append(cand_basic)
    # 最后尝试仅 model+target_layers
    candidates.append({'model': kwargs.get('model'), 'target_layers': kwargs.get('target_layers')})

    last_exc = None
    for c in candidates:
        try:
            cam = cam_cls(**c)
            return cam
        except TypeError as e:
            last_exc = e
            continue
    # 如果都失败，抛出最后一个异常
    raise last_exc if last_exc is not None else RuntimeError("Unable to construct CAM instance")


def main():
    args = parse_args()
    # 处理 device 字符串，如果没有 GPU 则退回 cpu
    requested = args.device
    device = torch.device(requested if torch.cuda.is_available() and "cuda" in requested else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    print("加载 CLIP 包装器并生成 text embeddings（labels）：", args.labels)
    wrapper = CLIPWrapper(args.model_name, args.labels, device=device).to(device).eval()

    # 读取并预处理图像（使用 wrapper.processor）
    from PIL import Image
    img = Image.open(args.image_path).convert("RGB")
    processor = wrapper.get_processor()
    inputs = processor(images=img, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)  # (1,3,H,W)

    # 用于可视化的原始图像（规范化到 0-1 RGB）
      # 取得 model 实际输入大小（pixel_values 的 H,W）
    _, _, H_in, W_in = pixel_values.shape
    # 把原始 PIL 图像 resize 到 model 输入大小，用于和 cam_map 对齐可视化
    img_resized_for_vis = img.resize((W_in, H_in))
    img_rgb = np.array(img_resized_for_vis).astype(np.float32) / 255.0

    # 选择 target layer（由 wrapper 自动查找）
    target_layers = wrapper.find_target_layers()
    if target_layers is None:
        print("警告：未能自动找到 target layer。请手动修改脚本选择合适的层（参考 wrapper.clip 的模块结构）。")
        if hasattr(wrapper.clip, "visual") and hasattr(wrapper.clip.visual, "ln_post"):
            target_layers = [wrapper.clip.visual.ln_post]
            print("使用 fallback target layer: clip.visual.ln_post")
        else:
            raise RuntimeError("无法自动定位 target layer，请手动指定。")

    methods = {"gradcam": GradCAM, "scorecam": ScoreCAM, "eigencam": EigenCAM, "gradcam++": GradCAMPlusPlus}
    cam_cls = methods.get(args.method)
    if cam_cls is None:
        raise ValueError("Unsupported method: %s" % args.method)

    use_cuda = device.type == "cuda"

    # 兼容不同版本的 CAM 构造函数：使用 try_create_cam 尝试不同签名
    try:
        cam = try_create_cam(cam_cls,
                             model=wrapper,
                             target_layers=target_layers,
                             reshape_transform=reshape_transform,
                             use_cuda=use_cuda)
    except Exception as e:
        # 如果失败，打印友好提示并重试更保守的签名
        print("尝试构造 CAM 失败：", repr(e))
        print("尝试使用最基本签名创建 CAM（model, target_layers）")
        cam = try_create_cam(cam_cls, model=wrapper, target_layers=target_layers)

    # 尝试设置 batch_size 属性（部分实现支持）
    try:
        cam.batch_size = 32
    except Exception:
        pass

    # 为每个 label 分别生成热力图
    for idx, label in enumerate(args.labels):
        targets = [ClassifierOutputTarget(idx)]
        print(f"\n生成热力图 for: {label} (index {idx})")
        # 调用 cam，返回格式依库版本而异，做通用处理
        try:
            grayscale_cam = cam(input_tensor=pixel_values, targets=targets)
        except TypeError:
            # 某些版本的接口名或参数不同，尝试不带 targets
            grayscale_cam = cam(input_tensor=pixel_values)
        # grayscale_cam 可能是 numpy、list 或 tensor
        if isinstance(grayscale_cam, (list, tuple)):
            grayscale_cam = grayscale_cam[0]
        grayscale_cam = np.array(grayscale_cam)
        # 可能为 (B,H,W) 或 (B,1,H,W)
        if grayscale_cam.ndim == 3:
            cam_map = grayscale_cam[0, :]
        elif grayscale_cam.ndim == 4:
            cam_map = grayscale_cam[0, 0, :]
        else:
            raise RuntimeError("Unexpected cam shape: %s" % (grayscale_cam.shape,))

        # show_cam_on_image 需要 img_rgb (H,W,3) float 0-1
        visualization = show_cam_on_image(img_rgb, cam_map, use_rgb=True)

        safe_label = label.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(args.output_dir, f"{args.method}_{safe_label}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
        print(f"保存结果： {out_path}")


if __name__ == "__main__":
    main()