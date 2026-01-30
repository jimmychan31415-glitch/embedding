# generate_all_three_cams_fixed.py - 最終修正版（正確 import + 三種 CAM）

import torch
import torch.nn as nn
from transformers import AutoProcessor, AutoModel
from pytorch_grad_cam import GradCAMPlusPlus, LayerCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

# ---------------------------------------------------------
# 基本設定
# ---------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

# ---------------------------------------------------------
# 載入 CLIP 模型與 Processor
# ---------------------------------------------------------
# 注意：需先在環境中安裝 transformers, pytorch, pytorch-grad-cam, pillow, matplotlib, opencv-python, numpy
# pip install transformers torch pytorch-grad-cam pillow matplotlib opencv-python numpy
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

# ---------------------------------------------------------
# 指定 target_layer（CLIP Vision Transformer 最後一層 encoder block）
# ---------------------------------------------------------
target_layer = model.vision_model.encoder.layers[-1]

# ---------------------------------------------------------
# reshape_transform：把 ViT 的 token 特徵轉成 (B, C, H, W)
# ---------------------------------------------------------
def reshape_transform(tensor):
    """
    pytorch-grad-cam 會把 hook 的 activation 傳進來。
    CLIP ViT encoder layer 的輸出在某些情況下可能是 tuple 或 list，
    因此先做類型判斷再轉換成 (B, C, H, W)。
    """
    # activation 可能是 (hidden_states,) 或類似結構
    if isinstance(tensor, (tuple, list)):
        tensor = tensor[0]

    # tensor shape: (B, N+1, C) [CLS + patches]
    # 去掉 CLS token
    result = tensor[:, 1:, :]  # (B, N, C)

    # N 應該是 H*W，例如 14*14 = 196
    height = width = int(result.shape[1] ** 0.5)
    # reshape 成 (B, H, W, C)
    result = result.reshape(tensor.size(0), height, width, tensor.size(-1))
    # 轉成 (B, C, H, W)
    result = result.permute(0, 3, 1, 2).contiguous()

    return result

# ---------------------------------------------------------
# Wrapper：給需要梯度的 CAM（GradCAM++ / LayerCAM）用
# 把輸出壓成 scalar，避免 loss 維度錯誤
# ---------------------------------------------------------
class ScalarWrapper(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        # x: pixel_values (B, 3, H, W)
        outputs = self.m.vision_model(x)
        # CLIPVisionModelOutput を想定
        if hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state  # (B, N+1, C)
        else:
            # 念のため tuple/list の場合も想定
            last_hidden = outputs[0]
        cls_token = last_hidden[:, 0, :]  # CLS token (B, C)
        return cls_token.mean()  # スカラー（CAM 用の "疑似ロス"）

wrapped_model = ScalarWrapper(model)

# ---------------------------------------------------------
# 主要函數：三種 CAM 生成 + 儲存
# ---------------------------------------------------------
def generate_all_cams(image_path: str):
    # 讀取影像
    image = Image.open(image_path).convert("RGB")

    # CLIP 前處理
    inputs = processor(images=image, return_tensors="pt").to(device)

    # CAM 疊圖用的原圖 (0~1 正規化, 224x224)
    img_rgb = np.array(image.resize((224, 224))) / 255.0

    # 三種 CAM 方法
    methods = {
        "GradCAM++": GradCAMPlusPlus,
        "LayerCAM": LayerCAM,
        "EigenCAM": EigenCAM,
    }

    for name, CamClass in methods.items():
        print(f"生成 {name} 熱圖...")

        # EigenCAM 可以直接用原模型；其他兩種需要梯度 -> 用 wrapped_model
        current_model = model if name == "EigenCAM" else wrapped_model

        cam = CamClass(
            model=current_model,
            target_layers=[target_layer],
            reshape_transform=reshape_transform,
            use_cuda=(device == "cuda"),
        )

        # pytorch-grad-cam 會在 forward hook + backward hook 裡面處理
        grayscale_cam = cam(input_tensor=inputs.pixel_values)  # shape: (B, H, W)

        cam_data = grayscale_cam[0]  # 取 batch 中第一張

        # 百分位裁剪，減少極值影響
        p5 = np.percentile(cam_data, 5)
        p95 = np.percentile(cam_data, 95)
        clipped = np.clip(cam_data, p5, p95)

        # 正規化到 0~1
        clipped = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)

        # 疊加到原圖上
        visualization = show_cam_on_image(
            img_rgb.astype(np.float32),
            clipped.astype(np.float32),
            use_rgb=True,
            colormap=cv2.COLORMAP_INFERNO,
        )

        # 儲存檔名（GradCAM++ -> gradcampp）
        save_path = f"heatmap_{name.lower().replace('++', 'pp')}.png"
        plt.imsave(save_path, visualization)
        print(f"{name} 熱圖已儲存為 {save_path}")

# ---------------------------------------------------------
# main
# ---------------------------------------------------------
if __name__ == "__main__":
    # 請確保這張圖存在於目前工作目錄（例：/home/khchaned/embedding/clip_jina/query_image.jpg）
    test_image = "query_image.jpg"
    generate_all_cams(test_image)

    print("\n完成！三種熱圖已產生。")