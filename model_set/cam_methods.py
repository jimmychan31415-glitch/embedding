# cam_methods.py - 最終修正版（使用 ScoreCAM + 自訂包裝，處理 DINOv2 輸出）

import torch
from transformers import AutoImageProcessor, AutoModel
from pytorch_grad_cam import ScoreCAM  # gradient-free 方法，適合無分類頭模型
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

# 載入 DINOv2
print("載入 DINOv2...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()

# 正確 target_layer
target_layer = dinov2_model.encoder.layer[-1]
print("Target layer 已設定：", target_layer)

# ViT 專用 reshape_transform
def vit_reshape_transform(tensor, height=16, width=16):
    result = tensor[:, 1:, :]  # 移除 CLS token
    result = result.reshape(tensor.size(0), height, width, tensor.size(-1))
    result = result.transpose(2, 3).transpose(1, 2)  # [batch, dim, H, W]
    return result

# 自訂 model wrapper（關鍵！讓套件能處理 DINOv2 輸出物件）
class Dinov2Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        return outputs.pooler_output  # 用 pooled output 作為「pseudo-logits」 ( [batch, dim] )

# 使用 wrapper
wrapped_model = Dinov2Wrapper(dinov2_model)

def generate_cam(image_path, save_path="cam_scorecam_dinov2.png"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    cam_method = ScoreCAM(
        model=wrapped_model,  # 用 wrapper 包裝，避免輸出物件問題
        target_layers=[target_layer],
        reshape_transform=vit_reshape_transform
    )
    
    grayscale_cam = cam_method(input_tensor=inputs.pixel_values, targets=None)
    
    img_rgb = np.array(image.resize((224, 224))) / 255.0
    
    visualization = show_cam_on_image(img_rgb, grayscale_cam[0], use_rgb=True)
    plt.imsave(save_path, visualization)
    print(f"ScoreCAM 熱圖已儲存為 {save_path}")

if __name__ == "__main__":
    test_image = "query_image.jpg"  # 改成你的圖片路徑
    
    generate_cam(test_image)
    
    print("完成！熱圖已產生。")