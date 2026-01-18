# generate_eigencam_heatmap.py - 使用 EigenCAM 生成熱圖（穩定版，適合 DINOv2）

import torch
from transformers import AutoImageProcessor, AutoModel
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

# 載入 DINOv2
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)
model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()

# 包裝模型以返回張量而非 BaseModelOutputWithPooling
class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        outputs = self.model(x)
        # 返回最後一層隱藏狀態的均值作為偽分類 logits
        mean_output = outputs.last_hidden_state.mean(dim=1)  # [batch_size, hidden_size]
        mean_scalar = mean_output.mean(dim=-1)  # [batch_size]
        # 擴展為 1000 維（假設 ImageNet 類別數）以符合 CAM 期望
        dummy_logits = mean_scalar.unsqueeze(1).repeat(1, 1000)
        return dummy_logits

wrapped_model = ModelWrapper(model).to(device).eval()

# target_layer - 使用最後一層的輸出
target_layer = wrapped_model.model.encoder.layer[-1]

# reshape_transform（動態計算 ViT 的空間尺寸）
def reshape_transform(tensor):
    batch_size, num_patches_plus_one, hidden_size = tensor.shape
    # 計算 patch 網格的尺寸（假設輸入為正方形）
    height = width = int((num_patches_plus_one - 1) ** 0.5)
    # 移除 cls token 並重塑為 [batch_size, height, width, hidden_size]
    result = tensor[:, 1:, :].reshape(batch_size, height, width, hidden_size)
    # 轉換為 [batch_size, hidden_size, height, width]
    return result.transpose(2, 3).transpose(1, 2)

def generate_eigencam(image_path, save_path="eigencam_heatmap.png"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    img_rgb = np.array(image.resize((224, 224))) / 255.0
    
    print("生成 EigenCAM 熱圖...")
    cam = EigenCAM(
        model=wrapped_model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )
    
    grayscale_cam = cam(input_tensor=inputs.pixel_values)
    
    # 加強對比（percentile 剪裁）
    cam_data = grayscale_cam[0]
    p5 = np.percentile(cam_data, 5)
    p95 = np.percentile(cam_data, 95)
    clipped = np.clip(cam_data, p5, p95)
    clipped = (clipped - clipped.min()) / (clipped.max() - clipped.min() + 1e-8)
    
    # 用鮮豔 colormap
    visualization = show_cam_on_image(img_rgb, clipped, use_rgb=True, colormap=cv2.COLORMAP_INFERNO)
    
    plt.imsave(save_path, visualization)
    print(f"EigenCAM 熱圖已儲存為 {save_path}")

if __name__ == "__main__":
    test_image = "query_image.jpg"  # 改成你的圖片路徑
    generate_eigencam(test_image)
    
    print("\n完成！")