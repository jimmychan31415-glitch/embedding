# generate_all_cams_torchcam.py - 使用 torchcam 生成三種熱圖（最終穩定版）

import torch
from transformers import AutoImageProcessor, AutoModel
from torchcam.methods import CAM, GradCAM, GradCAMpp
from torchcam.utils import overlay_mask
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

# 載入 DINOv2
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)
model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()

# target_layer（最後一層的 mlp.fc2）
target_layer = "encoder.layer.23.mlp.fc2"  # DINOv2-large 有 24 層 (0~23)
print(f"Target layer: {target_layer}")

def generate_heatmaps(image_path, save_prefix="heatmap"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    img_pil = image.resize((224, 224))
    img_np = np.array(img_pil) / 255.0
    
    # 先 forward 一次模型（關鍵！讓 torchcam hook 捕捉特徵）
    print("先執行 forward pass 以啟動 hook...")
    with torch.no_grad():
        _ = model(inputs.pixel_values)
    
    # 三種方法
    cam_methods = {
        "CAM": CAM(model, target_layer=target_layer),
        "GradCAM": GradCAM(model, target_layer=target_layer),
        "GradCAM++": GradCAMpp(model, target_layer=target_layer)
    }
    
    for name, cam_extractor in cam_methods.items():
        print(f"生成 {name} 熱圖...")
        activation_map = cam_extractor(inputs.pixel_values)
        heatmap = activation_map[0].squeeze(0).cpu().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        # 疊加熱圖（用 JET colormap 紅黃鮮豔）
        visualization = overlay_mask(img_np, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET)
        
        save_path = f"{save_prefix}_{name.lower()}.png"
        plt.imsave(save_path, visualization)
        print(f"{name} 熱圖已儲存為 {save_path}")
    
    # 清理 hook
    for cam_extractor in cam_methods.values():
        cam_extractor.remove_hooks()

if __name__ == "__main__":
    test_image = "query_image.jpg"  # 改成你的圖片路徑
    generate_heatmaps(test_image, save_prefix="heatmap_eiffel")
    
    print("\n完成！三種熱圖已產生。")
    print("下載指令：")
    print("scp -P 2200 khchaned@192.168.122.72:~/embedding/heatmap_eiffel_* ~/Desktop/")