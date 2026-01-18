# models_setup.py - 模型載入（已修正警告）

import torch
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device}")

print("載入 DINOv2...")
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large", use_fast=True)
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-large").to(device).eval()

print("載入 CLIP...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

def extract_embedding(model_type, image):
    """從單張 PIL Image 提取全局嵌入"""
    if model_type == "dinov2":
        inputs = dinov2_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = dinov2_model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    elif model_type == "clip":
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            return clip_model.get_image_features(**inputs)