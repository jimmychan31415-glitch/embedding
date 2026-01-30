# setup_clip.py - 設定 OpenAI CLIP 模型 + 試不同層

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用裝置：{device} (RTX 5090 D 已啟用)")

# 載入 OpenAI CLIP (ViT-L/14)
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14", use_fast=True)
clip_model = AutoModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()

def extract_embedding_clip(image, layer_idx=-1):
    """提取 CLIP 嵌入，支援最後層 (-1) 與中間層 (12)"""
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = clip_model.vision_model(inputs.pixel_values, output_hidden_states=True)
    
    layer_out = outputs.hidden_states[layer_idx]
    global_emb = layer_out[:, 0, :]  # CLS token 全局嵌入
    return global_emb

# 測試
if __name__ == "__main__":
    image = Image.open("query_image.jpg").convert("RGB")
    
    emb_last = extract_embedding_clip(image, -1)
    emb_mid = extract_embedding_clip(image, 12)
    
    print("CLIP 嵌入提取完成！")
    print(f"最後層嵌入形狀：{emb_last.shape}")
    print(f"中間層 (12) 嵌入形狀：{emb_mid.shape}")