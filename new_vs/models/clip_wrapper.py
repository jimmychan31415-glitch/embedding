#!/usr/bin/env python3
"""
CLIP 模型包装器：将 HuggingFace 的 CLIP 封装为一个分类器模块，返回文本标签的 logits，
并提供查找合适 target layer 的工具，供 pytorch-grad-cam 使用。
"""
from typing import List, Optional
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPWrapper(nn.Module):
    def __init__(self, model_name: str, labels: List[str], device: torch.device):
        super().__init__()
        self.device = device
        self.clip = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        # 预计算 text embeddings 并注册为 buffer（不训练）
        inputs = self.processor(text=labels, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_inputs = {k: v.to(device) for k, v in inputs.items()}
            text_embeds = self.clip.get_text_features(**text_inputs)  # (num_labels, dim)
            text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        self.register_buffer("text_embeds", text_embeds)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        输入：pixel_values（B,3,H,W） - 已由 CLIPProcessor 处理好的 tensor
        输出：logits (B, num_labels)
        """
        image_embeds = self.clip.get_image_features(pixel_values=pixel_values)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.clip.logit_scale.exp().to(image_embeds.dtype)
        logits = logit_scale * (image_embeds @ self.text_embeds.t())
        return logits

    def get_processor(self) -> CLIPProcessor:
        return self.processor

    def find_target_layers(self) -> Optional[List[nn.Module]]:
        """
        尝试自动寻找 CLIP 视觉骨干中可用于 CAM 的“末层”模块列表。
        返回 None 则表示自动查找失败，需要用户手动指定。
        常见候选：
          - clip.visual.transformer.resblocks[-1].ln_1 / ln_2
          - clip.visual.transformer.ln_post
          - clip.vision_model.encoder.layers[-1].layer_norm1 / layer_norm2
        """
        m = self.clip
        # 多个实现差异，尝试常见位置
        try:
            # HF CLIP older layout: m.visual.transformer.resblocks
            if hasattr(m, "visual") and hasattr(m.visual, "transformer") and hasattr(m.visual.transformer, "resblocks"):
                last = m.visual.transformer.resblocks[-1]
                # prefer ln_1 or ln_2 if present
                if hasattr(last, "ln_1"):
                    return [last.ln_1]
                if hasattr(last, "ln_2"):
                    return [last.ln_2]
            # HF CLIP newer: m.vision_model.encoder.layers
            if hasattr(m, "vision_model") and hasattr(m.vision_model, "encoder") and hasattr(m.vision_model.encoder, "layers"):
                last = m.vision_model.encoder.layers[-1]
                if hasattr(last, "layer_norm1"):
                    return [last.layer_norm1]
                if hasattr(last, "layer_norm2"):
                    return [last.layer_norm2]
            # ln_post is sometimes available
            if hasattr(m, "visual") and hasattr(m.visual, "ln_post"):
                return [m.visual.ln_post]
        except Exception:
            pass
        # 如果没找到
        return None