# tests/test_clip_cam.py
import unittest
import torch
import os
from src.clip_model import CLIPWrapper
from src.cam_methods import CLIPCAM

class TestCLIPCAM(unittest.TestCase):
    def setUp(self):
        self.clip_model = CLIPWrapper(model_name="ViT-B/32", device="cpu")
        
    def test_clip_loading(self):
        self.assertIsNotNone(self.clip_model.model)
        self.assertIsNotNone(self.clip_model.preprocess)
    
    def test_feature_extraction(self):
        # 使用测试图像
        features = self.clip_model.get_image_features("test_image.jpg")
        self.assertEqual(features.shape[1], 512)  # CLIP特征维度
    
    def test_similarity(self):
        similarity = self.clip_model.get_similarity("test_image.jpg", "a test image")
        self.assertIsInstance(similarity, float)
        self.assertTrue(-1 <= similarity <= 1)

if __name__ == '__main__':
    unittest.main()