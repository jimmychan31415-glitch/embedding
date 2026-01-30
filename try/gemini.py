import torch
import clip
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 1. Setup Device and Model
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load model and convert to float (FP32) to avoid HalfTensor issues and improve stability
model, preprocess = clip.load("ViT-L/14@336px", device=device)
model = model.float() 
model.eval()

# 2. Define the Fixed Wrapper
class CLIPVisualWrapper(torch.nn.Module):
    def __init__(self, model, prompt):
        super().__init__()
        self.model = model
        
        # Pre-compute text features
        text_token = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_token)
            # FIX: Use out-of-place normalization
            self.text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    def forward(self, x):
        # Encode image
        image_features = self.model.encode_image(x)
        
        # FIX: Changed /= to = / (Out-of-place)
        # This allows Autograd to track the original tensor for the backward pass
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity
        similarity = image_features @ self.text_features.T
        return similarity

# 3. Parameters
prompt = "a cute cat"
# Ensure you have an image file at this path
img_path = "better_dog_cat.jpg" 

wrapped_model = CLIPVisualWrapper(model, prompt).to(device)

# Target the LayerNorm layer in the last transformer block
# This is the standard target for ViT architectures in Grad-CAM
target_layers = [model.visual.transformer.resblocks[-1].ln_1]

# 4. Load and Preprocess Image
raw_image = Image.open(img_path).convert("RGB")
# We resize to 336 because that is the model's native resolution
input_tensor = preprocess(raw_image).unsqueeze(0).to(device)

# Create a normalized numpy image for visualization background
# We resize the raw image to match the model input for the overlay
rgb_img = np.array(raw_image.resize((336, 336))) / 255.0

# 5. Generate CAMs
# We initialize the methods
cam_methods = [
    ("GradCAM", GradCAM(model=wrapped_model, target_layers=target_layers)),
    ("GradCAM++", GradCAMPlusPlus(model=wrapped_model, target_layers=target_layers)),
    ("LayerCAM", LayerCAM(model=wrapped_model, target_layers=target_layers)),
    ("EigenCAM", EigenCAM(model=wrapped_model, target_layers=target_layers))
]

visualizations = []
for name, cam_method in cam_methods:
    print(f"Generating {name}...")
    # ClassifierOutputTarget(0) targets the first (and only) output logit
    grayscale_cam = cam_method(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])[0, :]
    
    # Overlay the heatmap on the original image
    vis = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    visualizations.append(vis)

# 6. Visualization Layout
fig, axes = plt.subplots(1, 6, figsize=(24, 6))
plt.subplots_adjust(wspace=0.1)

# Column 0: Label
axes[0].text(0.5, 0.5, "Cat", fontsize=25, fontweight='bold', ha='center', va='center')
axes[0].axis('off')
axes[0].set_title("Category", fontsize=14, fontweight='bold')

# Column 1: Original
axes[1].imshow(rgb_img)
axes[1].axis('off')
axes[1].set_title("Original Image", fontsize=14, fontweight='bold')

# Columns 2-5: CAMs
for i, (name, _) in enumerate(cam_methods):
    axes[i+2].imshow(visualizations[i])
    axes[i+2].axis('off')
    axes[i+2].set_title(name, fontsize=14, fontweight='bold')

plt.suptitle(f"CLIP ViT-L/14@336px Interpretation\nPrompt: '{prompt}'", fontsize=18, y=1.02)
plt.tight_layout()
plt.show()