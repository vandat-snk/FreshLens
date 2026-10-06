"""
GENERATE_GRADCAM_DEMO.py
Trực quan hóa Grad-CAM cho ảnh mẫu để phục vụ báo cáo TV3
"""

import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from freshlens_ai.models.efficientnet import make_model
from freshlens_ai.inference.gradcam import GradCAM, overlay_heatmap

def run_gradcam_demo():
    model_path = "models/cnn_efficientnet_b0/best.pt"
    image_path = "docs/assets/01_prediction.png"
    output_path = "eval_results/gradcam_result.png"

    os.makedirs("eval_results", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Model
    model = make_model(pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('model_state', checkpoint.get('model_state_dict', checkpoint))
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Xác định Target Layer cho EfficientNet-B0 (Lớp Conv cuối cùng)
    target_layer = model.features[-1]

    # Preprocess Image
    orig_img = Image.open(image_path).convert('RGB')
    orig_np = np.array(orig_img)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(orig_img).unsqueeze(0).to(device)

    # Khởi tạo Grad-CAM & Sinh Heatmap
    gradcam = GradCAM(model, target_layer)
    heatmap, pred_idx = gradcam.generate_heatmap(input_tensor)

    # Phủ Heatmap lên ảnh gốc
    result_overlay = overlay_heatmap(heatmap, cv2.resize(orig_np, (224, 224)))

    # Lưu kết quả
    Image.fromarray(result_overlay).save(output_path)
    print(f"✅ Đã tạo ảnh Grad-CAM trực quan hóa thành công tại: {output_path}")

if __name__ == "__main__":
    run_gradcam_demo()