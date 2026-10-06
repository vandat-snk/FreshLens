"""
freshlens_ai/inference/cnn_predict.py
Pipeline suy luận duy nhất dùng chung cho CLI và UI Streamlit.
Tích hợp Quality Rejection dựa trên công thức toán học xử lý ảnh (Giáo trình PTIT 2023).
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# Import các module nội bộ của dự án
from freshlens_ai.models.efficientnet import make_model
from freshlens_ai.inference.open_set import load_gate, supported_probability


def check_image_quality(image_input, blur_threshold=100.0, dark_threshold=40.0, bright_threshold=215.0):
    """
    Kiểm tra chất lượng ảnh dựa trên Công thức Toán học Xử lý Ảnh (Giáo trình PTIT 2023):
    1. Độ mờ (Blur): Phương sai toán tử Laplace 2D Var(∇²I) (Công thức 5-14)
    2. Độ sáng (Exposure): Giá trị độ sáng trung bình toàn cục m_G (Công thức 5-54)
    3. Kích thước tối thiểu: Width x Height >= 100x100
    """
    if isinstance(image_input, Image.Image):
        img_np = np.array(image_input)
    else:
        img_np = image_input.copy()
        
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
        
    h, w = gray.shape
    if h < 100 or w < 100:
        return False, f"Ảnh quá nhỏ ({w}x{h}), yêu cầu tối thiểu 100x100 pixels."

    # 1. Tính phương sai của toán tử Laplace 2D để đo độ mờ (Chương 5 - Công thức 5-14)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False, f"Ảnh bị mờ (Laplacian Var = {laplacian_var:.1f} < {blur_threshold}). Vui lòng chụp lại!"

    # 2. Tính giá trị độ sáng trung bình m_G (Chương 5 - Công thức 5-54)
    m_G = np.mean(gray)
    if m_G < dark_threshold:
        return False, f"Ảnh quá tối (Độ sáng m_G = {m_G:.1f} < {dark_threshold}). Vui lòng tăng ánh sáng!"
    if m_G > bright_threshold:
        return False, f"Ảnh bị cháy sáng (Độ sáng m_G = {m_G:.1f} > {bright_threshold}). Vui lòng chỉnh góc chụp!"

    return True, "Quality Accepted"


class FreshLensPredictor:
    """
    Bộ dự đoán chuẩn hóa cho FreshLens V2
    - Áp dụng Fruit-first decision rule
    - Kiểm tra Quality Rejection & Open-set Gate
    """
    FRUITS = ['apple', 'banana', 'orange', 'tomato']
    CONDITIONS = ['fresh', 'rotten']
    JOINT_CLASSES = [
        'apple::fresh', 'apple::rotten',
        'banana::fresh', 'banana::rotten',
        'orange::fresh', 'orange::rotten',
        'tomato::fresh', 'tomato::rotten'
    ]

    def __init__(self, model_path, openset_gate_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load mô hình EfficientNet-B0
        self.model = make_model(pretrained=False)
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Tự động trích xuất state_dict phù hợp
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Load Open-set Gate nếu có
        self.openset_gate = None
        if openset_gate_path:
            try:
                self.openset_gate = load_gate(openset_gate_path)
            except Exception:
                self.openset_gate = None

        # Transformation chuẩn
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_input, use_tta=False, check_quality=True):
        start_time = time.time()
        
        if isinstance(image_input, str):
            pil_img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert('RGB')
        else:
            pil_img = image_input.convert('RGB')

        # 1. Quality Rejection Check
        if check_quality:
            is_good, quality_msg = check_image_quality(pil_img)
            if not is_good:
                return {
                    'is_supported': False,
                    'rejection_type': 'quality_rejection',
                    'rejection_reason': quality_msg,
                    'latency_ms': (time.time() - start_time) * 1000
                }

        # 2. Tiền xử lý & Inference (có TTA hoặc đơn)
        if use_tta:
            # Test-Time Augmentation: Ảnh gốc + Lật ngang + Center Crop nhẹ
            img_t1 = self.transform(pil_img).unsqueeze(0).to(self.device)
            img_t2 = self.transform(pil_img.transpose(Image.FLIP_LEFT_RIGHT)).unsqueeze(0).to(self.device)
            
            w, h = pil_img.size
            crop_img = pil_img.crop((int(w*0.05), int(h*0.05), int(w*0.95), int(h*0.95)))
            img_t3 = self.transform(crop_img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits1, feat1 = self.model(img_t1, return_features=True) if hasattr(self.model, 'forward_features') else (self.model(img_t1), None)
                logits2, _ = self.model(img_t2, return_features=True) if hasattr(self.model, 'forward_features') else (self.model(img_t2), None)
                logits3, _ = self.model(img_t3, return_features=True) if hasattr(self.model, 'forward_features') else (self.model(img_t3), None)
                
                logits = (logits1 + logits2 + logits3) / 3.0
                features = feat1
        else:
            img_tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                if hasattr(self.model, 'forward_features'):
                    logits, features = self.model(img_tensor, return_features=True)
                else:
                    logits = self.model(img_tensor)
                    features = None

        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # 3. Decision Rule: Fruit-first Decoder (4 fruits x 2 conditions)
        probs_matrix = probs.reshape(4, 2)
        fruit_probs = probs_matrix.sum(axis=1) # Tổng xác suất mỗi quả
        pred_fruit_idx = np.argmax(fruit_probs)
        pred_fruit = self.FRUITS[pred_fruit_idx]

        cond_probs = probs_matrix[pred_fruit_idx]
        pred_cond_idx = np.argmax(cond_probs)
        pred_cond = self.CONDITIONS[pred_cond_idx]

        fruit_conf = float(fruit_probs[pred_fruit_idx])
        cond_conf = float(cond_probs[pred_cond_idx] / (fruit_conf + 1e-7))
        joint_conf = float(probs[pred_fruit_idx * 2 + pred_cond_idx])

        # 4. Open-set Rejection Check
        is_supported = True
        openset_msg = "Supported"
        if self.openset_gate is not None:
            try:
                p_supported = supported_probability(self.openset_gate, probs)
                if p_supported < 0.5:
                    is_supported = False
                    openset_msg = f"Đối tượng nằm ngoài phạm vi nhận diện (Probability: {p_supported:.2f})"
            except Exception:
                pass

        if not is_supported:
            return {
                'is_supported': False,
                'rejection_type': 'openset_rejection',
                'rejection_reason': openset_msg,
                'latency_ms': (time.time() - start_time) * 1000
            }

        latency_ms = (time.time() - start_time) * 1000

        return {
            'is_supported': True,
            'fruit': pred_fruit,
            'condition': pred_cond,
            'joint_class': f"{pred_fruit}::{pred_cond}",
            'fruit_confidence': fruit_conf,
            'condition_confidence': cond_conf,
            'joint_confidence': joint_conf,
            'latency_ms': latency_ms
        }