"""
freshlens_ai/inference/gradcam.py
Triển khai Grad-CAM (Gradient-weighted Class Activation Mapping) cho EfficientNet-B0.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook để bắt feature maps và gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()

        # Backward pass
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward(retain_graph=True)

        # Tính trọng số alpha k bằng Global Average Pooling của gradients
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Áp dụng ReLU và chuẩn hóa về [0, 1]
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx


def overlay_heatmap(heatmap, original_image_np, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Phủ heatmap lên ảnh gốc (RGB)
    """
    h, w = original_image_np.shape[:2]
    resized_heatmap = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * resized_heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(original_image_np, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay