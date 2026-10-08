"""Grad-CAM for the fruit-first selected joint class, with removable hooks."""
import cv2
import numpy as np
import torch

from freshlens_ai.models import decode_probabilities


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.handle = target_layer.register_forward_hook(self.save_activation)

    def save_activation(self, module, inputs, output):
        self.activations = output

    def close(self):
        self.handle.remove()
        self.activations = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def generate_heatmap(self, input_tensor, class_idx=None):
        if input_tensor.shape[0] != 1:
            raise ValueError('Grad-CAM requires one image.')
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.enable_grad():
                # Works even if all model parameters have been frozen.
                tensor = input_tensor.detach().clone().requires_grad_(True)
                output = self.model(tensor)
                if class_idx is None:
                    probs = output.detach().float().softmax(1).cpu().numpy()
                    class_idx = int(decode_probabilities(probs)['joint'][0])
                if not 0 <= class_idx < output.shape[1]:
                    raise ValueError('Invalid Grad-CAM target class.')
                gradients = torch.autograd.grad(output[0, class_idx], self.activations)[0]
                weights = gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * self.activations).sum(dim=1).relu()[0]
                maximum = cam.max()
                if maximum > 0:
                    cam = cam / maximum
                return cam.detach().cpu().numpy(), class_idx
        finally:
            self.activations = None
            self.model.train(was_training)


def overlay_heatmap(heatmap, image_rgb, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Image must use the same letterbox canvas as the model input."""
    if not 0 <= alpha <= 1:
        raise ValueError('Alpha must be between zero and one.')
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(heatmap, (w, h))
    colored = cv2.applyColorMap(np.uint8(255 * np.clip(resized, 0, 1)), colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, colored, alpha, 0)
