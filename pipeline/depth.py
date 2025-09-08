import torch
import cv2
import numpy as np

class DepthEstimator:
    """
    Loads MiDaS (DPT-Hybrid) via torch.hub lazily and provides an infer() method.
    """
    def __init__(self, device=None, model_type="DPT_Hybrid"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self._model = None
        self._transform = None

    def _load(self):
        if self._model is None:
            self._model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self._model.to(self.device)
            self._model.eval()
            transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self._transform = transforms.dpt_transform
        return self._model, self._transform

    @torch.inference_mode()
    def infer(self, image_path):
        model, transform = self._load()
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)[:, :, ::-1]  # BGR->RGB
        inp = transform(img).to(self.device)
        pred = model(inp)
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        # Normalize to [0,1], invert so nearer = larger value
        d = pred
        d = (d - d.min()) / (d.max() - d.min() + 1e-8)
        d = 1.0 - d
        return d
