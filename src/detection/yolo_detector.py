import logging
from typing import Dict, Any, List

import numpy as np
import torch

# -------------------------
# ✅ SAFE PATCH (LOCAL ONLY)
# -------------------------
_original_torch_load = torch.load

def safe_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = safe_torch_load

# -------------------------
# IMPORT YOLO AFTER PATCH
# -------------------------
from ultralytics import YOLO


logger = logging.getLogger(__name__)


class YOLODetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        device: str = None
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device or "cpu"

        logger.info(f"[YOLO] Loading model: {self.model_path}")
        logger.info(f"[YOLO] Using device: {self.device}")

        # ✅ NOW THIS WILL WORK
        self.model = YOLO(self.model_path)

    # -------------------------
    # SINGLE FRAME
    # -------------------------
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        results = self.model(
            frame,
            verbose=False,
            conf=self.conf_threshold
        )

        person_detected = False
        max_conf = 0.0
        count = 0

        for r in results:
            if r.boxes is None:
                continue

            for cls, conf in zip(r.boxes.cls, r.boxes.conf):
                if int(cls) == 0:
                    person_detected = True
                    count += 1
                    max_conf = max(max_conf, float(conf))

        return {
            "person_detected": person_detected,
            "confidence": max_conf,
            "num_persons": count
        }

    # -------------------------
    # WARMUP
    # -------------------------
    def warmup(self):
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        try:
            _ = self.detect(dummy)
            logger.info("[YOLO] Warmup complete")
        except Exception as e:
            logger.error(f"[YOLO] Warmup failed: {e}")