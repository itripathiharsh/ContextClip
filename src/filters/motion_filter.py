import numpy as np
import logging
from collections import deque


logger = logging.getLogger(__name__)


class MotionFilter:
    """
    Adaptive Motion Filter

    - Computes frame difference
    - Uses rolling mean + std for adaptive threshold
    - Filters out low-motion frames
    - Outputs motion score for downstream stages
    """

    def __init__(
        self,
        history_size: int = 50,
        threshold_multiplier: float = 1.5,
        min_threshold: float = 2.0
    ):
        """
        Args:
            history_size: number of past frames for adaptive stats
            threshold_multiplier: controls sensitivity
            min_threshold: lower bound to avoid too sensitive detection
        """
        self.prev_gray = None

        self.motion_history = deque(maxlen=history_size)

        self.threshold_multiplier = threshold_multiplier
        self.min_threshold = min_threshold

    def _to_gray(self, frame):
        """
        Convert RGB frame to grayscale using mean
        (fast + sufficient for motion detection)
        """
        return frame.mean(axis=2)

    def _compute_motion(self, current_gray):
        """
        Compute motion score using frame difference
        """
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return 0.0

        diff = np.abs(current_gray - self.prev_gray)
        motion_score = diff.mean()

        self.prev_gray = current_gray

        return motion_score

    def _compute_threshold(self):
        """
        Adaptive threshold based on history
        """
        if len(self.motion_history) < 5:
            return self.min_threshold

        mean = np.mean(self.motion_history)
        std = np.std(self.motion_history)

        threshold = mean + self.threshold_multiplier * std

        return max(threshold, self.min_threshold)

    def process(self, frame):
        """
        Main function to process a frame

        Returns:
            is_active (bool): whether frame has significant motion
            motion_score (float): raw motion intensity
        """
        gray = self._to_gray(frame)

        motion_score = self._compute_motion(gray)

        # update history AFTER computing score
        self.motion_history.append(motion_score)

        threshold = self._compute_threshold()

        is_active = motion_score > threshold

        logger.debug(
            f"[Motion] score={motion_score:.2f}, threshold={threshold:.2f}, active={is_active}"
        )

        return is_active, motion_score

    def reset(self):
        """
        Reset internal state (useful between videos)
        """
        self.prev_gray = None
        self.motion_history.clear()