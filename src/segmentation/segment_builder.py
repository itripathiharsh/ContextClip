import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class SegmentBuilder:
    """
    Segment Builder with:
    - Motion + Person based segmentation
    - Fixed bridge logic (no timestamp bug)
    - Normalized meaningfulness scoring
    - Noise filtering
    """

    def __init__(
        self,
        min_duration: float = 8.0,
        max_duration: float = 120.0,
        bridge_gap: float = 10.0,
        hard_limit: float = 150.0
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.bridge_gap = bridge_gap
        self.hard_limit = hard_limit

        self.reset()

    # -------------------------
    # RESET
    # -------------------------
    def reset(self):
        self.state = "IDLE"
        self.current_segment: Optional[Dict] = None
        self.segments: List[Dict] = []
        self.last_active_ts: Optional[float] = None

    # -------------------------
    # PROCESS FRAME
    # -------------------------
    def process(
        self,
        timestamp: float,
        motion_score: float,
        person_detected: bool,
        is_active: bool
    ):
        if self.state == "IDLE":
            if is_active:
                self._start_segment(timestamp, motion_score, person_detected)

        elif self.state == "ACTIVE":
            if is_active:
                self._update_segment(timestamp, motion_score, person_detected)

                duration = timestamp - self.current_segment["start"]

                # Soft split (max duration reached)
                if duration >= self.max_duration:
                    self._finalize_segment("soft_split")

            else:
                self.state = "BRIDGE"

        elif self.state == "BRIDGE":
            gap = timestamp - (self.last_active_ts or timestamp)

            if is_active:
                self.state = "ACTIVE"
                self._update_segment(timestamp, motion_score, person_detected)

            elif gap >= self.bridge_gap:
                self._finalize_segment("gap_split")

    # -------------------------
    # START SEGMENT
    # -------------------------
    def _start_segment(self, timestamp, motion_score, person_detected):
        self.current_segment = {
            "start": timestamp,
            "end": timestamp,
            "motion_scores": [motion_score],
            "person_frames": 1 if person_detected else 0,
            "total_frames": 1
        }

        self.last_active_ts = timestamp
        self.state = "ACTIVE"

    # -------------------------
    # UPDATE SEGMENT
    # -------------------------
    def _update_segment(self, timestamp, motion_score, person_detected):
        seg = self.current_segment

        seg["end"] = timestamp
        seg["motion_scores"].append(motion_score)
        seg["total_frames"] += 1

        if person_detected:
            seg["person_frames"] += 1

        # ✅ FIX: track both motion + person activity
        if person_detected or motion_score > 0:
            self.last_active_ts = timestamp

    # -------------------------
    # FINALIZE SEGMENT
    # -------------------------
    def _finalize_segment(self, reason: str):
        seg = self.current_segment

        if not seg:
            return

        duration = seg["end"] - seg["start"]

        # ❌ Drop very small clips
        if duration < self.min_duration:
            logger.debug(f"[Segment] DROPPED (too short): {duration:.2f}s")
            self._reset_current()
            return

        motion_avg = sum(seg["motion_scores"]) / len(seg["motion_scores"])
        person_ratio = seg["person_frames"] / max(seg["total_frames"], 1)

        # ✅ FIX: Normalize motion score
        motion_normalized = min(motion_avg / 50.0, 1.0)

        meaningfulness_score = (
            0.6 * motion_normalized +
            0.4 * person_ratio
        )

        # ❌ Drop meaningless clips
        MIN_MEANINGFULNESS = 0.15
        if meaningfulness_score < MIN_MEANINGFULNESS:
            logger.debug(
                f"[Segment] DROPPED (low meaningfulness): {meaningfulness_score:.3f}"
            )
            self._reset_current()
            return

        segment_output = {
            "start": seg["start"],
            "end": seg["end"],
            "duration": duration,
            "motion_score": motion_avg,
            "person_ratio": person_ratio,
            "meaningfulness_score": meaningfulness_score,
            "end_reason": reason
        }

        self.segments.append(segment_output)

        logger.info(
            f"[Segment] Added | {duration:.1f}s | Score: {meaningfulness_score:.3f} | Reason: {reason}"
        )

        self._reset_current()

    # -------------------------
    # RESET CURRENT
    # -------------------------
    def _reset_current(self):
        self.current_segment = None
        self.state = "IDLE"
        self.last_active_ts = None

    # -------------------------
    # FINALIZE AT END
    # -------------------------
    def finalize(self):
        if self.current_segment:
            self._finalize_segment("end_of_video")

        return self.segments