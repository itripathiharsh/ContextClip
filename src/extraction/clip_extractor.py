import os
import subprocess
import logging
from typing import List, Dict


logger = logging.getLogger(__name__)


class ClipExtractor:
    """
    Extracts video clips using FFmpeg (no re-encoding)

    - Fast (stream copy)
    - Timestamp accurate
    - Batch extraction supported
    """

    def __init__(self, input_video_path: str, output_dir: str = "output/clips"):
        self.input_video_path = input_video_path
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    # -------------------------
    # Core extraction function
    # -------------------------

    def extract_clip(
        self,
        start: float,
        end: float,
        clip_id: int
    ) -> str:
        """
        Extract a single clip

        Returns:
            output file path
        """

        duration = max(0, end - start)

        if duration <= 0:
            logger.warning(f"[Extractor] Invalid duration for clip {clip_id}")
            return ""

        output_path = os.path.join(self.output_dir, f"clip_{clip_id}.mp4")

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-ss", str(start),
            "-i", self.input_video_path,
            "-t", str(duration),
            "-map", "0",           # ← add this
            "-c", "copy",
            "-avoid_negative_ts", "1",
            output_path
        ]       

        logger.info(
            f"[Extractor] Extracting clip {clip_id} | start={start:.2f}s | duration={duration:.2f}s"
        )

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"[Extractor] Failed clip {clip_id}: {e}")
            return ""

        return output_path

    # -------------------------
    # Batch extraction
    # -------------------------

    def extract_all(self, segments: List[Dict]) -> List[Dict]:
        """
        Extract all segments and attach file paths

        Returns:
            updated segment metadata
        """

        extracted_segments = []

        for seg in segments:
            clip_id = seg.get("clip_id")

            start = seg["start"]
            end = seg["end"]

            path = self.extract_clip(start, end, clip_id)

            if not path:
                continue

            seg["file_path"] = path
            extracted_segments.append(seg)

        logger.info(f"[Extractor] Extracted {len(extracted_segments)} clips")

        return extracted_segments

    # -------------------------
    # Optional: safer extraction (re-encode)
    # -------------------------

    def extract_clip_reencode(
        self,
        start: float,
        end: float,
        clip_id: int
    ) -> str:
        """
        Slower but frame-accurate extraction (if needed)
        """

        duration = max(0, end - start)

        output_path = os.path.join(self.output_dir, f"clip_{clip_id}_re.mp4")

        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-y",
            "-ss", str(start),
            "-i", self.input_video_path,
            "-t", str(duration),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-c:a", "aac",
            output_path
        ]

        logger.warning(f"[Extractor] Using RE-ENCODE for clip {clip_id}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"[Extractor] Failed (reencode) clip {clip_id}: {e}")
            return ""

        return output_path