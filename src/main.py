import os
import time
import json
import logging

from src.ingestion.ffmpeg_stream import FFmpegFrameStreamer
from src.filters.motion_filter import MotionFilter
from src.detection.yolo_detector import YOLODetector
from src.segmentation.segment_builder import SegmentBuilder
from src.extraction.clip_extractor import ClipExtractor


# -------------------------
# LOGGING
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# -------------------------
# CONFIG
# -------------------------
VIDEO_PATH = "data/input.mp4"
OUTPUT_DIR = "output/clips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

YOLO_SAMPLE_INTERVAL = 3.0
PERSON_SAMPLE_INTERVAL = 2.0


# -------------------------
# MAIN
# -------------------------
def run_pipeline():

    logger.info("========== VIDEO PIPELINE START ==========")

    streamer = FFmpegFrameStreamer(VIDEO_PATH)
    motion_filter = MotionFilter()
    detector = YOLODetector()
    segment_builder = SegmentBuilder()
    extractor = ClipExtractor(VIDEO_PATH)

    start_time = time.time()

    processed_frames = 0
    last_yolo_ts = -999
    yolo_calls = 0

    # -------------------------
    # LOOP
    # -------------------------
    for frame, timestamp in streamer.frames():

        processed_frames += 1

        # -------------------------
        # MOTION
        # -------------------------
        is_active, motion_score = motion_filter.process(frame)

        # -------------------------
        # YOLO (FIXED)
        # -------------------------
        person_detected = False
        should_run_yolo = False

        if is_active and (timestamp - last_yolo_ts) >= YOLO_SAMPLE_INTERVAL:
            should_run_yolo = True
        elif (timestamp - last_yolo_ts) >= PERSON_SAMPLE_INTERVAL:
            should_run_yolo = True

        if should_run_yolo:
            result = detector.detect(frame)
            person_detected = result["person_detected"]
            last_yolo_ts = timestamp
            yolo_calls += 1

        # -------------------------
        # SEGMENT BUILDER
        # -------------------------
        segment_builder.process(
            timestamp=timestamp,
            motion_score=motion_score,
            person_detected=person_detected,
            is_active=is_active
        )

        # -------------------------
        # PROGRESS
        # -------------------------
        if processed_frames % 500 == 0:
            elapsed = time.time() - start_time
            fps_proc = processed_frames / elapsed

            logger.info(
                f"[PROGRESS] Frames={processed_frames} | "
                f"YOLO={yolo_calls} | "
                f"Speed={fps_proc:.2f} FPS | "
                f"Time={elapsed/60:.2f} min"
            )

    # -------------------------
    # FINALIZE
    # -------------------------
    segments = segment_builder.finalize()

    logger.info(f"[INFO] Segments Generated: {len(segments)}")

    # -------------------------
    # CLIP EXTRACTION
    # -------------------------
    final_segments = []

    for idx, seg in enumerate(segments):
        clip_path = os.path.join(OUTPUT_DIR, f"clip_{idx}.mp4")

        extractor.extract_clip(
            start=seg["start"],
            end=seg["end"],
            clip_id=idx
        )

        seg["clip_id"] = idx
        seg["file_path"] = clip_path

        final_segments.append(seg)

    # -------------------------
    # SAVE METADATA
    # -------------------------
    with open("output/metadata.json", "w") as f:
        json.dump(final_segments, f, indent=2)

    # -------------------------
    # STATS
    # -------------------------
    total_output_duration = sum(
        seg["end"] - seg["start"] for seg in final_segments
    )

    total_time = time.time() - start_time

    logger.info("========== PIPELINE COMPLETE ==========")
    logger.info(f"Processing Time: {total_time/60:.2f} min")
    logger.info(f"YOLO Calls: {yolo_calls}")
    logger.info(f"Output Duration: {total_output_duration/60:.2f} min")


# -------------------------
# ENTRY
# -------------------------
if __name__ == "__main__":
    run_pipeline()