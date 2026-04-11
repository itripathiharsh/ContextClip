import subprocess
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FFmpegFrameStreamer:
    """
    Streams video frames using FFmpeg without writing to disk.
    Returns (frame, timestamp)
    """

    def __init__(
        self,
        video_path: str,
        fps: int = 2,
        width: int = 640,
        height: int = 360
    ):
        self.video_path = video_path
        self.fps = fps
        self.width = width
        self.height = height

        self.process = None
        self.frame_size = self.width * self.height * 3  # RGB24

    def _build_command(self):
        return [
            "ffmpeg",
            "-loglevel", "error",
            "-i", self.video_path,
            "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-"
        ]

    def start(self):
        cmd = self._build_command()

        logger.info(f"[FFmpeg] Starting stream: {self.video_path}")
        logger.info(f"[FFmpeg] FPS={self.fps}, RES={self.width}x{self.height}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8
        )

    def read_frame(self):
        raw_bytes = self.process.stdout.read(self.frame_size)

        if not raw_bytes or len(raw_bytes) < self.frame_size:
            return None

        frame = np.frombuffer(raw_bytes, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))

        return frame

    def frames(self):
        self.start()

        frame_count = 0

        while True:
            frame = self.read_frame()

            if frame is None:
                logger.info("[FFmpeg] Stream ended.")
                break

            timestamp = frame_count / self.fps
            frame_count += 1

            if frame_count % 200 == 0:
                logger.info(f"[FFmpeg] Processed {frame_count} frames")

            yield frame, timestamp

        self.stop()

    def stop(self):
        if self.process:
            logger.info("[FFmpeg] Stopping stream...")
            self.process.stdout.close()
            self.process.stderr.close()
            self.process.wait()
            self.process = None