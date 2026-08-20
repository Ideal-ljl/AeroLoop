from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .observers import RolloutObserver
from .types import EpisodeResult, EpisodeSpec, Observation, StepRecord


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return cleaned[:160] or "episode"


@dataclass(frozen=True)
class MediaConfig:
    show_window: bool = False
    save_video: bool = False
    video_dir: str = "eval_results/videos"
    video_fps: float = 10.0
    ffmpeg_bin: str = "ffmpeg"
    window_name: str = "AeroLoop"
    wait_ms: int = 1
    overlay: bool = True
    save_collision_frame: bool = False
    collision_dir: str = "eval_results/collisions"

    def __post_init__(self) -> None:
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")
        if self.wait_ms < 1:
            raise ValueError("wait_ms must be at least 1")


class FFmpegVideoWriter:
    """Stream RGB uint8 frames to an H.264 MP4 without buffering a trajectory."""

    def __init__(self, path: str | Path, fps: float, ffmpeg_bin: str = "ffmpeg"):
        executable = shutil.which(ffmpeg_bin)
        if executable is None:
            raise FileNotFoundError(f"ffmpeg executable not found: {ffmpeg_bin}")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.executable = executable
        self.process = None
        self.width = None
        self.height = None

    @staticmethod
    def _prepare(frame):
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("video output requires aeroloop[media]") from exc
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"video frame must have shape (H,W,3), got {frame.shape}")
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        pad_bottom = frame.shape[0] % 2
        pad_right = frame.shape[1] % 2
        if pad_bottom or pad_right:
            frame = cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_REPLICATE)
        return np.ascontiguousarray(frame)

    def _start(self, frame) -> None:
        self.height, self.width = map(int, frame.shape[:2])
        command = [
            self.executable,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            str(self.fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    def write(self, frame) -> None:
        frame = self._prepare(frame)
        if self.process is None:
            self._start(frame)
        if frame.shape[:2] != (self.height, self.width):
            import cv2

            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(frame.tobytes())

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=30)
        self.process = None
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with code {return_code}: {self.path}")


def render_overlay(image, episode: EpisodeSpec, observation: Observation, record: StepRecord | None):
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("visual overlays require aeroloop[media]") from exc
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB frame must have shape (H,W,3), got {rgb.shape}")
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    panel_height = 164 if record is not None else 86
    overlay = bgr.copy()
    cv2.rectangle(overlay, (8, 8), (min(bgr.shape[1] - 8, 820), min(bgr.shape[0] - 8, panel_height)), (0, 0, 0), -1)
    bgr = cv2.addWeighted(overlay, 0.62, bgr, 0.38, 0)
    pose = observation.pose
    lines = [
        f"{episode.env_name} | {episode.episode_id}",
        f"step={observation.step_index} pose=[{pose.x:.2f}, {pose.y:.2f}, {pose.z:.2f}, yaw={pose.yaw:.2f}]",
    ]
    if record is not None:
        action = record.action
        lines.extend(
            [
                f"action=[{action.dx:.2f}, {action.dy:.2f}, {action.dz:.2f}, dyaw={action.d_yaw:.2f}, stop={action.stop:.2f}]",
                f"distance={record.distances.get('endpoint_3d', float('nan')):.2f} collision={record.collision}",
                f"inference_call={record.inference_call} chunk_index={record.action_index} infer_ms={record.inference_ms or 0.0:.1f}",
            ]
        )
    for index, line in enumerate(lines):
        color = (40, 80, 255) if record is not None and record.collision and index == 3 else (255, 255, 255)
        cv2.putText(bgr, line, (18, 30 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 1, cv2.LINE_AA)
    return np.ascontiguousarray(bgr[..., ::-1])


class MediaObserver(RolloutObserver):
    def __init__(self, config: MediaConfig):
        self.config = config
        self.writer: FFmpegVideoWriter | None = None
        self._artifacts: dict[str, Any] = {}
        self._window_enabled = config.show_window
        self._collision_saved = False

    def _frame(self, episode: EpisodeSpec, observation: Observation, record: StepRecord | None):
        if observation.rgb is None:
            return None
        return render_overlay(observation.rgb, episode, observation, record) if self.config.overlay else observation.rgb

    def _show(self, frame) -> bool:
        if not self._window_enabled or frame is None:
            return True
        try:
            import cv2

            cv2.imshow(self.config.window_name, frame[..., ::-1])
            key = cv2.waitKey(self.config.wait_ms) & 0xFF
            if key in (27, ord("q")):
                return False
            if key == ord(" "):
                while True:
                    paused_key = cv2.waitKey(50) & 0xFF
                    if paused_key in (27, ord("q")):
                        return False
                    if paused_key == ord(" "):
                        break
        except Exception as exc:
            self._window_enabled = False
            print(f"[media warning] disabling visualization window: {type(exc).__name__}: {exc}")
        return True

    def _write(self, frame) -> None:
        if self.writer is not None and frame is not None:
            self.writer.write(frame)
            self._artifacts["video_path"] = str(self.writer.path)

    def on_episode_start(self, episode: EpisodeSpec, observation: Observation) -> None:
        self._artifacts = {}
        self._collision_saved = False
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.config.save_video:
            video_path = (
                Path(self.config.video_dir) / f"{_safe_name(episode.env_name)}__{_safe_name(episode.episode_id)}.mp4"
            )
            self.writer = FFmpegVideoWriter(video_path, self.config.video_fps, self.config.ffmpeg_bin)
        frame = self._frame(episode, observation, None)
        self._write(frame)
        self._show(frame)

    def on_step(self, episode: EpisodeSpec, observation: Observation, record: StepRecord) -> bool:
        frame = self._frame(episode, observation, record)
        self._write(frame)
        if record.collision and self.config.save_collision_frame and not self._collision_saved and frame is not None:
            import cv2

            path = (
                Path(self.config.collision_dir)
                / f"{_safe_name(episode.env_name)}__{_safe_name(episode.episode_id)}.png"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(path), frame[..., ::-1])
            self._artifacts["collision_frame_path"] = str(path)
            self._collision_saved = True
        return self._show(frame)

    def on_episode_end(self, episode: EpisodeSpec, observation: Observation, result: EpisodeResult) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def artifacts(self):
        return dict(self._artifacts)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.config.show_window:
            try:
                import cv2

                cv2.destroyWindow(self.config.window_name)
            except Exception:
                pass
