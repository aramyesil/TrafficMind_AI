"""Detect, track, and count vehicles in an uploaded traffic video."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from src.counting import LineCounter
from src.labels import DIRECTION_NAMES_TR, class_color, class_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "runs" / "detect" / "runs" / "yolo26n" / "weights" / "best.pt"
FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
)


@dataclass
class AnalysisResult:
    output_video: Path
    frame_stats: pd.DataFrame
    crossings: pd.DataFrame
    instant_peak: int
    total_crossings: int
    fps: float
    frame_count: int


@dataclass
class _Detection:
    track_id: int | None
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def anchor(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, float(self.y2))


def load_model(model_path: Path = MODEL_PATH) -> YOLO:
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained weights were not found: {model_path}")
    return YOLO(str(model_path))


def _font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1


def line_pixel(orientation: str, ratio: float, width: int, height: int) -> int:
    ratio = min(max(ratio, 0.02), 0.98)
    if orientation == "horizontal":
        return int(round(height * ratio))
    return int(round(width * ratio))


def read_first_frame(video_path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return None
    return frame


def draw_counting_line(frame: np.ndarray, orientation: str, line_position: int) -> np.ndarray:
    preview = frame.copy()
    height, width = preview.shape[:2]
    color = (250, 204, 21)
    if orientation == "horizontal":
        cv2.line(preview, (0, line_position), (width - 1, line_position), color, 2, cv2.LINE_AA)
    else:
        cv2.line(preview, (line_position, 0), (line_position, height - 1), color, 2, cv2.LINE_AA)
    return preview


def _detections_from_result(result) -> list[_Detection]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().numpy()
    confidences = boxes.conf.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    if boxes.id is None:
        track_ids = [None] * len(xyxy)
    else:
        track_ids = [int(value) for value in boxes.id.detach().cpu().numpy()]
    names = result.names
    detections = []
    for box, confidence, class_id, track_id in zip(xyxy, confidences, classes, track_ids):
        x1, y1, x2, y2 = (int(round(value)) for value in box)
        class_name = names[int(class_id)] if isinstance(names, dict) else names[int(class_id)]
        detections.append(
            _Detection(
                track_id=None if track_id is None else int(track_id),
                class_name=str(class_name),
                confidence=float(confidence),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
            )
        )
    return detections


def _annotate(
    frame: np.ndarray,
    detections: list[_Detection],
    orientation: str,
    line_position: int,
    instant_count: int,
    total_count: int,
) -> np.ndarray:
    annotated = draw_counting_line(frame, orientation, line_position)
    for detection in detections:
        color = class_color(detection.class_name)
        bgr = (color[2], color[1], color[0])
        cv2.rectangle(annotated, (detection.x1, detection.y1), (detection.x2, detection.y2), bgr, 2, cv2.LINE_AA)

    image = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    height, width = frame.shape[:2]
    label_font = _font(max(14, width // 55))
    hud_font = _font(max(18, width // 42))
    for detection in detections:
        identity = "-" if detection.track_id is None else str(detection.track_id)
        text = f"{identity}  {class_label(detection.class_name)}  {detection.confidence:.2f}"
        left, top, right, bottom = draw.textbbox((0, 0), text, font=label_font)
        text_width = right - left
        text_height = bottom - top
        text_x = max(0, min(detection.x1, width - text_width - 8))
        text_y = detection.y1 - text_height - 8
        if text_y < 0:
            text_y = min(detection.y2 + 4, height - text_height - 6)
        color = class_color(detection.class_name)
        draw.rectangle(
            (text_x, text_y, text_x + text_width + 8, text_y + text_height + 6),
            fill=color + (220,),
        )
        draw.text((text_x + 4, text_y + 2), text, font=label_font, fill=(255, 255, 255, 255))

    hud = f"Anlık araç: {instant_count}    Geçen araç: {total_count}"
    left, top, right, bottom = draw.textbbox((0, 0), hud, font=hud_font)
    draw.rectangle((12, 12, 24 + right - left, 20 + bottom - top), fill=(15, 23, 42, 190))
    draw.text((18, 14), hud, font=hud_font, fill=(255, 255, 255, 255))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _transcode_for_browser(source: Path, destination: Path) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        message = completed.stderr.strip() or "ffmpeg could not encode the video"
        raise RuntimeError(message)


def ensure_decodable_video(video_path: Path) -> Path:
    """Return a path OpenCV can read, converting unusual uploads when needed."""
    capture = cv2.VideoCapture(str(video_path))
    opened = capture.isOpened()
    readable = False
    if opened:
        readable, _ = capture.read()
    capture.release()
    if opened and readable:
        return video_path
    converted = video_path.with_name(f"{video_path.stem}_browser.mp4")
    _transcode_for_browser(video_path, converted)
    return converted


def analyze_video(
    model: YOLO,
    source_path: Path,
    output_path: Path,
    confidence: float,
    orientation: str,
    line_ratio: float,
    count_once_per_direction: bool = True,
    progress_callback=None,
) -> AnalysisResult:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise RuntimeError("Video açılamadı. MP4, AVI veya MOV dosyasını kontrol edin.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0) or 25.0
    width = _even(int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    height = _even(int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    capture.release()
    if width < 2 or height < 2:
        raise RuntimeError("Video boyutu okunamadı.")

    line_position = line_pixel(orientation, line_ratio, width, height)
    counter = LineCounter(orientation, line_position, count_once_per_direction)
    model.predictor = None
    device = 0 if torch.cuda.is_available() else "cpu"
    stream = model.track(
        source=str(source_path),
        stream=True,
        persist=False,
        tracker="bytetrack.yaml",
        conf=confidence,
        imgsz=640,
        device=device,
        verbose=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(temporary),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("İşlenmiş video yazılamadı.")

    frame_rows: list[dict] = []
    crossing_rows: list[dict] = []
    instant_peak = 0
    try:
        for frame_index, result in enumerate(stream):
            frame = result.orig_img
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            detections = _detections_from_result(result)
            instant_count = len(detections)
            instant_peak = max(instant_peak, instant_count)
            for detection in detections:
                direction = counter.update(detection.track_id, detection.anchor)
                if direction is None:
                    continue
                crossing_rows.append(
                    {
                        "Zaman (sn)": round(frame_index / fps, 2),
                        "Kare": frame_index,
                        "Takip no": detection.track_id,
                        "Sınıf": class_label(detection.class_name),
                        "Yön": DIRECTION_NAMES_TR[direction],
                        "Güven": round(detection.confidence, 3),
                    }
                )
            annotated = _annotate(frame, detections, orientation, line_position, instant_count, counter.total)
            writer.write(annotated)
            frame_rows.append(
                {
                    "Zaman (sn)": round(frame_index / fps, 2),
                    "Kare": frame_index,
                    "Anlık araç": instant_count,
                    "Toplam geçen": counter.total,
                }
            )
            if progress_callback is not None:
                progress_callback(frame_index + 1, reported_frames)
    finally:
        writer.release()

    if not frame_rows:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Videodan kare okunamadı.")
    _transcode_for_browser(temporary, output_path)
    temporary.unlink(missing_ok=True)

    return AnalysisResult(
        output_video=output_path,
        frame_stats=pd.DataFrame(frame_rows),
        crossings=pd.DataFrame(crossing_rows),
        instant_peak=instant_peak,
        total_crossings=counter.total,
        fps=fps,
        frame_count=len(frame_rows),
    )
