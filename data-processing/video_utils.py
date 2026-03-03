"""Video I/O utilities for EchoCLR dataset creation."""

import cv2
import numpy as np
from pathlib import Path


def save_video(frames: np.ndarray, output_path: str, fps: float = 30.0) -> None:
    """Save frames as AVI video using MJPEG codec.

    Args:
        frames: Array of shape (num_frames, height, width) or (num_frames, height, width, channels)
        output_path: Output video path
        fps: Frames per second
    """
    if frames.ndim == 3:
        frames = np.expand_dims(frames, axis=-1)

    num_frames, height, width, channels = frames.shape

    # Convert grayscale to BGR for OpenCV
    if channels == 1:
        frames_bgr = np.repeat(frames, 3, axis=-1)
    else:
        frames_bgr = frames

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames_bgr:
        out.write(frame.astype(np.uint8))

    out.release()


def load_video(fpath: str) -> np.ndarray:
    """Load video file as numpy array.

    Args:
        fpath: Path to video file

    Returns:
        Array of shape (num_frames, height, width, channels)
    """
    if not Path(fpath).exists():
        raise FileNotFoundError(fpath)

    capture = cv2.VideoCapture(fpath)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for i in range(frame_count):
        ret, frame = capture.read()
        if ret:
            v[i] = frame

    capture.release()
    return v
