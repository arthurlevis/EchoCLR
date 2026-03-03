"""Image processing utilities for ultrasound preprocessing."""

import cv2
import numpy as np


def compute_sector_transform_coords(height: int, width: int, sector_half_angle_deg: float = 35.0):
    """Compute coordinate mapping for polar-to-Cartesian sector transform."""
    ref_x = width // 2
    ref_y = 0
    theta_rad = np.radians(sector_half_angle_deg)

    r_idx = np.arange(height)[:, np.newaxis]
    theta_idx = np.arange(width)[np.newaxis, :]

    angle = -theta_rad + (2 * theta_rad) * (theta_idx / (width - 1))

    x = (ref_x + r_idx * np.sin(angle)).astype(int)
    y = (ref_y + r_idx * np.cos(angle)).astype(int)

    valid_mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    valid_r_idx, valid_theta_idx = np.where(valid_mask)
    valid_x = x[valid_mask]
    valid_y = y[valid_mask]

    return valid_r_idx, valid_theta_idx, valid_x, valid_y


def map_ultrasound_sector(img: np.ndarray, sector_half_angle_deg: float = 35.0) -> np.ndarray:
    """Map polar ultrasound sector to Cartesian plane."""
    height, width = img.shape[:2]
    polar_img = np.zeros_like(img)

    valid_r_idx, valid_theta_idx, valid_x, valid_y = compute_sector_transform_coords(
        height, width, sector_half_angle_deg
    )
    polar_img[valid_r_idx, valid_theta_idx] = img[valid_y, valid_x]

    return polar_img


def map_ultrasound_sectors_batch(images: np.ndarray, sector_half_angle_deg: float = 35.0) -> np.ndarray:
    """Map batch of polar ultrasound sectors to Cartesian plane (vectorized)."""
    height, width = images.shape[1:3]
    polar_imgs = np.zeros_like(images)

    valid_r_idx, valid_theta_idx, valid_x, valid_y = compute_sector_transform_coords(
        height, width, sector_half_angle_deg
    )
    polar_imgs[:, valid_r_idx, valid_theta_idx] = images[:, valid_y, valid_x]

    return polar_imgs


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize batch of frames using cv2.INTER_AREA."""
    resized = np.zeros((frames.shape[0], size[0], size[1]), dtype=frames.dtype)
    for i, frame in enumerate(frames):
        resized[i] = cv2.resize(frame, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return resized
