"""Empty frame detection and filtering."""

import cv2
import numpy as np


def compute_variance(images: np.ndarray) -> np.ndarray:
    """Compute variance for each image in batch. Shape: (N, H, W) -> (N,)"""
    return np.var(images, axis=(1, 2), dtype=np.float32)


def compute_gradient_energy(images: np.ndarray) -> np.ndarray:
    """Compute gradient energy for each image using Sobel filters."""
    energies = np.zeros(len(images), dtype=np.float32)
    for i, img in enumerate(images):
        sobelx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        energies[i] = np.sum(np.sqrt(sobelx**2 + sobely**2))
    return energies


def compute_entropy(images: np.ndarray) -> np.ndarray:
    """Compute histogram entropy for each image."""
    entropies = np.zeros(len(images), dtype=np.float32)
    for i, img in enumerate(images):
        hist, _ = np.histogram(img, bins=256, range=(0, 255))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropies[i] = -np.sum(hist * np.log2(hist))
    return entropies


def detect_empty_frames(
    images: np.ndarray,
    variance_threshold: float = 100.0,
    gradient_threshold: float = 10.0,
    entropy_threshold: float = 3.0,
) -> np.ndarray:
    """Detect empty/invalid frames. Returns boolean mask where True = empty frame."""
    variance = compute_variance(images)
    gradient = compute_gradient_energy(images)
    entropy = compute_entropy(images)

    is_empty = (
        (variance < variance_threshold)
        | (gradient < gradient_threshold)
        | (entropy < entropy_threshold)
    )
    return is_empty
