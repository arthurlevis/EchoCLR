"""ECG peak detection utilities."""

import numpy as np
import neurokit2


def detect_r_peaks_neurokit(times: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """Detect R-peaks using neurokit2."""
    fs = 1.0 / np.median(np.diff(times))
    _, results = neurokit2.ecg_peaks(amplitudes, sampling_rate=fs)
    peak_indices = np.array(results["ECG_R_Peaks"]).astype(int)
    return times[peak_indices]


def compute_clip_indices_between_peaks(
    r_peak_times: np.ndarray,
    frame_timestamps: np.ndarray,
    subsample: int = 1,
) -> list[tuple[int, int]]:
    """Compute frame indices for clips between consecutive R-peaks.

    Args:
        r_peak_times: R-peak timestamps
        frame_timestamps: Frame timestamps
        subsample: Take every Nth frame (default: 1 = all frames)

    Returns:
        List of (start_idx, end_idx, subsample) tuples for each clip
    """
    clips = []

    for i in range(len(r_peak_times) - 1):
        start_time = r_peak_times[i]
        end_time = r_peak_times[i + 1]

        start_idx = int(np.argmin(np.abs(frame_timestamps - start_time)))
        end_idx = int(np.argmin(np.abs(frame_timestamps - end_time)))

        if end_idx > start_idx:
            clips.append((start_idx, end_idx, subsample))

    return clips


def validate_clip_bounds(
    clips: list[tuple[int, int, int]], total_frames: int
) -> list[tuple[int, int, int]]:
    """Filter clips that exceed dataset boundaries."""
    return [(s, e, sub) for s, e, sub in clips if s >= 0 and e <= total_frames]
