#!/usr/bin/env python3
"""
Prepare EchoCLR dataset from nov_dec_data_collection folder structure.

This script:
- Extracts only PLAX videos from robot_scan/plax folders
- Includes all rosbag folders for each patient (not just latest)
- Converts MP4 to AVI format using FFmpeg with MJPEG codec
- Creates CSV files with train/val/test splits at patient level
"""

import argparse
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger
from tqdm import tqdm


def get_all_rosbag_folders(plax_path: Path) -> List[Path]:
    """
    Find all rosbag folders in plax directory.

    Folder format: rosbag_YYYYMMDD_HHMMSS/

    Args:
        plax_path: Path to robot_scan/plax/ folder

    Returns:
        List of rosbag folder paths, sorted by timestamp (oldest to newest)
    """
    if not plax_path.exists():
        return []

    # Pattern: rosbag_YYYYMMDD_HHMMSS or rosbag_transition_to_*_YYYYMMDD_HHMMSS
    rosbag_pattern = re.compile(r"rosbag.*_(\d{8}_\d{6})")

    rosbag_folders = []
    for item in plax_path.iterdir():
        if item.is_dir():
            match = rosbag_pattern.match(item.name)
            if match:
                timestamp = match.group(1)
                rosbag_folders.append((timestamp, item))

    if not rosbag_folders:
        return []

    # Sort by timestamp (YYYYMMDD_HHMMSS format sorts lexicographically)
    rosbag_folders.sort(key=lambda x: x[0])

    return [folder for _, folder in rosbag_folders]


def convert_mp4_to_avi(input_path: Path, output_path: Path) -> bool:
    """
    Convert MP4 to AVI using FFmpeg (10-100x faster than OpenCV).

    Args:
        input_path: Path to input .mp4 file
        output_path: Path to output .avi file

    Returns:
        True if successful, False otherwise
    """
    try:
        # FFmpeg command: MJPEG codec, quality 2 (high quality), multi-threaded
        cmd = [
            "ffmpeg",
            "-i",
            str(input_path),
            "-c:v",
            "mjpeg",
            "-q:v",
            "2",  # Quality (2-5 is good, lower = better)
            "-y",  # Overwrite output
            "-loglevel",
            "error",  # Only show errors
            str(output_path),
        ]

        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed for {input_path}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error converting {input_path}: {e}")
        return False


def convert_single_video(task: Tuple[Path, Path]) -> Tuple[bool, Path]:
    """
    Worker function for parallel video conversion.

    Args:
        task: Tuple of (input_path, output_path)

    Returns:
        Tuple of (success, input_path)
    """
    input_path, output_path = task
    success = convert_mp4_to_avi(input_path, output_path)
    return success, input_path


def find_plax_videos(data_root: Path) -> List[Dict[str, str]]:
    """
    Find all PLAX videos from robot_scan/plax folders.

    Args:
        data_root: Path to annotation_video/nov_dec_data_collection/

    Returns:
        List of dicts with keys: patient_id, rosbag_folder, video_path
    """
    videos = []

    # Iterate through patient directories
    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        # Look for robot_scan/plax folder
        plax_path = patient_dir / "robot_scan" / "plax"

        if not plax_path.exists():
            logger.warning(f"No PLAX folder for patient {patient_id}")
            continue

        # Get all rosbag folders
        rosbag_folders = get_all_rosbag_folders(plax_path)

        if not rosbag_folders:
            logger.warning(f"No valid rosbag folders in {plax_path}")
            continue

        # Process all rosbag folders (not just latest)
        for rosbag_folder in rosbag_folders:
            # Find all .mp4 files in rosbag folder (excluding _recovered files)
            mp4_files = [f for f in rosbag_folder.glob("*.mp4") if "_recovered" not in f.name]

            if not mp4_files:
                logger.warning(f"No MP4 files in {rosbag_folder}")
                continue

            for mp4_file in mp4_files:
                videos.append(
                    {
                        "patient_id": patient_id,
                        "rosbag_folder": rosbag_folder.name,
                        "video_path": mp4_file,
                    }
                )

    return videos


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
):
    """
    Prepare EchoCLR dataset from nov_dec_data_collection structure.

    Args:
        input_dir: Path to annotation_video/nov_dec_data_collection/
        output_dir: Path to output data_dir
        train_ratio: Fraction of patients for training
        val_ratio: Fraction of patients for validation
        test_ratio: Fraction of patients for testing
        num_workers: Number of parallel workers for video conversion
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Train/val/test ratios must sum to 1.0"
    )

    # Create output directory structure
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Split ratios - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}")

    # Find all PLAX videos
    logger.info("Scanning for PLAX videos...")
    plax_videos = find_plax_videos(input_dir)

    if not plax_videos:
        logger.error("No PLAX videos found!")
        return

    logger.info(
        f"Found {len(plax_videos)} PLAX videos from {len(set(v['patient_id'] for v in plax_videos))} patients"
    )

    # Group by patient ID
    patient_videos = {}
    for video_info in plax_videos:
        patient_id = video_info["patient_id"]
        if patient_id not in patient_videos:
            patient_videos[patient_id] = []
        patient_videos[patient_id].append(video_info)

    # Split patients into train/val/test
    patient_ids = sorted(patient_videos.keys())
    n_patients = len(patient_ids)

    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train : n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val :])

    logger.info(f"Patient split:")
    logger.info(f"   Train: {len(train_patients)} patients")
    logger.info(f"   Val:   {len(val_patients)} patients")
    logger.info(f"   Test:  {len(test_patients)} patients")

    logger.info("Using FFmpeg for fast parallel conversion")

    # Collect all conversion tasks
    conversion_tasks = []
    video_metadata = []  # Store metadata for CSV creation

    for patient_id, videos in patient_videos.items():
        # Determine split
        if patient_id in train_patients:
            split = "train"
        elif patient_id in val_patients:
            split = "val"
        else:
            split = "test"

        # Process each video for this patient
        for video_idx, video_info in enumerate(videos):
            mp4_path = video_info["video_path"]
            avi_filename = f"{patient_id}_{video_info['rosbag_folder']}_{video_idx}.avi"
            avi_path = videos_dir / avi_filename

            conversion_tasks.append((mp4_path, avi_path))
            video_metadata.append(
                {
                    "split": split,
                    "fpath": avi_filename,
                    "acc_num": patient_id,
                    "plax_prob": 1.0,
                    "video_num": video_idx,
                    "label": 0,
                }
            )

    # Convert videos in parallel
    logger.info(
        f"Converting {len(conversion_tasks)} videos using {num_workers} parallel workers..."
    )

    records = {"train": [], "val": [], "test": []}
    failed_conversions = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(convert_single_video, task): idx
            for idx, task in enumerate(conversion_tasks)
        }

        # Process results with progress bar
        with tqdm(total=len(conversion_tasks), desc="Converting videos", unit="video") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                success, input_path = future.result()

                if success:
                    # Add to appropriate split
                    metadata = video_metadata[idx]
                    records[metadata["split"]].append(metadata)
                else:
                    failed_conversions.append(input_path)

                pbar.update(1)

    if failed_conversions:
        logger.warning(f"Failed to convert {len(failed_conversions)} videos")
        for path in failed_conversions[:10]:
            logger.warning(f"  - {path}")
        if len(failed_conversions) > 10:
            logger.warning(f"  ... and {len(failed_conversions) - 10} more")

    # Create CSV files
    logger.info("Creating CSV files...")
    for split in ["train", "val", "test"]:
        if records[split]:
            df = pd.DataFrame(records[split])
            csv_path = output_dir / f"{split}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"   {split}.csv: {len(df)} videos from {df['acc_num'].nunique()} patients")
        else:
            logger.warning(f"   {split}.csv: No videos (empty split)")

    logger.info("Dataset preparation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"   - videos/: {len(list(videos_dir.glob('*.avi')))} AVI files")
    logger.info(f"   - train.csv, val.csv, test.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare EchoCLR dataset from nov_dec_data_collection"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Path to annotation_video/nov_dec_data_collection/ folder",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Path to output data_dir (will create videos/ and CSV files)",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.75,
        help="Fraction of patients for training (default: 0.75)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Fraction of patients for validation (default: 0.10)",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction of patients for testing (default: 0.15)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of parallel workers for video conversion (default: 4)",
    )

    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
