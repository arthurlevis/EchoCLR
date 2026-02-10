#!/usr/bin/env python3
"""
Verify structure of the videos folder before processing.

This script performs a dry-run to check:
- Number of patients with PLAX videos
- Multiple rosbag folders per patient (all will be used)
- Total videos that will be processed (from all rosbag folders)
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict

from loguru import logger


def verify_structure(data_root: Path):
    """Verify and report on data structure."""

    if not data_root.exists():
        logger.error(f"Directory not found: {data_root}")
        return

    logger.info(f"Scanning: {data_root}")

    patient_stats = {}
    total_videos = 0
    patients_with_multiple_rosbags = []

    # Iterate through patient directories
    for patient_dir in sorted(data_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        # Look for robot_scan/plax folder
        plax_path = patient_dir / "robot_scan" / "plax"

        if not plax_path.exists():
            patient_stats[patient_id] = {"status": "NO_PLAX", "rosbag_folders": [], "videos": 0}
            continue

        # Find all rosbag folders
        rosbag_pattern = re.compile(r"rosbag.*_(\d{8}_\d{6})")
        rosbag_folders = []

        for item in plax_path.iterdir():
            if item.is_dir():
                match = rosbag_pattern.match(item.name)
                if match:
                    timestamp = match.group(1)
                    # Count non-recovered MP4 files
                    mp4_files = [f for f in item.glob("*.mp4") if "_recovered" not in f.name]
                    rosbag_folders.append(
                        {"name": item.name, "timestamp": timestamp, "num_videos": len(mp4_files)}
                    )

        if not rosbag_folders:
            patient_stats[patient_id] = {"status": "NO_ROSBAG", "rosbag_folders": [], "videos": 0}
            continue

        # Sort by timestamp
        rosbag_folders.sort(key=lambda x: x["timestamp"])

        # Count all videos from all rosbag folders
        total_patient_videos = sum(rb["num_videos"] for rb in rosbag_folders)

        if len(rosbag_folders) > 1:
            patients_with_multiple_rosbags.append((patient_id, rosbag_folders))

        patient_stats[patient_id] = {
            "status": "OK",
            "rosbag_folders": rosbag_folders,
            "videos": total_patient_videos,
        }

        total_videos += total_patient_videos

    # Print summary
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    ok_patients = [p for p, s in patient_stats.items() if s["status"] == "OK"]
    no_plax = [p for p, s in patient_stats.items() if s["status"] == "NO_PLAX"]
    no_rosbag = [p for p, s in patient_stats.items() if s["status"] == "NO_ROSBAG"]

    logger.info(f"Valid patients: {len(ok_patients)}")
    logger.info(f"Patients without PLAX folder: {len(no_plax)}")
    logger.info(f"Patients with PLAX but no rosbag: {len(no_rosbag)}")
    logger.info(f"Total PLAX videos to process: {total_videos}")

    # Patients with multiple rosbag folders
    if patients_with_multiple_rosbags:
        logger.info(f"Patients with multiple rosbag folders: {len(patients_with_multiple_rosbags)}")

    # Show patients without PLAX
    if no_plax:
        logger.warning(f"Patients without robot_scan/plax folder:")
        for patient_id in no_plax[:10]:
            logger.warning(f"      - {patient_id}")
        if len(no_plax) > 10:
            logger.warning(f"      ... and {len(no_plax) - 10} more")

    # Multi-instance learning statistics
    multi_video_patients = [
        p for p, s in patient_stats.items() if s["status"] == "OK" and s["videos"] > 1
    ]
    if multi_video_patients:
        video_counts = defaultdict(int)
        for p in multi_video_patients:
            count = patient_stats[p]["videos"]
            video_counts[count] += 1

    logger.info("=" * 80)
    logger.info("Structure verification complete!")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Verify nov_dec_data_collection structure before processing"
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Path to annotation_video/nov_dec_data_collection/ folder",
    )

    args = parser.parse_args()

    verify_structure(args.input_dir)


if __name__ == "__main__":
    main()
