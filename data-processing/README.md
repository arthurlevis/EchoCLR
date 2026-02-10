# EchoCLR Data Processing

Scripts to prepare the EchoCLR dataset from `nov_dec_data_collection` video data (as saved on AWS S3).

## Overview

The `prepare_dataset.py` script:
- ✅ Extracts **only PLAX videos** from `robot_scan/plax/` folders
- ✅ Takes the **latest rosbag** if multiple exist per patient
- ✅ Converts **MP4 → AVI** (MJPEG codec for fast loading)
- ✅ Uses **patient IDs** (e.g., `bold-meals-travel`) as `acc_num`
- ✅ Creates **train/val/test splits** at patient level
- ✅ Generates CSV files compatible with EchoCLR dataset format
- ❌ **Ignores all `manual_scan` folders**

### Expected Input Structure

```
annotation_video/nov_dec_data_collection/
├── bold-meals-travel/
│   ├── manual_scan/              # IGNORED
│   └── robot_scan/
│       ├── plax/                 # ✅ EXTRACTED
│       │   ├── rosbag_20251201_144645/
│       │   │   └── rosbag_20251201_144645_0.mp4
│       │   └── rosbag_20251201_145107/  # ← Latest, will be used
│       │       └── rosbag_20251201_145107_0.mp4
│       ├── apical/               # IGNORED
│       └── psax/                 # IGNORED
├── brave-spies-smell/
│   └── robot_scan/
│       └── plax/
│           └── rosbag_20251126_150022/
│               └── rosbag_20251126_150022_0.mp4
└── ...
```

### Output Structure

```
data_dir/
├── videos/
│   ├── bold-meals-travel_rosbag_20251201_145107_0.avi
│   ├── brave-spies-smell_rosbag_20251126_150022_0.avi
│   └── ...
├── train.csv
├── val.csv
└── test.csv
```

## Quick Start

### 1. Verify Data Structure (Recommended)

Before processing, run the verification script to check your data:

```bash
uv run python data-processing/check_video_folder.py \
    /path/to/annotation_video/nov_dec_data_collection
```

This will show:
- ✅ Number of valid patients with PLAX videos
- ⚠️ Patients missing PLAX folders
- 🔄 Patients with multiple rosbag folders (shows which will be used)
- 📊 Total videos that will be processed
- 🔗 Multi-instance learning statistics

### 2. Process Dataset

#### Basic Usage

```bash
uv run python data-processing/prepare_dataset.py \
    --input_dir /path/to/annotation_video/nov_dec_data_collection \
    --output_dir data/echoclr_dataset
```

#### Custom Number of Workers (Multiprocessing)

```bash
uv run python data-processing/prepare_dataset.py \
    --input_dir /path/to/annotation_video/nov_dec_data_collection \
    --output_dir data/echoclr_dataset \
    --num_workers 8
```

Recommended number of workers: `Cores per socket × Sockets` :
```bash
lscpu | grep "Core(s) per socket"
lscpu | grep "Socket(s)"
```

## CSV Format

Each CSV file contains:

| Column      | Description                   | Example                                          |
| ----------- | ----------------------------- | ------------------------------------------------ |
| `fpath`     | Video filename                | `bold-meals-travel_rosbag_20251201_145107_0.avi` |
| `acc_num`   | Patient ID (accession number) | `bold-meals-travel`                              |
| `plax_prob` | PLAX confidence (dummy = 1.0) | `1.0`                                            |
| `video_num` | Video index for patient       | `0`                                              |
| `label`     | Placeholder for SSL (unused)  | `0`                                              |

## Multi-Instance Learning

Patients with **multiple PLAX videos** in the latest rosbag folder will have multiple rows with the same `acc_num`. EchoCLR will use these as positive pairs for contrastive learning.

Example:
```csv
fpath,acc_num,plax_prob,video_num,label
patient1_rosbag_20251201_120000_0.avi,patient1,1.0,0,0
patient1_rosbag_20251201_120000_1.avi,patient1,1.0,1,0  # Same patient!
patient2_rosbag_20251202_130000_0.avi,patient2,1.0,0,0
```

## Video Conversion

Videos are converted from MP4 to AVI using **FFmpeg** with **MJPEG codec**:
- Fast parallel conversion using all CPU cores
- MJPEG provides fast decoding during training (important for data loading performance)
- Files are ~50% larger than MP4, but this is a worthwhile trade-off for training speed
- Quality setting: `-q:v 2` (high quality)

## Requirements

### System Requirements

Install FFmpeg (required for video conversion):

```bash
sudo apt install ffmpeg
```

### Python Dependencies

Dependencies are managed via `pyproject.toml`:

```bash
cd /home/arthur/dev/EchoCLR
uv sync 
```
