# EchoCLR Data Processing

Scripts to prepare the EchoCLR dataset from `nov_dec_data_collection` video data (as saved on AWS S3).

## Overview

The `prepare_dataset.py` script:
- ✅ Extracts **only PLAX videos** from `robot_scan/plax/` folders
- ✅ Includes **all rosbag folders** for each patient (not just latest)
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
│       ├── plax/                 # ✅ ALL VIDEOS EXTRACTED
│       │   ├── rosbag_20251201_144645/
│       │   │   └── rosbag_20251201_144645_0.mp4  # ✅ Included
│       │   └── rosbag_20251201_145107/
│       │       └── rosbag_20251201_145107_0.mp4  # ✅ Included
│       ├── apical/               # IGNORED
│       └── psax/                 # IGNORED
├── brave-spies-smell/
│   └── robot_scan/
│       └── plax/
│           └── rosbag_20251126_150022/
│               └── rosbag_20251126_150022_0.mp4  # ✅ Included
└── ...
```

### Output Structure

```
data_dir/
├── videos/
│   ├── bold-meals-travel_rosbag_20251201_144645_0.avi  # video_num=0
│   ├── bold-meals-travel_rosbag_20251201_145107_0.avi  # video_num=1
│   ├── brave-spies-smell_rosbag_20251126_150022_0.avi  # video_num=0
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
- 🔄 Patients with multiple rosbag folders (all will be used)
- 📊 Total videos that will be processed
- 🔗 Multi-instance learning statistics

### 2. Process Dataset

#### Basic Usage

```bash
uv run python data-processing/prepare_dataset.py \
    --input_dir /path/to/annotation_video/nov_dec_data_collection \
    --output_dir data/echoclr_dataset
```

#### Custom Split Ratios

```bash
uv run python data-processing/prepare_dataset.py \
    --input_dir /path/to/annotation_video/nov_dec_data_collection \
    --output_dir data/echoclr_dataset \
    --train_ratio 0.80 \
    --val_ratio 0.10 \
    --test_ratio 0.10
```

#### Parallel Workers

```bash
uv run python data-processing/prepare_dataset.py \
    --input_dir /path/to/annotation_video/nov_dec_data_collection \
    --output_dir data/echoclr_dataset \
    --num_workers 8  # Use 8 parallel workers (default: 4)
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

Patients with **multiple PLAX videos** (across all rosbag folders) will have multiple rows with the same `acc_num`. EchoCLR will use these as positive pairs for contrastive learning.

This includes:
- Multiple videos within a single rosbag folder
- Videos from different rosbag folders for the same patient

Example:
```csv
fpath,acc_num,plax_prob,video_num,label
patient1_rosbag_20251201_120000_0.avi,patient1,1.0,0,0
patient1_rosbag_20251201_120000_1.avi,patient1,1.0,1,0  # Same patient, same rosbag
patient1_rosbag_20251201_150000_0.avi,patient1,1.0,2,0  # Same patient, different rosbag
patient2_rosbag_20251202_130000_0.avi,patient2,1.0,0,0
```

## Video Conversion

Videos are converted from MP4 to AVI using **FFmpeg** with **MJPEG codec**:
- Fast parallel conversion using configurable number of workers (default: 4)
- MJPEG provides fast decoding during training (important for data loading performance)
- Files are ~50% larger than MP4, but this is a worthwhile trade-off for training speed
- Quality setting: `-q:v 2` (high quality)

### Why CPU-only (no GPU encoding)?

GPU encoding (NVENC) only supports H.264/HEVC, not MJPEG. Since we specifically use MJPEG for fast training-time decoding, GPU encoding would defeat the purpose. CPU-based FFmpeg with MJPEG is the optimal choice for this workflow.

### Performance Tuning

- **Default**: 4 workers balances speed and system load
- **More workers**: Use `--num_workers 8` or higher if you have many CPU cores and fast I/O
- **Fewer workers**: Use `--num_workers 2` if system is under heavy load or I/O is slow
- **Rule of thumb**: Set to number of physical CPU cores for best performance

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
uv sync  # Install dependencies
```

Required packages:
- `pandas` (CSV generation)
- `tqdm` (progress bars)
- `loguru` (logging)

## Troubleshooting

### No PLAX videos found

**Issue:** Script reports "No PLAX videos found"

**Solution:** Verify:
1. Input directory path is correct
2. Structure matches expected format (`patient/robot_scan/plax/rosbag_*/`)
3. Videos are `.mp4` files (not `.avi` already)

### Conversion errors

**Issue:** FFmpeg conversion fails

**Solution:**
1. Verify FFmpeg is installed: `which ffmpeg`
2. Check FFmpeg can read the MP4: `ffmpeg -i video.mp4`
3. Verify source video is not corrupted
4. Check disk space (AVI files are ~50% larger than MP4)

### Missing patients in splits

**Issue:** Some patients not appearing in train/val/test

**Solution:**
- Patients without valid `robot_scan/plax/rosbag_*/` folders are skipped
- Check warnings during execution: `⚠️  No PLAX folder for patient X`
