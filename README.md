# Photo Tagger

Local photo scanning tool that detects faces, objects, and scenes in your photo library using on-device AI. Results are saved to a JSON file. Everything runs locally — no cloud APIs, your photos never leave your machine.

---

## Requirements

- Linux or macOS
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- NVIDIA GPU (optional, but recommended for large libraries)

---

## Project Structure

```
prep.sh        # conda environment setup script
pt.py          # main entry point — run all commands through this
common.py      # shared utilities, constants, GPU config (not run directly)
enroll.py      # enroll command — register known people for face matching
list.py        # list command — preview which files would be scanned
scan.py        # scan command — detect faces, objects, and/or scenes
objects.py     # object detection module (YOLO11, COCO dataset)
scenes.py      # scene classification module (YOLO11, ImageNet dataset)
report.py      # report command — summarise results
```

---

## Setup

**1. Run the prep script to create the conda environment and install dependencies:**

```bash
chmod +x prep.sh pt.py
./prep.sh
```

This will:
- Create a conda environment named `photo-tagger` with Python 3.12
- Detect your GPU and install CUDA/cuDNN automatically if an NVIDIA GPU is found
- Install all required Python packages including DeepFace and Ultralytics YOLO

**2. Activate the environment:**

```bash
conda activate photo-tagger
```

**3. Deactivate when done:**

```bash
conda deactivate
```

---

## Usage

All commands are run through `pt.py`:

```bash
./pt.py <command> [options]
```

---

### `list` — Preview which files would be scanned

Run this before scanning to verify the right images are included — and nothing you want to skip is accidentally picked up.

```bash
# List all images in a folder
./pt.py list --photos ./my_photos

# Limit to specific file types
./pt.py list --photos ./my_photos --ext jpg png heic

# Save the list to a text file for review
./pt.py list --photos ./my_photos --output file-list.txt
```

| Argument | Default | Description |
|---|---|---|
| `--photos` | required | Folder to inspect (searched recursively) |
| `--ext` | all supported | Limit to specific extensions e.g. `jpg png heic` |
| `--output` | none | Save the file list to a text file |

Supported extensions: `.jpg` `.jpeg` `.png` `.heic` `.bmp` `.tiff` `.webp`

The output is grouped by sub-folder so it's easy to spot unexpected directories being included:

```
  📁 my_photos/
     birthday.jpg
     holiday.png

  📁 my_photos/2024/
     beach.jpg
     concert.heic
```

---

### `enroll` — Register known people for face matching

Required before running face scans. Point it at a folder of reference photos, one sub-folder per person:

```
known_people/
    Alice/
        photo1.jpg
        photo2.jpg
    Bob/
        photo1.jpg
```

```bash
./pt.py enroll --known ./known_people --db faces.db
```

| Argument | Default | Description |
|---|---|---|
| `--known` | required | Folder containing one sub-folder per person |
| `--db` | `faces.db` | Path to the face database file |

**Tips for good reference photos:**
- Use 5–10 photos per person
- Include variety: different lighting, slight angles, with/without glasses
- Use clear, solo portraits — no group photos
- Faces should be at least 150×150 pixels

---

### `scan` — Scan your photo library

```bash
# Scan for everything (default)
./pt.py scan --photos ./my_photos --db faces.db --output results.json

# Scan for specific types only
./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types faces
./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types objects
./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types scenes
./pt.py scan --photos ./my_photos --db faces.db --output results.json --scan-types faces objects

# Skip already-processed photos
./pt.py scan --photos ./my_photos --db faces.db --output results.json --skip-processed
```

| Argument | Default | Description |
|---|---|---|
| `--photos` | required | Folder of photos to scan (searched recursively) |
| `--db` | `faces.db` | Path to the face database file |
| `--output` | `results.json` | Output JSON file |
| `--scan-types` | `faces objects scenes` | Which scan types to run |
| `--skip-processed` | off | Skip photos already in the database cache |

#### Scan types

| Type | Model | What it detects |
|---|---|---|
| `faces` | DeepFace + ArcFace | Identifies people by matching against enrolled reference photos |
| `objects` | YOLO11 (COCO) | Detects and locates 80 common objects: people, cars, dogs, chairs, etc. |
| `scenes` | YOLO11 (ImageNet) | Classifies the overall scene into 1000 categories: beach, kitchen, forest, etc. |

> **Note:** YOLO models (~6MB each) are downloaded automatically from Ultralytics on first use.

---

### `report` — View a summary of results

```bash
# Full summary of all scan types
./pt.py report --output results.json

# Summary for a specific scan type only
./pt.py report --output results.json --type faces
./pt.py report --output results.json --type objects
./pt.py report --output results.json --type scenes
```

| Argument | Default | Description |
|---|---|---|
| `--output` | `results.json` | JSON results file to summarise |
| `--type` | `all` | Which scan type to report: `all`, `faces`, `objects`, `scenes` |

---

## Output Format

All results are written to a single JSON file as a list. Each entry has a `scan_type` field identifying which scanner produced it.

### Face entry
```json
{
  "file_path": "/home/user/photos/birthday.jpg",
  "file_name": "birthday.jpg",
  "scan_type": "faces",
  "face_index": 0,
  "matched_name": "Alice",
  "confidence": 87.3,
  "distance": 0.127,
  "face_region": { "x": 142, "y": 88, "w": 210, "h": 210 },
  "scanned_at": "2026-03-07T14:22:01",
  "notes": ""
}
```

### Object entry
```json
{
  "file_path": "/home/user/photos/birthday.jpg",
  "file_name": "birthday.jpg",
  "scan_type": "objects",
  "object_count": 3,
  "labels": ["cake", "cup", "person"],
  "objects": [
    {
      "label": "cake",
      "confidence": 91.2,
      "bbox": { "x1": 210, "y1": 300, "x2": 480, "y2": 520, "w": 270, "h": 220 }
    }
  ],
  "scanned_at": "2026-03-07T14:22:01",
  "notes": ""
}
```

### Scene entry
```json
{
  "file_path": "/home/user/photos/birthday.jpg",
  "file_name": "birthday.jpg",
  "scan_type": "scenes",
  "top_scene": "dining_table",
  "scenes": [
    { "label": "dining_table", "confidence": 72.4 },
    { "label": "bakery",       "confidence": 14.1 },
    { "label": "restaurant",   "confidence": 8.3  }
  ],
  "scanned_at": "2026-03-07T14:22:01",
  "notes": ""
}
```

---

## GPU Acceleration

The tool automatically detects and uses your GPU at startup for all scan types.

| Hardware | Approximate speed |
|---|---|
| CPU only | 2–5 photos/min |
| NVIDIA GPU (mid-range) | 30–60 photos/min |
| NVIDIA GPU (high-end) | 100+ photos/min |

For NVIDIA GPUs, `prep.sh` installs `cudatoolkit` and `cuDNN` via conda, which handles driver compatibility automatically.

---

## Tuning Accuracy

### Face matching threshold
Defined in `common.py`:
```python
THRESHOLD = 0.40  # Lower = stricter (0.0–1.0)
```
- **Lower** (e.g. `0.30`) → fewer false positives, more `Unknown` results
- **Higher** (e.g. `0.50`) → more matches, higher risk of incorrect tags

### Object/scene confidence threshold
Also in `common.py`:
```python
YOLO_CONFIDENCE = 0.30  # Minimum confidence to include a detection
```
- **Lower** → more detections, including less certain ones
- **Higher** → only high-confidence detections are included

### Face recognition model
The default is **ArcFace** via **RetinaFace** detector. Both can be changed in `common.py`:
```python
MODEL_NAME = "ArcFace"    # alternatives: VGG-Face, Facenet, DeepFace
DETECTOR   = "retinaface" # alternatives: opencv, mtcnn, ssd
```
