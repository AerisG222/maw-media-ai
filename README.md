# Photo Tagger

Local facial recognition tool that scans a photo library, detects faces, matches them against known people, and saves results to a JSON file. Everything runs locally — no cloud APIs, your photos never leave your machine.

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
enroll.py      # enroll command implementation
scan.py        # scan command implementation
report.py      # report command implementation
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
- Install all required Python packages

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

### Commands

#### `enroll` — Register known people

Point the tool at a folder of reference photos, one sub-folder per person:

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

#### `scan` — Scan your photo library

```bash
./pt.py scan --photos ./my_photos --db faces.db --output results.json
```

To skip photos already processed in a previous run:

```bash
./pt.py scan --photos ./my_photos --db faces.db --output results.json --skip-processed
```

| Argument | Default | Description |
|---|---|---|
| `--photos` | required | Folder of photos to scan (searched recursively) |
| `--db` | `faces.db` | Path to the face database file |
| `--output` | `results.json` | Output JSON file |
| `--skip-processed` | off | Skip photos already in the database cache |

---

#### `report` — View a summary of results

```bash
./pt.py report --output results.json
```

| Argument | Default | Description |
|---|---|---|
| `--output` | `results.json` | JSON results file to summarise |

---

## Output Format

Results are written to a JSON file. Each detected face is a separate entry:

```json
{
  "file_path": "/home/user/photos/birthday.jpg",
  "file_name": "birthday.jpg",
  "face_index": 0,
  "matched_name": "Alice",
  "confidence": 87.3,
  "distance": 0.127,
  "face_region": { "x": 142, "y": 88, "w": 210, "h": 210 },
  "scanned_at": "2026-03-07T14:22:01",
  "notes": ""
}
```

| Field | Description |
|---|---|
| `file_path` | Full path to the photo |
| `file_name` | Filename only |
| `face_index` | Index of this face within the photo (0-based) |
| `matched_name` | Matched person's name, `Unknown`, `No face detected`, or `ERROR` |
| `confidence` | Match confidence as a percentage (higher = better) |
| `distance` | Raw cosine distance (lower = closer match) |
| `face_region` | Bounding box of the face in pixels |
| `scanned_at` | ISO 8601 timestamp |
| `notes` | Error message if something went wrong |

---

## GPU Acceleration

The tool automatically detects and uses your GPU at startup. No configuration needed.

| Hardware | Approximate speed |
|---|---|
| CPU only | 2–5 photos/min |
| NVIDIA GPU (mid-range) | 30–60 photos/min |
| NVIDIA GPU (high-end) | 100+ photos/min |

For NVIDIA GPUs, `prep.sh` installs `cudatoolkit` and `cuDNN` via conda, which handles driver compatibility automatically.

---

## Tuning Accuracy

The matching threshold is defined in `common.py`:

```python
THRESHOLD = 0.40  # Lower = stricter matching (0.0–1.0)
```

- **Lower the threshold** (e.g. `0.30`) to reduce false positives — more faces will be tagged as `Unknown`
- **Raise the threshold** (e.g. `0.50`) to catch more matches — at the risk of occasional incorrect tags

The default model is **ArcFace**, which handles variation in lighting, age, and angle well. The detector is **RetinaFace**, which is accurate but slower. You can change both in `common.py` if needed.

---

## License

MIT
