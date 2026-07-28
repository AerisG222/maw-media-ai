# Face Scanner

Detects and recognises faces in a local photo library, storing results in
Postgres (with pgvector) for querying from a .NET web application.

## Stack

| Component | Library |
|-----------|---------|
| Face detection + embedding | InsightFace `buffalo_l` |
| Clustering | HDBSCAN |
| Vector storage + search | pgvector (Postgres extension) |
| Database driver | psycopg3 |

---

## Prerequisites

- Python 3.11+
- [Podman](https://podman.io/docs/installation) (for the Postgres + pgvector container)
- `postgresql-client` for `psql`, used by the setup script to apply the schema (`sudo apt install postgresql-client`)
- (Optional) CUDA GPU for faster inference

### Install Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **GPU users:** `requirements.txt` already specifies `onnxruntime-gpu` and
> `opencv-python-headless`. Note that installing `insightface` pulls in the
> plain `onnxruntime` and `opencv-python` packages as dependencies, and those
> **shadow** the ones above — silently disabling CUDA. See the install-trap note
> in `requirements.txt`, or install from `requirements.lock` to get the resolved,
> working set.

---

## Setup

### 1. Start the Postgres + pgvector container

```bash
chmod +x setup-db.sh
./setup-db.sh
```

This will:
- Pull the official `pgvector/pgvector:pg16` image
- Create a named Podman volume (`face_scanner_pgdata`) for persistent storage
- Start the container, bound to `127.0.0.1:5433` (avoids conflicts with any local Postgres)
- Apply `schema.sql` automatically

The script prints the `FACE_SCANNER_DSN` export line when done.

#### Container management

```bash
./setup-db.sh start    # start after a reboot
./setup-db.sh stop     # stop (data is preserved)
./setup-db.sh psql     # open an interactive psql session
./setup-db.sh logs     # tail container logs
./setup-db.sh destroy  # stop + delete container AND all data
```

#### Configuration

Override any default via environment variable before running the script:

| Variable | Default | Description |
|----------|---------|-------------|
| `PG_PORT` | `5433` | Host port to bind |
| `PG_DB` | `face_scanner` | Database name |
| `PG_USER` | `face_scanner` | Postgres user |
| `PG_PASSWORD` | `face_scanner_secret` | Password — change this |
| `CONTAINER_NAME` | `face_scanner_db` | Podman container name |
| `VOLUME_NAME` | `face_scanner_pgdata` | Podman volume name |

```bash
PG_PASSWORD=mysecret PG_PORT=5434 ./setup-db.sh
```

### 2. Set the connection string

```bash
export FACE_SCANNER_DSN="postgresql://face_scanner:face_scanner_secret@localhost:5433/face_scanner"
```

Or edit the `DB_DSN` default at the top of `scan-faces.py`.

---

## Usage

The scanner is a single CLI with subcommands. Run any command with `-h` for
its full options. A typical pipeline is:

```
scan → cluster → (label some clusters in the UI) → suggest → merge-clusters
```

### Scan for faces

Detects faces in the photo library, stores embeddings, and caches a
display-sized crop of each face on disk for the web UI. Photos already in the
database are skipped automatically, so this is safe to re-run as new photos
arrive — no separate "incremental" flag is needed.

```bash
python scan-faces.py scan --photo-dir /path/to/photos
```

Image decoding is overlapped with inference on background threads. Tune with
`--workers N` (detection itself stays serial).

### Cluster into persons

Groups all stored embeddings into person clusters using HDBSCAN.
Run this after a scan, or after adding many new photos.

```bash
python scan-faces.py cluster
```

### Suggest assignments

After you've labelled some clusters (given them names), this finds unassigned
faces — and faces still sitting in unnamed clusters — whose nearest named
person is within a cosine-distance threshold, and records a suggestion for
each. Review and confirm them in the web UI's "Review Suggestions" view.

```bash
python scan-faces.py suggest
python scan-faces.py suggest --threshold 0.30   # lower = more conservative
```

### Merge clusters

Merges unnamed clusters into the nearest named person when their centroids are
close (tighter default threshold than `suggest`, since a whole cluster moves at
once). Preview first with `--dry-run`.

```bash
python scan-faces.py merge-clusters --dry-run
python scan-faces.py merge-clusters
python scan-faces.py merge-clusters --threshold 0.20
```

### Check stats

```bash
python scan-faces.py stats
```

---

## Labelling persons

After clustering, each person has a row in the `person` table with
`name = NULL`. You label them once, then let `suggest` / `merge-clusters`
fold the remaining unassigned faces into the named people.

**Recommended — the Streamlit web UI** (see below). It shows a grid of face
thumbnails per cluster, lets you name clusters, assign uncategorised faces
(sorted by similarity to a chosen person), merge similar clusters, and review
suggestions. This is by far the most efficient workflow for large libraries.

**Or SQL directly** (person `id` is a UUID):
```sql
-- Find the biggest unnamed clusters
SELECT per.id, per.face_count
FROM person_v per WHERE per.name IS NULL ORDER BY face_count DESC;

-- Name a person
UPDATE person SET name = 'Jane Smith'
WHERE id = '00000000-0000-0000-0000-000000000000';
```

Once some clusters are named, run `suggest` and/or `merge-clusters` (above) to
propagate those labels to the rest of the faces.

### Clusters you don't intend to name

Not every cluster is someone you want to label. Mark those in the web UI as
**Unknown** (a real person you're not naming) or **Not a person** (a bad
detection). The cluster is *kept*, so the faces stay grouped in case you
recognise them later — they are not merged into one big bucket.

A marked cluster is excluded from `cluster`, `suggest` and `merge-clusters`, so
it stops reappearing in your triage queue. Two consequences worth knowing:

- `cluster` will not delete or regroup a marked cluster, so re-running it is
  safe once you've started triaging.
- `suggest --include-unknown` re-evaluates them anyway, which is worth doing
  occasionally as the classifier improves.

The states live in the `person_status` table, so adding one is an `INSERT`
rather than a schema change, and the UI picks it up automatically.

---

## Viewing Clusters and Faces (Web UI)

A Streamlit web interface is provided to browse persons/clusters and their
faces, and to drive the whole labelling workflow: naming clusters, assigning
uncategorised faces (optionally sorted by similarity to a target person),
merging similar clusters, reviewing `suggest` results, and clearing unwanted
clusters.

### 1. Install Streamlit (if not already done)

```bash
pip install streamlit
```

### 2. Set the database connection

Make sure your `FACE_SCANNER_DSN` environment variable is set, as described above.

### 3. Run the viewer

```bash
streamlit run view-clusters-app.py
```

Then open the provided URL in your browser (usually http://localhost:8501).

---

## Querying from .NET

Install the packages:
```
dotnet add package Npgsql.EntityFrameworkCore.PostgreSQL
dotnet add package Pgvector
```

### Example queries

**All photos containing a named person:**
```sql
SELECT DISTINCT p.file_path
FROM photo p
JOIN face_detection fd ON fd.photo_id = p.id
JOIN person per        ON per.id = fd.person_id
WHERE per.name = 'Jane Smith';
```

**Everyone in a specific photo:**
```sql
SELECT per.name, fd.bounding_box, fd.detection_score
FROM face_detection fd
JOIN person per ON per.id = fd.person_id
WHERE fd.photo_id = $1
ORDER BY fd.detection_score DESC;
```

**Photos with multiple specific people (AND logic):**
```sql
SELECT p.file_path
FROM photo p
WHERE (
    SELECT COUNT(DISTINCT per.name)
    FROM face_detection fd
    JOIN person per ON per.id = fd.person_id
    WHERE fd.photo_id = p.id
      AND per.name IN ('Jane Smith', 'John Smith')
) = 2;
```

**Similarity search — find photos with a face similar to a given embedding:**
```sql
SELECT p.file_path, fd.detection_score,
       fd.embedding <=> $1::vector AS distance
FROM face_detection fd
JOIN photo p ON p.id = fd.photo_id
ORDER BY fd.embedding <=> $1::vector
LIMIT 20;
```

**Use the convenience view:**
```sql
SELECT * FROM photo_person_summary
WHERE people ? 'Jane Smith';
```

---

## Tuning accuracy

| Problem | Fix |
|---------|-----|
| One person split into many clusters | Lower `HDBSCAN_CLUSTER_THRESHOLD` (e.g. 0.5) |
| Different people merged together | Raise `HDBSCAN_CLUSTER_THRESHOLD` (e.g. 0.3) |
| Too many small/spurious clusters | Raise `HDBSCAN_MIN_CLUSTER_SIZE` (e.g. 10) |
| Blurry / tiny faces causing wrong matches | Raise `MIN_FACE_SIZE_PX` (e.g. 60) |
| Incremental recognition too aggressive | Lower `RECOGNITION_DISTANCE_THRESHOLD` (e.g. 0.35) |

All tuning parameters can be set via environment variables — see the top
of `scan-faces.py` for the full list.

---

## Performance notes

- **First run** downloads the `buffalo_l` model (~300MB) to `~/.insightface/`.
- Expect ~1–5 images/sec on CPU, ~20–50 images/sec on GPU.
- A library of 50,000 photos typically takes 3–8 hours on CPU.
- HDBSCAN clustering over 500K embeddings takes 2–10 minutes.
- Re-run clustering after any large batch of new photos.
- After bulk inserts, rebuild the ivfflat index for best query performance:
  ```sql
  REINDEX INDEX face_detections_embedding_idx;
  ```

---

## CUDA Setup

On my fedora system, CUDA 13 was installed, but ONNX seems to want 12.2.  This is how to get that installed:

1. download the CUDA 12.2 installer from [NVIDIA](https://developer.nvidia.com/cuda-12-2-0-download-archive)
2. run the installer and follow the prompts
```bash
sudo sh /home/mmorano/cuda/cuda_12.2.2_535.104.05_linux.run --override --toolkit --installpath=/usr/local/cuda-12.2
```
3. add the CUDA bin directory to your `PATH` and `LD_LIBRARY_PATH`:

> **Note:** `LD_LIBRARY_PATH` must be exported *before* Python starts — the
> dynamic loader reads it once at process start, so nothing the script does at
> runtime can substitute for it. `scan-faces.py` logs which provider it actually
> used; if you see "running on CPU", it prints the exact `export` line for this
> environment.

```bash
export CUDA_HOME=/usr/local/cuda-12.2
export PATH=/usr/local/cuda-12.2/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:~/git/maw-media-ai/.venv/lib/python3.14/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```
