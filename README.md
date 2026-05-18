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

> **GPU users:** replace `onnxruntime` with `onnxruntime-gpu` in requirements.txt
> for significantly faster scanning.

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

### Full scan (first time)

Detects faces in every photo and stores embeddings. No clustering yet.

```bash
python scan-faces.py scan --photo-dir /path/to/photos
```

### Cluster into persons

Groups all stored embeddings into person clusters using HDBSCAN.
Run this after a full scan, or after adding many new photos.

```bash
python scan-faces.py cluster
```

### Incremental scan (ongoing)

Skips photos already in the database; processes only new arrivals.

```bash
python scan-faces.py scan --photo-dir /path/to/photos --incremental
```

After adding a batch of new photos, re-run clustering so the new faces
get folded into existing persons (or new ones are created).

### Check stats

```bash
python scan-faces.py stats
```

---

## Labelling persons

After clustering, each person has a row in the `persons` table with
`name = NULL`. You need to label them once. Options:

**Option A — SQL directly:**
```sql
-- Find a cluster and see sample faces
SELECT per.id, per.face_count, per.cluster_label
FROM persons per ORDER BY face_count DESC;

-- Name a person
UPDATE persons SET name = 'Jane Smith' WHERE id = 42;
```

**Option B — Build a simple Razor admin page** that shows a grid of face
thumbnails per cluster (crop using the `bounding_box` coordinates) and
lets you type a name. This is the most efficient workflow for large libraries.

---

## Viewing Clusters and Faces (Web UI)

A simple web interface is provided to browse detected persons/clusters and their associated faces.

### 1. Install Streamlit (if not already done)

```bash
pip install streamlit
```

### 2. Set the database connection

Make sure your `FACE_SCANNER_DSN` environment variable is set, as described above.

### 3. Run the viewer

```bash
streamlit run view_clusters_app.py
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
FROM photos p
JOIN face_detections fd ON fd.photo_id = p.id
JOIN persons per        ON per.id = fd.person_id
WHERE per.name = 'Jane Smith';
```

**Everyone in a specific photo:**
```sql
SELECT per.name, fd.bounding_box, fd.detection_score
FROM face_detections fd
JOIN persons per ON per.id = fd.person_id
WHERE fd.photo_id = $1
ORDER BY fd.detection_score DESC;
```

**Photos with multiple specific people (AND logic):**
```sql
SELECT p.file_path
FROM photos p
WHERE (
    SELECT COUNT(DISTINCT per.name)
    FROM face_detections fd
    JOIN persons per ON per.id = fd.person_id
    WHERE fd.photo_id = p.id
      AND per.name IN ('Jane Smith', 'John Smith')
) = 2;
```

**Similarity search — find photos with a face similar to a given embedding:**
```sql
SELECT p.file_path, fd.detection_score,
       fd.embedding <=> $1::vector AS distance
FROM face_detections fd
JOIN photos p ON p.id = fd.photo_id
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
