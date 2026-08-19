# Face Scanner

Detects and recognises faces in a local media library, storing results in
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

This is required — every script refuses to run without it. There is no
in-source default, so nothing can connect to an unintended database just
because a variable was forgotten. `./setup-db.sh` prints the exact line for
your container.

---

## Usage

The scanner is a single CLI with subcommands. Run any command with `-h` for
its full options. A typical pipeline is:

```
scan → cluster → (label some clusters in the UI) → suggest → merge-clusters
```

### Scan for faces

Detects faces in the media library, stores embeddings, and caches a
display-sized crop of each face on disk for the web UI. Files already in the
database are skipped automatically, so this is safe to re-run as new media
arrives — no separate "incremental" flag is needed.

```bash
python scan-faces.py scan --media-dir /path/to/media
```

Image decoding is overlapped with inference on background threads. Tune with
`--workers N` (detection itself stays serial).

### Cluster into persons

Groups unsettled faces into person clusters using HDBSCAN. Faces belonging to a
named person, or to a cluster you have marked Unknown / Not a person, are left
completely alone — so re-running this never disturbs labelling or triage work.

```bash
python scan-faces.py cluster
```

Embeddings are reduced to 60 dimensions (PCA) before clustering. HDBSCAN falls
back to brute-force pairwise distances above 60 dimensions, which makes raw
512-dim clustering O(n²) — measured at n^2.01, roughly 35 minutes for 62k faces
versus about a minute reduced. Scored against already-labelled faces, the
reduced version is also *more* accurate (ARI 0.961 vs 0.908).

```bash
python scan-faces.py cluster --no-pca              # raw 512-dim (slow, for comparison)
python scan-faces.py cluster --pca-components 96   # different reduction
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

## Publishing to maw-media

`publish-faces.py` pushes the pipeline's conclusions — who a person is, which
faces are theirs, and where each face sits in its image — to
[maw-media](https://github.com/mmorano/maw-media), so visitors can search for
photos of people they know. See `docs/face-sync.md` for the full design.

No embedding ever crosses the wire, and nothing flows back: maw-media holds a
read-only projection, this project stays the system of record.

### What gets published

Named people and their faces. Unnamed clusters, triaged clusters and unassigned
faces stay here until the suggestion loop gives visitors a way to name them.
The policy is the two `*_IN_SCOPE` constants at the top of `publish-faces.py`;
widening it is a one-line change, and the outbox then offers up the
newly-in-scope rows on the next run.

### Setup

Apply the outbox migration once (adds revision tracking; takes ~20s against
255k faces, and `schema.sql` already contains it for fresh databases):

```bash
psql "$FACE_SCANNER_DSN" -f migrations/007-publish-outbox.sql
```

Then write one credentials file per environment, at
`~/maw-media-ai/<env>/config.json`:

```bash
mkdir -p ~/maw-media-ai/prod
cat > ~/maw-media-ai/prod/config.json <<'EOF'
{
    "client_id": "...",
    "client_secret": "..."
}
EOF
chmod 600 ~/maw-media-ai/prod/config.json
```

The credentials belong to the machine-to-machine Auth0 application holding the
`face-recognition:publish` scope, whose `media.user` row must also have the
**admin** role (the sync functions call `media.get_is_admin`).

Credentials are read from this file and never from the environment, so no
`export` can leak a production secret into a shell history, a crash dump or a
child process. The file is JSON — the standard library reads it, and it is
plain enough to edit by hand.

A directory per environment is the canonical layout, leaving room for anything
else an environment needs later. `~/maw-media-ai/<env>.json` and a plain
`~/maw-media-ai/<env>` file also work, and `--config PATH` overrides the
location entirely. If none is found, the error lists every path it tried.

Optional keys let a file point an environment elsewhere without touching the
script: `api_url`, `audience`, `auth_url`, plus the two TLS keys below. Any of
them is announced at startup as an override. An unrecognised key is an error
rather than a silent no-op, so a typo cannot leave you wondering why a setting
did nothing.

The only environment variables the publisher reads are `FACE_SCANNER_DSN` and
`ASSET_ROOT` (default `/data/maw-media-assets`) — neither is a secret.

### Choosing an environment

There is no default target. Every `publish` names its environment, so nothing
reaches the live site without the word `prod` in the command:

| Flag | Target | Audience |
|------|--------|----------|
| `--prod` | `https://media.mikeandwan.us` | `https://media.mikeandwan.us` |
| `--staging` | `https://staging-media.mikeandwan.us:8090` | `https://staging-media.mikeandwan.us` |
| `--dev` | `https://dev-media.mikeandwan.us:8091` | `https://dev-media.mikeandwan.us` |

One flag picks the url, the audience *and* the credentials file together, so
they cannot disagree — a token minted for production is rejected by dev, which is
the outcome you want. Each run prints all three before doing anything:

```
INFO: Target: PROD -> https://media.mikeandwan.us/api/v1
INFO:         audience https://media.mikeandwan.us
INFO:         config   /home/you/maw-media-ai/prod
```

**Only `--prod` records progress.** The outbox has one `published_revision` per
row and it means "what production holds"; if a dev run stamped it, the next
production run would find an empty queue and the website would silently stop
updating. So non-production runs send real data but write nothing locally, which
also makes them repeatable.

### Private CAs (the internal environments)

A CA installed with `update-ca-trust` is trusted by browsers and `openssl`, but
**not** by Python: `requests` verifies against `certifi`'s bundle of public
roots, which never contains it. The publisher therefore defaults to the OS trust
store (`/etc/pki/tls/certs/ca-bundle.crt` and friends), falling back to certifi
when no system bundle exists. `"ca_bundle": "/path/to/bundle"` overrides it.

That leaves a second, separate trap. Python 3.13 turned on `VERIFY_X509_STRICT`
by default, which enforces RFC 5280 structural rules that OpenSSL and browsers
do not. A certificate missing the **Authority Key Identifier** extension is
accepted by `openssl s_client` (`Verify return code: 0 (ok)`) and rejected by
Python:

```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: Missing Authority Key Identifier
```

The fix is to reissue the server certificate with

```
authorityKeyIdentifier = keyid,issuer
```

in its OpenSSL extension section. Until then `"tls_strict": false` relaxes only
those structural checks for that one environment — chain of trust, expiry and
hostname verification all stay on, and the run warns when it is in effect.

Both failures print the diagnosis and the exact key to add, since the confusing
part is that the browser succeeds where Python does not.

### Usage

```bash
python publish-faces.py status                    # what is pending; no network calls
python publish-faces.py publish --dev --dry-run   # build and validate every payload, send nothing
python publish-faces.py publish --dev             # real send; outbox untouched
python publish-faces.py publish --prod --limit 50 # cautious first run (faces only; persons always go in full)
python publish-faces.py publish --prod
```

`--dry-run` works without a config file, since it sends nothing — useful for
checking payloads and path transforms on a machine that has no credentials.

Re-running is always safe. Rows are stamped as published only after the API
accepts them, so an interrupted run simply resumes, and the API's revision guard
turns a duplicate into a no-op. The initial publish is ~171 requests and takes a
couple of minutes.

### How it decides what changed

Not timestamps — two databases on two machines have two clocks, and a
transaction that commits after a watermark is read becomes invisible forever.
Instead every person and face carries a `revision` from a shared sequence, and
a row needs publishing when `published_revision < revision`. Triggers bump the
revision only when a *published* column changes, so a re-scan that produces
identical assignments does not enqueue a quarter of a million no-op updates.

Three things can put a row on the queue:

| Situation | What happens |
|-----------|--------------|
| New or changed | Published, and `published_revision` stamped on acceptance |
| Deleted here | A tombstone row drives an explicit deletion — absence from a batch means "unchanged", not "gone" |
| Left publish scope (e.g. you cleared a person's name) | **Retracted**: deleted from maw-media and marked unpublished here, so it returns automatically if you name them again |

### Which face represents a person

Every named person publishes one face image — the same face the Streamlit app
shows on its cluster card: the face you starred, or the strongest detection when
you have not starred one (ties broken by id, so the choice is stable).

The star is left alone locally; writing a guess into `preferred_face_id` would be
indistinguishable from an operator decision. The fallback is resolved at publish
time and maw-media is told the result, so `media.person.preferred_face_id` always
points at a face whose image was actually uploaded.

This is why the queue shows 448 face images rather than the 64 you have starred.
Star a different face and the next publish moves the pointer and uploads that
crop.

### When a link breaks

Faces are joined to media by published file path, so renaming or recategorising
a file upstream breaks the link. That is reported rather than silent: the API
answers `unresolved_path` for that row, the face is left unstamped so it retries,
and the reason is recorded locally.

```sql
SELECT file_path, publish_error FROM media WHERE publish_error IS NOT NULL;
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

A cluster card is badged **AUTO** when nobody picked its face by hand, so the
strongest detection is standing in — which is also the face the publisher sends
to the website. The **No face chosen** filter narrows the grid to those, and
composes with the Show filter, so *Named* + *No face chosen* is the list worth
working through. Click the star on any face to make the choice explicit and the
badge goes away.

Inside a cluster, **Select faces** turns on a checkbox per face (plus *Select
all on page*) and offers two bulk actions for that subset: **Move** them to
another named person, or **Remove** them so they return to Uncategorized. Use
Move for a cluster that is mostly one person with a few intruders — merging
would move the whole cluster instead. Both actions recompute the centroids of
every cluster involved, so `suggest` and `merge-clusters` stay accurate.

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
SELECT DISTINCT m.file_path
FROM media m
JOIN face_detection fd ON fd.media_id = m.id
JOIN person per        ON per.id = fd.person_id
WHERE per.name = 'Jane Smith';
```

**Everyone in a specific photo:**
```sql
SELECT per.name, fd.bounding_box, fd.detection_score
FROM face_detection fd
JOIN person per ON per.id = fd.person_id
WHERE fd.media_id = $1
ORDER BY fd.detection_score DESC;
```

**Photos with multiple specific people (AND logic):**
```sql
SELECT m.file_path
FROM media m
WHERE (
    SELECT COUNT(DISTINCT per.name)
    FROM face_detection fd
    JOIN person per ON per.id = fd.person_id
    WHERE fd.media_id = m.id
      AND per.name IN ('Jane Smith', 'John Smith')
) = 2;
```

**Similarity search — find photos with a face similar to a given embedding:**
```sql
SELECT m.file_path, fd.detection_score,
       fd.embedding <=> $1::vector AS distance
FROM face_detection fd
JOIN media m ON m.id = fd.media_id
ORDER BY fd.embedding <=> $1::vector
LIMIT 20;
```

**Use the convenience view:**
```sql
SELECT * FROM media_person_summary
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
