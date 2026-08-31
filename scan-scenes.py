#!/usr/bin/env python3
"""
Classify the scene in every media file: how outdoor it looks, and what it is.

See docs/place-covers.md in maw-media for the design.  Two products from one
pass over the library:

  * `outdoor_score` -- probability mass over the 204 outdoor categories of
    Places365's 365.  Ranks candidates for a location's cover image.
  * the top-K scene categories with their probabilities -- folded into
    maw-media's search at the lowest tsvector weight, and used to find a
    `museum/indoor` shot for a place that has no good outdoor photo.

Rows are claimed by `scene_scored_at IS NULL`, so a run resumes where the last
one stopped and re-running after new media arrives only does the new work.

    scan-scenes.py --media-dir /data/maw-media-assets
    scan-scenes.py --limit 500      # a small first pass
    scan-scenes.py --rescan-all     # forget previous scores and redo them
    scan-scenes.py --sample 40      # score without writing, and print the result

Run export-scene-model.sh first; it converts the published PyTorch weights to
ONNX so this script needs only onnxruntime, which is already here for faces.

A separate script rather than a `scan-faces.py` subcommand: this needs no part of
insightface, and scan-faces.py imports it at module scope.

COVERAGE.  The work list comes from the media library on disk, not from what the
face scan happened to record, so "every photo" means every photo.  In practice
the two agree -- media already holds a row per scanned file whether or not it
contained a face -- but they drift the moment new media lands, and a scene scan
should not silently inherit the face scan's backlog.

A photo it finds with no media row is registered so it can be scored.  That is
safe as of migration 010: the face scan claims work by `faces_scanned_at` rather
than by the existence of a row, so a row created here is still face-scanned
later.  Either scanner can run first, and neither hides work from the other.
"""

import argparse
import glob
import itertools
import logging
import os
import sysconfig
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid7

import cv2
import numpy as np
import onnxruntime as ort
import psycopg
from psycopg.rows import dict_row
from tqdm import tqdm

# --- Configuration -----------------------------------------------------------
# No default; see the note in scan-faces.py.  connect() enforces it.
DB_DSN = os.getenv("FACE_SCANNER_DSN")

MODEL_DIR = Path(
    os.getenv("SCENE_MODEL_DIR", "~/.cache/maw-media-ai/models")
).expanduser()
MODEL_PATH = MODEL_DIR / "resnet18_places365.onnx"
CATEGORIES_PATH = MODEL_DIR / "categories_places365.txt"
IO_PATH = MODEL_DIR / "IO_places365.txt"

# How many scene labels to keep per media.  5 is far more than the UI shows; the
# extra rows cost ~835k for the whole library and mean the probability floor for
# search can be retuned without re-scoring anything.
TOP_K = int(os.getenv("SCENE_TOP_K", "5"))

# Inference batch.  The model is tiny (44MB, resnet18) so this is bounded by
# decode throughput rather than GPU memory.
BATCH_SIZE = int(os.getenv("SCENE_BATCH_SIZE", "64"))
LOADER_THREADS = int(os.getenv("SCENE_LOADER_THREADS", "8"))

# Where the library lives, and which files count as photos.  Both mirror
# scan-faces.py: only images inside a directory named `full`, which is the scale
# the publisher writes originals to.  A video contributes its poster frame, which
# is an .avif in the same place, so "photos only" needs no extra rule.
MEDIA_DIR = os.getenv("MEDIA_DIR", "/data/maw-media-assets")  # default for --media-dir
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".tif", ".bmp", ".avif"}

# Read order.  A 224x224 classifier has no use for the 5400x3600 `full` file the
# face scanner needs: decoding it measured 3 img/s against 147 for `nhd`, which
# is the difference between minutes and most of a day.  Fall through rather than
# skip, since video posters do not always have every variant.
SCALE_PREFERENCE = ("nhd", "qvg", "full-hd", "full")

# ImageNet preprocessing, which is what Places365 was trained with.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_RESIZE_SHORT = 256
_CROP = 224

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
log = logging.getLogger("scan-scenes")


def connect(**kwargs):
    """Open a connection, refusing to fall back to libpq's defaults."""
    if not DB_DSN:
        raise SystemExit(
            "FACE_SCANNER_DSN is not set.  Export the connection string first:\n"
            '    export FACE_SCANNER_DSN="postgresql://user:pass@host:5433/face_scanner"'
        )

    return psycopg.connect(DB_DSN, **kwargs)


# --- Model -------------------------------------------------------------------
def _bundled_nvidia_lib_dirs() -> list[str]:
    """Directories of the NVIDIA libraries shipped inside this environment."""
    dirs = set()
    for site in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
        for so in glob.glob(os.path.join(site, "nvidia", "*", "lib", "lib*.so*")):
            dirs.add(os.path.realpath(os.path.dirname(so)))
    return sorted(dirs)


def _warn_cpu_fallback(active: list[str]) -> None:
    """Explain a CPU fallback and, when we can, print the fix.

    Same cause as in scan-faces.py: the CUDA provider links against cuDNN, which
    ships in the venv but is only found via LD_LIBRARY_PATH -- read once at
    process start, so it must be exported before python runs.
    """
    log.warning(
        f"Scene model is running on CPU ({', '.join(active) or 'unknown'}). "
        "This still works, just slower."
    )
    lib_dirs = _bundled_nvidia_lib_dirs()
    if lib_dirs:
        log.warning(
            "NVIDIA libraries are present in this environment but not on the "
            "loader path. Export this, then re-run:"
        )
        log.warning(
            '    export LD_LIBRARY_PATH="%s:$LD_LIBRARY_PATH"',
            ":".join(lib_dirs),
        )


class SceneModel:
    """Places365 ResNet18, plus the label tables that give its output meaning."""

    def __init__(self):
        missing = [p for p in (MODEL_PATH, CATEGORIES_PATH, IO_PATH) if not p.is_file()]
        if missing:
            raise SystemExit(
                "Scene model not found:\n"
                + "".join(f"  {p}\n" for p in missing)
                + "\nRun ./export-scene-model.sh to build it."
            )

        # "/a/airfield 0" -> "airfield"; the leading "/x/" is an alphabetical
        # bucket, not part of the name.
        self.categories = [
            line.strip().split(" ")[0][3:] for line in CATEGORIES_PATH.open()
        ]
        io_flags = np.array(
            [int(line.strip().split(" ")[1]) for line in IO_PATH.open()]
        )
        self.outdoor_mask = io_flags == 2  # 1 = indoor, 2 = outdoor

        if len(self.categories) != len(self.outdoor_mask):
            raise SystemExit(
                f"Model label files disagree: {len(self.categories)} categories "
                f"vs {len(self.outdoor_mask)} indoor/outdoor flags."
            )

        self.session = ort.InferenceSession(
            str(MODEL_PATH),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        active = self.session.get_providers()
        if "CUDAExecutionProvider" not in active:
            _warn_cpu_fallback(active)

        log.info(
            "Scene model on %s — %d categories (%d outdoor, %d indoor)",
            active[0],
            len(self.categories),
            int(self.outdoor_mask.sum()),
            int((~self.outdoor_mask).sum()),
        )

    def predict(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (outdoor_score per image, probabilities per image)."""
        logits = self.session.run(["logits"], {"input": batch})[0]
        # Subtract the row max before exponentiating; the raw logits are large
        # enough that exp() overflows to inf and every probability becomes nan.
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=1, keepdims=True)

        return probs[:, self.outdoor_mask].sum(axis=1), probs


# --- Image loading -----------------------------------------------------------
def scaled_variant(file_path: str) -> str | None:
    """The smallest acceptable rendition of a media file, or None if absent.

    Paths look like /.../{category}/{scale}/{name}, so the scale is swapped by
    replacing one path component.
    """
    p = Path(file_path)
    media_dir = p.parent.parent
    for scale in SCALE_PREFERENCE:
        candidate = media_dir / scale / p.name
        if candidate.is_file():
            return str(candidate)

    return None


def preprocess(path: str) -> np.ndarray | None:
    """Decode and normalise one image to the model's input, or None if unreadable."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None

    h, w = img.shape[:2]
    if min(h, w) == 0:
        return None

    # Resize the short edge to 256 then centre crop 224, the standard evaluation
    # transform these weights were trained under.  Anything else shifts the
    # scores in ways the published accuracy numbers no longer describe.
    scale = _RESIZE_SHORT / min(h, w)
    img = cv2.resize(
        img,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )

    h, w = img.shape[:2]
    top, left = (h - _CROP) // 2, (w - _CROP) // 2
    img = img[top : top + _CROP, left : left + _CROP]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return ((img - _MEAN) / _STD).transpose(2, 0, 1)


def load_one(row: dict) -> np.ndarray | None:
    """Thread worker: resolve a rendition and decode it."""
    variant = scaled_variant(row["file_path"])
    if variant is None:
        return None

    return preprocess(variant)


def iter_prefetched(rows: list[dict], n_workers: int):
    """Yield ``(row, array)`` in order, decoding in worker threads.

    A bounded look-ahead keeps ~2x workers images resident, so decode overlaps
    with inference without holding the whole library in memory -- each
    preprocessed array is 3x224x224 float32, so an unbounded map over 167k media
    would ask for ~100GB.  Mirrors _iter_prefetched_images in scan-faces.py.
    """
    n_workers = max(1, n_workers)
    window = n_workers * 2

    with ThreadPoolExecutor(
        max_workers=n_workers, thread_name_prefix="scene-load"
    ) as pool:
        it = iter(rows)
        pending: deque = deque()
        for row in itertools.islice(it, window):
            pending.append((row, pool.submit(load_one, row)))

        while pending:
            row, fut = pending.popleft()
            array = fut.result()
            nxt = next(it, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(load_one, nxt)))
            yield row, array


# --- Database ----------------------------------------------------------------
def iter_library(media_dir: str) -> list[Path]:
    """Every photo in the library, by the same rule scan-faces.py uses."""
    root = Path(media_dir)
    if not root.is_dir():
        raise SystemExit(
            f"Media directory not found: {media_dir}\n"
            "Set MEDIA_DIR to the root of the library."
        )

    paths = [
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.parent.name == "full"
    ]
    paths.sort()

    return paths


def register_media(conn, paths: list[str]) -> list[dict]:
    """Create media rows for photos nothing has recorded yet.

    faces_scanned_at is left NULL, which is what tells scan-faces.py these still
    need detecting.  ON CONFLICT rather than a pre-check because the face scan
    may be running at the same time, and file_path is unique.
    """
    created = []

    with conn.cursor() as cur:
        for path in paths:
            cur.execute(
                """
                INSERT INTO media (id, file_path, file_name)
                VALUES (%s, %s, %s)
                ON CONFLICT (file_path) DO UPDATE SET file_path = EXCLUDED.file_path
                RETURNING id, file_path, scene_scored_at
                """,
                (str(uuid7()), path, os.path.basename(path)),
            )
            created.append(cur.fetchone())
    conn.commit()

    return created


def build_worklist(
    conn, media_dir: str, limit: int | None, rescan: bool, randomize: bool
) -> tuple[list[dict], int, int]:
    """Reconcile the library on disk against the media table.

    Returns (rows to score, files with no media row, media rows whose file is
    gone).  The scoring set is the intersection: a file needs a media row to
    hang a score on, and a row needs a file to score.
    """
    log.info("Walking %s", media_dir)
    on_disk = {str(p) for p in iter_library(media_dir)}

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT m.id, m.file_path, m.scene_scored_at
            FROM media m
            ORDER BY m.file_path
            """
        )
        media_rows = cur.fetchall()

    # Only rows under the directory being walked are in scope.  Comparing against
    # the whole table would report every photo outside a subtree scan as missing.
    root_prefix = str(Path(media_dir)) + os.sep
    media_rows = [r for r in media_rows if r["file_path"].startswith(root_prefix)]
    known = {r["file_path"] for r in media_rows}
    unregistered = sorted(on_disk - known)
    missing_file = len(known - on_disk)

    if unregistered:
        log.info("Registering %d photo(s) not yet in the media table", len(unregistered))
        media_rows.extend(register_media(conn, unregistered))

    candidates = [
        r
        for r in media_rows
        if r["file_path"] in on_disk and (rescan or r["scene_scored_at"] is None)
    ]

    if randomize:
        # Sampling orders randomly: path order means one directory, and 25
        # consecutive photos from a single afternoon say nothing about the model.
        import random

        random.shuffle(candidates)

    if limit:
        candidates = candidates[:limit]

    return candidates, len(unregistered), missing_file


def write_scores(conn, scored: list[tuple]) -> None:
    """Persist a batch: the score on media, the labels in their own table.

    One transaction per batch, so an interrupted run leaves whole media scored or
    untouched -- never a score without its labels.
    """
    if not scored:
        return

    ids = [s[0] for s in scored]
    outdoor = [float(s[1]) for s in scored]

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE media m
            SET outdoor_score = v.score, scene_scored_at = now()
            FROM (SELECT unnest(%s::uuid[]) AS id, unnest(%s::real[]) AS score) v
            WHERE m.id = v.id
            """,
            (ids, outdoor),
        )

        # Replace rather than accumulate, so a rescan with a different TOP_K
        # cannot leave stale ranks behind.
        cur.execute(
            "DELETE FROM media_scene_label WHERE media_id = ANY(%s::uuid[])", (ids,)
        )

        rows = [
            (mid, rank, code, float(p))
            for mid, _, labels in scored
            for rank, (code, p) in enumerate(labels, start=1)
        ]
        if rows:
            cur.executemany(
                "INSERT INTO media_scene_label (media_id, rank, code, probability) "
                "VALUES (%s, %s, %s, %s)",
                rows,
            )

    conn.commit()


def mark_unreadable(conn, media_ids: list) -> None:
    """Stamp media whose rendition is missing or undecodable.

    Stamped, not skipped: without this they are re-attempted on every run
    forever.  outdoor_score stays NULL, which is what "no opinion" looks like to
    every consumer.
    """
    if not media_ids:
        return

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE media SET scene_scored_at = now() WHERE id = ANY(%s::uuid[])",
            (media_ids,),
        )
    conn.commit()


# --- Commands ----------------------------------------------------------------
def run(
    media_dir: str, limit: int | None, rescan: bool, sample: int | None
) -> None:
    model = SceneModel()

    with connect(row_factory=dict_row) as conn:
        rows, unregistered, missing_file = build_worklist(
            conn, media_dir, sample or limit, rescan, randomize=bool(sample)
        )

        if unregistered:
            log.info(
                "%d photo(s) were new to the media table; they are registered "
                "and scored here, and still await a face scan.",
                unregistered,
            )
        if missing_file:
            log.info(
                "%d media row(s) point at a file that no longer exists; skipped.",
                missing_file,
            )

        if not rows:
            log.info("Nothing to score.")
            return

        log.info("%d media to score", len(rows))
        started = time.time()
        scored_count = unreadable_count = 0
        pending: list[tuple] = []
        unreadable: list = []
        preview: list[tuple] = []
        batch_rows: list[dict] = []
        batch_arrays: list[np.ndarray] = []

        def infer():
            """Run the batch that has accumulated, and move it to `pending`."""
            nonlocal scored_count, batch_rows, batch_arrays

            if not batch_arrays:
                return

            outdoor, probs = model.predict(np.stack(batch_arrays))
            for row, score, prob in zip(batch_rows, outdoor, probs):
                top = np.argsort(-prob)[:TOP_K]
                labels = [(model.categories[i], float(prob[i])) for i in top]
                pending.append((row["id"], float(score), labels))
                if sample:
                    preview.append((float(score), labels[0][0], row["file_path"]))

            scored_count += len(batch_rows)
            batch_rows, batch_arrays = [], []

        for row, array in tqdm(
            iter_prefetched(rows, LOADER_THREADS), total=len(rows), desc="Scoring"
        ):
            if array is None:
                unreadable.append(row["id"])
                unreadable_count += 1
                continue

            batch_rows.append(row)
            batch_arrays.append(array)

            if len(batch_arrays) >= BATCH_SIZE:
                infer()
                # Commit as we go: a long run that dies partway should keep what
                # it scored, and `scene_scored_at` is what makes the next run
                # resume rather than start over.
                if not sample:
                    write_scores(conn, pending)
                    pending = []

        infer()

        if sample:
            preview.sort(reverse=True)
            print(f"\n{'outdoor':>7}  {'top scene':22} file")
            print("-" * 78)
            for score, scene, path in preview:
                print(f"{score:7.3f}  {scene:22} {path[-44:]}")
            print(f"\n{len(preview)} scored, nothing written (--sample)")
            return

        write_scores(conn, pending)
        mark_unreadable(conn, unreadable)

        elapsed = time.time() - started
        log.info(
            "Scored %d media in %.1fs (%.0f/s); %d unreadable",
            scored_count,
            elapsed,
            scored_count / elapsed if elapsed else 0,
            unreadable_count,
        )


def cmd_stats() -> None:
    with connect(row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE scene_scored_at IS NOT NULL) AS scored,
                   count(*) FILTER (WHERE outdoor_score IS NOT NULL) AS with_score,
                   count(*) FILTER (WHERE outdoor_score >= 0.9) AS strongly_outdoor,
                   count(*) FILTER (WHERE outdoor_score < 0.1) AS strongly_indoor
            FROM media
            """
        )
        m = cur.fetchone()
        cur.execute("SELECT count(*) AS n FROM media_scene_label")
        labels = cur.fetchone()["n"]
        cur.execute(
            """
            SELECT code, count(*) AS n FROM media_scene_label
            WHERE rank = 1 GROUP BY code ORDER BY n DESC LIMIT 10
            """
        )
        top = cur.fetchall()

    print("=== Scene scan ===")
    print(f"  Media total:        {m['total']:>9,}")
    print(f"  Scored:             {m['scored']:>9,}")
    print(f"  With a score:       {m['with_score']:>9,}")
    print(f"  Outdoor >= 0.9:     {m['strongly_outdoor']:>9,}")
    print(f"  Outdoor <  0.1:     {m['strongly_indoor']:>9,}")
    print(f"  Labels stored:      {labels:>9,}")
    if top:
        print("\n  Most common top scene:")
        for r in top:
            print(f"    {r['code']:24} {r['n']:>7,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--media-dir",
        default=MEDIA_DIR,
        help=f"Root of the media library to walk (default {MEDIA_DIR})",
    )
    parser.add_argument(
        "--limit", type=int, help="Score at most this many media, then stop"
    )
    parser.add_argument(
        "--rescan-all",
        action="store_true",
        help="Re-score media that already have a score",
    )
    parser.add_argument(
        "--sample",
        type=int,
        metavar="N",
        help="Score N media and print the ranking without writing anything",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show what has been scored so far"
    )
    args = parser.parse_args()

    if args.stats:
        cmd_stats()
        return

    run(args.media_dir, args.limit, args.rescan_all, args.sample)


if __name__ == "__main__":
    main()
