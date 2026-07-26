#!/usr/bin/env python3
"""
Evaluate a trained classifier against the current nearest-centroid `suggest`
logic, using existing manual labels as ground truth.

Both methods are fit on the SAME training split and scored on the SAME held-out
faces, so the comparison is apples-to-apples.

Three questions are answered:

  1. CLOSED SET  — given a face that does belong to a known person, how often
     does each method pick the right one?
  2. PRECISION / COVERAGE — if we only auto-suggest above a confidence bar, how
     much work gets done and how much of it is wrong?  This is what actually
     matters: a confident wrong suggestion costs more than no suggestion.
  3. OPEN SET — some classes are withheld from training entirely, so their test
     faces represent strangers.  How often does each method confidently assign
     a stranger to a known person?

Nothing is written to the database.

    python evaluate-classifier.py [--holdout-classes 30] [--cap 1500]
"""

import argparse
import os
import sys
import time

import numpy as np
import psycopg
from psycopg.rows import dict_row
from sklearn.linear_model import LogisticRegression

DB_DSN = os.getenv(
    "FACE_SCANNER_DSN",
    "postgresql://face_scanner:face_scanner_secret@localhost:5433/face_scanner",
)

TEST_FRACTION = 0.20
RANDOM_SEED = 42


def log(msg: str) -> None:
    print(msg, flush=True)


def load_labeled(dsn: str) -> tuple[np.ndarray, np.ndarray, dict[str, str]]:
    """Return (X, y, id->name) for every face belonging to a NAMED person."""
    log("Loading labelled embeddings…")
    t0 = time.time()
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT fd.person_id::text AS pid, p.name, fd.embedding::text AS emb
                FROM face_detection fd
                JOIN person p ON p.id = fd.person_id
                WHERE p.name IS NOT NULL AND fd.embedding IS NOT NULL
                """
            )
            rows = cur.fetchall()

    X = np.empty((len(rows), 512), dtype=np.float32)
    y = np.empty(len(rows), dtype=object)
    names: dict[str, str] = {}
    for i, r in enumerate(rows):
        X[i] = np.fromstring(r["emb"].strip("[]"), sep=",", dtype=np.float32)
        y[i] = r["pid"]
        names[r["pid"]] = r["name"]
    log(f"  {len(rows):,} faces, {len(names)} people  ({time.time()-t0:.1f}s)")
    return X, y, names


def stratified_split(y: np.ndarray, rng: np.random.Generator):
    """Per-class split so every class appears in train (and test when possible)."""
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        idx = np.flatnonzero(y == cls)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * TEST_FRACTION))
        # keep at least 1 training sample; only spare a test sample if >= 2 exist
        n_test = min(n_test, len(idx) - 1)
        if len(idx) >= 2:
            n_test = max(n_test, 1)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(train_idx), np.array(test_idx)


def cap_per_class(X, y, idx, cap, rng):
    """Downsample over-represented classes -- balances AND speeds up training."""
    keep = []
    for cls in np.unique(y[idx]):
        c = idx[y[idx] == cls]
        if len(c) > cap:
            c = rng.choice(c, cap, replace=False)
        keep.extend(c)
    return np.array(keep)


def centroid_model(X, y, idx):
    """Current `suggest` logic: L2-normalised mean embedding per class."""
    classes = np.unique(y[idx])
    C = np.empty((len(classes), X.shape[1]), dtype=np.float32)
    for i, cls in enumerate(classes):
        v = X[idx[y[idx] == cls]].mean(axis=0)
        n = np.linalg.norm(v)
        C[i] = v / n if n > 0 else v
    return classes, C


def centroid_predict(C, classes, Xt):
    """Return (pred, confidence) where confidence = 1 - cosine_distance."""
    Xn = Xt / np.clip(np.linalg.norm(Xt, axis=1, keepdims=True), 1e-9, None)
    sims = Xn @ C.T                       # cosine similarity
    best = sims.argmax(axis=1)
    return classes[best], sims[np.arange(len(Xt)), best]


def report_threshold_table(name, conf, correct, is_known):
    """Precision / coverage as the confidence bar rises."""
    log(f"\n  {name} — precision vs coverage")
    log("    thresh | coverage | precision | stranger false-accepts")
    log("    -------+----------+-----------+-----------------------")
    for t in (0.0, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        sel = conf >= t
        n_sel = int(sel.sum())
        if n_sel == 0:
            log(f"    {t:>6.2f} |    0.0%  |     --    |          --")
            continue
        known_sel = sel & is_known
        prec = correct[known_sel].mean() if known_sel.any() else float("nan")
        strangers = int((sel & ~is_known).sum())
        n_strangers = int((~is_known).sum())
        far = strangers / n_strangers if n_strangers else 0.0
        log(
            f"    {t:>6.2f} |  {100*n_sel/len(conf):>5.1f}%  "
            f"|   {100*prec:>5.1f}%  |  {strangers:>6,} of {n_strangers:,} ({100*far:.1f}%)"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout-classes", type=int, default=30,
                    help="People withheld from training to act as strangers (default 30).")
    ap.add_argument("--cap", type=int, default=1500,
                    help="Max training faces per person (default 1500).")
    args = ap.parse_args()

    rng = np.random.default_rng(RANDOM_SEED)
    X, y, names = load_labeled(DB_DSN)

    # --- withhold whole classes to simulate strangers -----------------------
    classes_all = np.unique(y)
    eligible = [c for c in classes_all if (y == c).sum() >= 10]
    rng.shuffle(eligible)
    stranger_classes = set(eligible[: args.holdout_classes])
    log(f"\nWithholding {len(stranger_classes)} people from training to act as strangers.")

    known_mask = np.array([c not in stranger_classes for c in y])
    Xk, yk = X[known_mask], y[known_mask]

    tr_uncapped, te = stratified_split(yk, rng)
    tr = cap_per_class(Xk, yk, tr_uncapped, args.cap, rng)
    log(f"Train: {len(tr):,} capped (centroids use all {len(tr_uncapped):,}) / {len(np.unique(yk[tr]))} people"
        f"   Test: {len(te):,} faces")

    # test set = held-out known faces + ALL faces of the withheld people
    X_test = np.vstack([Xk[te], X[~known_mask]])
    y_test = np.concatenate([yk[te], y[~known_mask]])
    is_known = np.concatenate([np.ones(len(te), bool), np.zeros((~known_mask).sum(), bool)])
    log(f"Evaluation set: {len(X_test):,} faces "
        f"({is_known.sum():,} known + {(~is_known).sum():,} stranger)")

    # --- A. centroid (current behaviour) ------------------------------------
    log("\n[A] nearest-centroid (current `suggest`)")
    t0 = time.time()
    # NB: fit on the UNCAPPED training split -- production computes
    # representative_embedding as avg() over every face of the person.
    classes, C = centroid_model(Xk, yk, tr_uncapped)
    pred_c, conf_c = centroid_predict(C, classes, X_test)
    log(f"    fit+predict in {time.time()-t0:.1f}s")

    # --- B. logistic regression ---------------------------------------------
    log("\n[B] logistic regression")
    t0 = time.time()
    clf = LogisticRegression(
        max_iter=1000, C=10.0, class_weight="balanced",
    )
    clf.fit(Xk[tr], yk[tr])
    log(f"    trained in {time.time()-t0:.1f}s")
    proba = clf.predict_proba(X_test)
    best = proba.argmax(axis=1)
    pred_l, conf_l = clf.classes_[best], proba[np.arange(len(X_test)), best]

    # --- results -------------------------------------------------------------
    corr_c = (pred_c == y_test)
    corr_l = (pred_l == y_test)
    log("\n" + "=" * 68)
    log("CLOSED-SET TOP-1 ACCURACY (known faces only, no threshold)")
    log(f"  nearest-centroid : {100*corr_c[is_known].mean():.2f}%")
    log(f"  logistic regr.   : {100*corr_l[is_known].mean():.2f}%")

    report_threshold_table("nearest-centroid", conf_c, corr_c, is_known)
    report_threshold_table("logistic regression", conf_l, corr_l, is_known)

    log("\n" + "=" * 68)
    log("Notes:")
    log("  * 'coverage'  = share of ALL evaluated faces that clear the bar.")
    log("  * 'precision' = of the KNOWN faces that clear the bar, share assigned")
    log("                  to the correct person.")
    log("  * 'stranger false-accepts' = withheld people wrongly given a suggestion;")
    log("                  every one of these is manual rejection work for you.")
    log("  * centroid confidence is 1 - cosine distance, so the current default")
    log("    SUGGEST_DISTANCE_THRESHOLD=0.35 corresponds to a 0.65 bar here.")


if __name__ == "__main__":
    sys.exit(main())
