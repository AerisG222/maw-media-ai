#!/usr/bin/env python3
"""
Publish people and faces from this project's database to maw-media.

See docs/face-sync.md for the full design.  The short version:

  * maw-media-ai owns the data; maw-media holds a read-only projection.  Nothing
    flows back through this tool.
  * No embedding ever crosses the wire -- only the conclusions: who a person is,
    which faces are theirs, and where each face sits in its image.
  * Media are joined by published file path.  maw-media resolves the path to its
    own media id inside the publish transaction, so this project never learns
    (or caches) a maw-media id.

Delivery is at-least-once.  `published_revision` is stamped only after the API
accepts a row, so a crash anywhere simply republishes, and the API's revision
guard makes the retry a no-op.

The target environment is always named explicitly -- there is no default, so
production is only ever reached by typing --prod.  The flag picks the api url,
the Auth0 audience and the credentials file together, so they cannot disagree.

Credentials come from ~/maw-media-ai/<environment>/config.json, a JSON file
holding at least client_id and client_secret:

    {
        "client_id": "...",
        "client_secret": "..."
    }

It may also carry api_url, audience or auth_url to point an environment
somewhere else (a local API, a mock) without editing this script.

    publish-faces.py status                 # what is pending, no network calls
    publish-faces.py publish --dev --dry-run
    publish-faces.py publish --dev          # real send, outbox untouched
    publish-faces.py publish --prod
    publish-faces.py reset --faces          # force a full republish

Deliberately a separate script rather than a `scan-faces.py` subcommand:
scan-faces.py imports insightface, cv2 and hdbscan at module scope, and a
publish run has no business loading a GPU inference stack.
"""

import argparse
import json
import logging
import os
import re
import ssl
import sys
import time
from datetime import timezone
from pathlib import Path

import psycopg
import requests
from psycopg.rows import dict_row

# --- Configuration -----------------------------------------------------------
DB_DSN = os.getenv("FACE_SCANNER_DSN")

# --- Environments ------------------------------------------------------------
# There is no default target: production is reached only by passing --prod, and
# every other environment names itself too.  Publishing is one-directional and
# immediately visible on a public website, so "which system am I about to write
# to" should never be answered by whatever happens to be in the shell.
#
# The audience is per-environment and NOT interchangeable: a token minted for
# https://media.mikeandwan.us is simply rejected by dev, which is the good
# outcome -- deriving it from the same flag as the url means the two cannot
# disagree.  Values come from maw-media's appsettings.*.json and the matching
# maw-photos-solid environments/.env.* files.
ENVIRONMENTS = {
    "prod": {
        "api_url": "https://media.mikeandwan.us/api/v1",
        "audience": "https://media.mikeandwan.us",
    },
    "staging": {
        "api_url": "https://staging-media.mikeandwan.us:8090/api/v1",
        "audience": "https://staging-media.mikeandwan.us",
    },
    "dev": {
        "api_url": "https://dev-media.mikeandwan.us:8091/api/v1",
        "audience": "https://dev-media.mikeandwan.us",
    },
}

# One Auth0 tenant serves every environment -- only the audience changes -- so
# this is a default rather than something each config file has to repeat.
DEFAULT_AUTH_URL = "https://login.mikeandwan.us"

# Credentials live in a per-environment file, NOT the environment, so that a
# stray export cannot send a production secret somewhere else, and so `env` in a
# shell history or a crash dump never contains one.
CONFIG_DIR = Path("~/maw-media-ai").expanduser()

# --- TLS ---------------------------------------------------------------------
# requests verifies against certifi's bundle, which contains public roots only.
# A CA added to the OS trust store -- the thing that makes a browser accept the
# internal environments -- is therefore invisible to Python unless we point at
# the system bundle explicitly.  Falls back to certifi where none of these exist.
SYSTEM_CA_BUNDLES = (
    "/etc/pki/tls/certs/ca-bundle.crt",        # fedora / rhel
    "/etc/ssl/certs/ca-bundle.crt",            # fedora, via symlink
    "/etc/ssl/certs/ca-certificates.crt",      # debian / ubuntu
)


def system_ca_bundle() -> str | None:
    return next((p for p in SYSTEM_CA_BUNDLES if os.path.exists(p)), None)

# The path transform is a root swap: this project reads a local copy of the tree
# maw-media serves.  See docs/face-sync.md "The path contract".
ASSET_ROOT = os.getenv("ASSET_ROOT", "/data/maw-media-assets")
WEB_PREFIX = os.getenv("WEB_ASSET_PREFIX", "/assets")

REQUEST_TIMEOUT = int(os.getenv("PUBLISH_TIMEOUT_SECONDS", "120"))

# Must not exceed the server-side caps (PersonRoutes.MAX_PERSONS,
# FaceRoutes.MAX_FACES, ConfigRoutes.MAX_PERSON_STATUSES); a larger batch is
# rejected with a 400 naming the limit.
STATUS_BATCH = 100
PERSON_BATCH = 500
FACE_BATCH = 1000
DELETION_BATCH = 1000

# --- Publish scope -----------------------------------------------------------
# WHAT GETS PUBLISHED, in one place.  This predicate is the whole policy, and it
# is read three ways: rows in scope and stale get published, rows out of scope
# that were published get RETRACTED, and `status` counts both.  Widening the
# policy is a one-line change here -- the outbox then offers up the newly
# in-scope rows on the next run with no backfill needed.
#
# Currently: named people only.  Unnamed clusters are withheld until the
# suggestion loop (face-sync.md §8 phase 4) gives visitors a way to name them.
PERSON_IN_SCOPE = "p.name IS NOT NULL"
FACE_IN_SCOPE = "per.name IS NOT NULL"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
log = logging.getLogger("publish-faces")


class PublishError(Exception):
    pass


CONFIG_TEMPLATE = """{
    "client_id": "...",
    "client_secret": "..."
}"""

# api_url / audience / auth_url may also be set, to point an environment
# somewhere else (a local API, a mock) without touching this file.
# ca_bundle / tls_strict cover internal environments served by a private CA.
CONFIG_KEYS = {
    "client_id",
    "client_secret",
    "api_url",
    "audience",
    "auth_url",
    "ca_bundle",
    "tls_strict",
}
REQUIRED_KEYS = ("client_id", "client_secret")


def config_candidates(environment: str) -> list[Path]:
    """Where an environment's config may live, in order of preference.

    A directory per environment is canonical -- it leaves room for anything else
    an environment might need later without moving this file.  The two flat
    forms are accepted because they are the obvious things to try, and a missing
    config reports every path it looked at rather than just the first.
    """
    return [
        CONFIG_DIR / environment / "config.json",
        CONFIG_DIR / f"{environment}.json",
        CONFIG_DIR / environment,  # only used when it is a file, not a directory
    ]


def load_config(environment: str, explicit: str | None = None) -> dict:
    paths = [Path(explicit).expanduser()] if explicit else config_candidates(environment)
    path = next((p for p in paths if p.is_file()), None)

    if path is None:
        raise PublishError(
            f"No config for the {environment} environment. Looked in:\n"
            + "".join(f"  {p}\n" for p in paths)
            + "\nCreate it with mode 600 and the shape:\n"
            + CONFIG_TEMPLATE
        )

    try:
        config = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise PublishError(f"{path} is not valid JSON: {e}") from e

    if not isinstance(config, dict):
        raise PublishError(f"{path} should hold a JSON object, not {type(config).__name__}.")

    missing = [k for k in REQUIRED_KEYS if not config.get(k)]
    if missing:
        raise PublishError(f"{path} is missing: {', '.join(missing)}")

    unknown = set(config) - CONFIG_KEYS
    if unknown:
        # Loud, because a typo'd key means a setting silently did nothing.
        raise PublishError(
            f"{path} has unrecognised key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(CONFIG_KEYS))}."
        )

    # The file holds a secret; anything readable beyond its owner is worth
    # saying out loud, but not worth refusing to run over.
    mode = path.stat().st_mode & 0o077
    if mode:
        log.warning(
            "%s is readable by group/other (mode %s) -- chmod 600 it",
            path,
            oct(path.stat().st_mode & 0o777),
        )

    config["config_path"] = path

    return config


def resolve_target(
    environment: str, config_path: str | None = None, require_config: bool = True
) -> dict:
    """Everything that differs by environment, decided in one place."""
    try:
        config = load_config(environment, config_path)
    except PublishError:
        if require_config:
            raise
        # A dry run builds payloads and sends nothing, so demanding credentials
        # would be friction for the one command that cannot do any harm.
        log.warning("No config file found -- continuing, since a dry run sends nothing.")
        config = {"config_path": "(none)", "client_id": "", "client_secret": ""}

    target = dict(ENVIRONMENTS[environment])
    target["environment"] = environment
    target["auth_url"] = DEFAULT_AUTH_URL
    target["config_path"] = config["config_path"]
    target["client_id"] = config["client_id"]
    target["client_secret"] = config["client_secret"]

    # The OS trust store by default, so a CA installed with update-ca-trust just
    # works -- the same expectation a browser sets.
    target["ca_bundle"] = system_ca_bundle()
    target["tls_strict"] = True

    # A url in the file wins over the built-in default, and is announced at
    # startup rather than applied quietly, since it defeats the point of the flag.
    for key in ("api_url", "audience", "auth_url", "ca_bundle"):
        if config.get(key):
            target[key] = config[key]
            target.setdefault("overrides", []).append(key)

    if config.get("tls_strict") is False:
        target["tls_strict"] = False

    return target


# --- Path + slug helpers -----------------------------------------------------
def to_web_path(file_path: str) -> str:
    """Local scanner path -> the path maw-media stores in media.file.path.

    relative_to rather than a string replace, so a trailing slash on the
    configured root cannot produce a doubled separator.
    """
    rel = Path(file_path).relative_to(Path(ASSET_ROOT))

    return f"{WEB_PREFIX}/{rel.as_posix()}"


_SLUG_COLLAPSE = re.compile(r"--+")
_SLUG_STRIP = re.compile(r"[^a-zA-Z0-9_\-]")


def make_slug(name: str) -> str:
    """Port of MawMediaPublisher SlugHelper.MakeSafeSlug, so person urls read
    the same as every other slug on the site.

    Slugs are generated here rather than in maw-media because this project owns
    the name.  media.person.slug is UNIQUE and person names are not guaranteed
    unique -- they happen to be today (448 named, no duplicates), and a
    collision surfaces loudly as a failed batch rather than silently, so no
    disambiguation suffix is invented until one is actually needed.
    """
    slug = name.replace(" ", "-").replace("_", "-").lower()

    return _SLUG_STRIP.sub("", _SLUG_COLLAPSE.sub("-", slug))


# --- API client --------------------------------------------------------------
class _RelaxedTlsAdapter(requests.adapters.HTTPAdapter):
    """Everything ssl.create_default_context() gives, minus VERIFY_X509_STRICT.

    Hostname checking, expiry, chain of trust and CERT_REQUIRED all stay on --
    only the RFC 5280 structural checks Python 3.13 turned on by default are
    relaxed, and only for an environment whose config asks for it.
    """

    def __init__(self, ca_bundle: str | None, **kwargs):
        self._ca_bundle = ca_bundle
        super().__init__(**kwargs)

    def _context(self) -> ssl.SSLContext:
        ctx = ssl.create_default_context(cafile=self._ca_bundle)
        ctx.verify_flags &= ~ssl.VERIFY_X509_STRICT

        return ctx

    def init_poolmanager(self, *args, **kwargs):
        kwargs["ssl_context"] = self._context()

        return super().init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs["ssl_context"] = self._context()

        return super().proxy_manager_for(*args, **kwargs)


def tls_hint(error: Exception, target: dict) -> str:
    """Turn the two TLS failures this tool actually hits into instructions.

    Both are confusing precisely because a browser succeeds where Python fails,
    so the message names the difference rather than just the symptom.
    """
    text = str(error)
    config = target.get("config_path")

    if "unable to get local issuer certificate" in text:
        found = system_ca_bundle()

        return (
            "\n\nThe issuing CA is not in the bundle being verified against"
            f" ({target.get('ca_bundle') or 'certifi, the python default'}).\n"
            "A browser trusts a CA added with update-ca-trust; python does not,"
            " because it verifies\nagainst certifi's public-root bundle instead"
            " of the OS trust store.\n"
            + (
                f'Add  "ca_bundle": "{found}"  to {config}\n'
                if found
                else "No system CA bundle was found to point at.\n"
            )
        )

    if "Missing Authority Key Identifier" in text or "Missing Subject Key" in text:
        return (
            "\n\nThe certificate chain is missing an extension that RFC 5280"
            " requires. Python 3.13+\nenables VERIFY_X509_STRICT by default, so"
            " it rejects chains that openssl and browsers accept.\n\n"
            "The real fix is to reissue the server certificate with\n"
            "    authorityKeyIdentifier = keyid,issuer\n"
            "in its openssl extension section.\n\n"
            f'As a stopgap, add  "tls_strict": false  to {config}.\n'
        )

    return ""


def build_session(target: dict) -> requests.Session:
    session = requests.Session()
    ca_bundle = target.get("ca_bundle")

    # Set on the session in both cases: the relaxed path carries the bundle in
    # its SSLContext too, but leaving session.verify at its default would mean
    # requests and the context disagreed about which roots are trusted.
    if ca_bundle:
        session.verify = ca_bundle

    if not target.get("tls_strict", True):
        session.mount("https://", _RelaxedTlsAdapter(ca_bundle))
        log.warning(
            "tls_strict is off for %s: certificates are still verified, but the "
            "RFC 5280 checks Python 3.13+ applies by default are not.",
            target["environment"],
        )

    return session


class MawMediaClient:
    def __init__(self, target: dict, dry_run: bool = False):
        self.target = target
        self.dry_run = dry_run
        self.session = build_session(target)
        self._token = None

    @property
    def environment(self) -> str:
        return self.target["environment"]

    @property
    def stamps(self) -> bool:
        """Whether this run may record progress in the outbox.

        The outbox has ONE published_revision per row, and it means "what
        production holds".  If a dev run stamped it, the next production run
        would find an empty queue and publish nothing -- the site would silently
        stop updating.  So non-production runs send real data but write nothing
        locally, which also makes them repeatable.
        """
        return self.environment == "prod" and not self.dry_run

    def login(self) -> None:
        if self.dry_run:
            return

        auth_url = self.target["auth_url"]

        try:
            resp = self.session.post(
                f"{auth_url.rstrip('/')}/oauth/token",
                json={
                    "client_id": self.target["client_id"],
                    "client_secret": self.target["client_secret"],
                    "audience": self.target["audience"],
                    "grant_type": "client_credentials",
                },
                timeout=REQUEST_TIMEOUT,
            )
        except requests.RequestException as e:
            raise PublishError(
                f"Could not reach {auth_url}: {e}{tls_hint(e, self.target)}"
            ) from e

        if not resp.ok:
            raise PublishError(
                f"Auth failed ({resp.status_code}) for audience "
                f"{self.target['audience']} using {self.target['config_path']}: "
                f"{resp.text}"
            )

        self._token = resp.json().get("access_token")
        if not self._token:
            raise PublishError(f"Auth response carried no access_token: {resp.text}")

        self.session.headers["Authorization"] = f"Bearer {self._token}"
        log.info("Authenticated for audience %s", self.target["audience"])

    def post(self, path: str, payload: list) -> list[dict]:
        """POST one batch and return its per-item results.

        Every publish endpoint answers with the same
        (entity, entityId, outcome, detail) shape, so partial success is
        reported per row rather than sinking the batch.
        """
        if self.dry_run:
            # `would_send` deliberately is not in SETTLED, so a dry run cannot
            # stamp anything even if the stamping guard were removed.
            return [
                {
                    "entity": "dry-run",
                    "entityId": None,
                    "outcome": "would_send",
                    "detail": None,
                }
                for _ in payload
            ]

        url = f"{self.target['api_url'].rstrip('/')}{path}"

        try:
            resp = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)

            if resp.status_code == 401:
                # Tokens outlive a run in practice, but a long backfill can cross
                # the expiry; one retry after re-auth costs nothing.
                log.info("Token rejected; re-authenticating.")
                self.login()
                resp = self.session.post(url, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            # An unreachable host is an ordinary operator mistake -- wrong
            # environment, vpn down, service stopped -- so report it as one.
            # Nothing has been stamped, so the run resumes cleanly once fixed.
            raise PublishError(
                f"Could not reach the {self.environment} API at {url}: {e}"
                f"{tls_hint(e, self.target)}"
            ) from e

        if not resp.ok:
            raise PublishError(f"POST {url} -> {resp.status_code}: {resp.text}")

        return resp.json()


# --- Outcome handling --------------------------------------------------------
# Both mean maw-media holds this revision or newer, so the row is in sync and
# must be stamped.  Anything else is left unstamped and retried next run.
SETTLED = {"applied", "skipped_stale", "deleted", "not_found"}


def summarize(results: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        outcome = r.get("outcome", "?")
        counts[outcome] = counts.get(outcome, 0) + 1

    return counts


def check_forbidden(results: list[dict]) -> None:
    if any(r.get("outcome") == "forbidden" for r in results):
        raise PublishError(
            "The API rejected the batch as `forbidden`: the publisher's "
            "media.user row needs the admin role (media.get_is_admin), not just "
            "the face-recognition:publish scope."
        )


# --- Queries -----------------------------------------------------------------
PENDING_PERSONS_SQL = f"""
    SELECT p.id, p.name, p.status_code, p.preferred_face_id,
           pv.face_count, p.revision, p.updated_at
    FROM person p
    JOIN person_v pv ON pv.id = p.id
    WHERE {PERSON_IN_SCOPE}
      AND (p.published_revision IS NULL OR p.published_revision < p.revision)
    ORDER BY p.revision
"""

PENDING_FACES_SQL = f"""
    SELECT fd.id, m.file_path, fd.media_id, fd.person_id,
           fd.bounding_box, fd.detection_score, fd.revision
    FROM face_detection fd
    JOIN media m    ON m.id = fd.media_id
    JOIN person per ON per.id = fd.person_id
    WHERE {FACE_IN_SCOPE}
      AND (fd.published_revision IS NULL OR fd.published_revision < fd.revision)
    ORDER BY fd.revision
"""

# Published rows that have since left publish scope.  Distinct from a tombstone:
# the row still exists here, it simply no longer belongs on the website, so the
# projection has to lose it or the site keeps showing a person you unnamed.
RETRACT_FACES_SQL = """
    SELECT fd.id
    FROM face_detection fd
    LEFT JOIN person per ON per.id = fd.person_id
    WHERE fd.published_revision IS NOT NULL
      AND (per.id IS NULL OR per.name IS NULL)
"""

RETRACT_PERSONS_SQL = """
    SELECT p.id
    FROM person p
    WHERE p.published_revision IS NOT NULL
      AND p.name IS NULL
"""

TOMBSTONES_SQL = """
    SELECT entity_id, revision
    FROM deleted_entity
    WHERE entity_type = %s AND published_revision IS NULL
"""


def to_instant(ts) -> str | None:
    """Format for NodaTime's Instant, which is what PersonSync.SourceModified is.

    Its ExtendedIso pattern wants a literal `Z`, not the `+00:00` offset that
    datetime.isoformat() produces, so convert to UTC and swap the suffix.

    Not a stylistic preference: deserialising the offset form with the API's own
    JsonSerializerOptions throws
    "The JSON value could not be converted to ... PersonSync", so a bare
    isoformat() would 400 every person batch.
    """
    if ts is None:
        return None

    return ts.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def person_payload(row: dict) -> dict:
    """camelCase, per the API's global JSON options.  snake_case will not bind.

    An upsert replaces the whole row, so every field is sent every time --
    omitting one clears it rather than leaving it untouched.
    """
    name = row["name"]

    return {
        "id": str(row["id"]),
        "name": name,
        "slug": make_slug(name) if name else None,
        "statusCode": row["status_code"],
        "preferredFaceId": (
            str(row["preferred_face_id"]) if row["preferred_face_id"] else None
        ),
        "faceCount": row["face_count"],
        "sourceRevision": row["revision"],
        "sourceModified": to_instant(row["updated_at"]),
    }


def face_payload(row: dict) -> dict:
    # bounding_box already carries normalised 0..1 values alongside the pixel
    # ones; only the normalised pair is published, so a crop survives any scale.
    # A face clipped by the frame edge can sit slightly outside 0..1 -- that is
    # expected and NUMERIC(7,6) stores it fine.
    box = row["bounding_box"]

    return {
        "id": str(row["id"]),
        "filePath": to_web_path(row["file_path"]),
        "personId": str(row["person_id"]) if row["person_id"] else None,
        "boxX": box["x"],
        "boxY": box["y"],
        "boxWidth": box["width"],
        "boxHeight": box["height"],
        "detectionScore": row["detection_score"],
        "sourceRevision": row["revision"],
    }


def chunk(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


# --- Commands ----------------------------------------------------------------
def cmd_status() -> None:
    """What a publish would do, without touching the network.

    Takes no environment: the outbox records one thing, what production holds,
    so this is always production's queue.
    """
    with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) AS n FROM person p WHERE {PERSON_IN_SCOPE}")
            in_scope_persons = cur.fetchone()["n"]
            cur.execute(
                f"""SELECT count(*) AS n FROM face_detection fd
                    JOIN person per ON per.id = fd.person_id
                    WHERE {FACE_IN_SCOPE}"""
            )
            in_scope_faces = cur.fetchone()["n"]

            cur.execute(PENDING_PERSONS_SQL)
            persons = cur.fetchall()
            cur.execute(PENDING_FACES_SQL)
            faces = cur.fetchall()
            cur.execute(RETRACT_PERSONS_SQL)
            retract_p = cur.fetchall()
            cur.execute(RETRACT_FACES_SQL)
            retract_f = cur.fetchall()
            cur.execute(TOMBSTONES_SQL, ("person",))
            tomb_p = cur.fetchall()
            cur.execute(TOMBSTONES_SQL, ("face",))
            tomb_f = cur.fetchall()

            cur.execute(
                "SELECT count(*) AS n FROM media WHERE publish_error IS NOT NULL"
            )
            errors = cur.fetchone()["n"]

    print("Queue state for PRODUCTION (the only environment the outbox tracks)")
    print(f"Scope: {PERSON_IN_SCOPE}")
    print(f"  in scope: {in_scope_persons:,} persons, {in_scope_faces:,} faces")
    print("\nPending publish:")
    print(f"  persons        {len(persons):>8,}  ({-(-len(persons) // PERSON_BATCH)} requests)")
    print(f"  faces          {len(faces):>8,}  ({-(-len(faces) // FACE_BATCH)} requests)")
    print("\nPending removal:")
    print(f"  persons deleted{len(tomb_p):>8,}")
    print(f"  persons retract{len(retract_p):>8,}   (unnamed since publishing)")
    print(f"  faces deleted  {len(tomb_f):>8,}")
    print(f"  faces retract  {len(retract_f):>8,}   (no longer on a named person)")
    if errors:
        print(f"\n  {errors:,} media rows carry a publish_error (see media.publish_error)")


def publish_statuses(conn, client: MawMediaClient) -> None:
    """person_status must land before any person referencing it.

    Sent on every run even under a named-only scope (where status_code is always
    null): it is two rows, and it keeps the lookup correct the moment the scope
    widens or a new triage state is added here.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT code, label, description, sort_order FROM person_status "
            "ORDER BY sort_order"
        )
        rows = cur.fetchall()

    payload = [
        {
            "code": r["code"],
            "label": r["label"],
            "description": r["description"],
            "sortOrder": r["sort_order"],
        }
        for r in rows
    ]

    for batch in chunk(payload, STATUS_BATCH):
        results = client.post("/config/person-statuses/sync", batch)
        check_forbidden(results)

    log.info("Statuses: %d sent", len(payload))


def publish_persons(conn, client: MawMediaClient) -> dict:
    """Always the full pending set, never limited.

    A face whose person has not been published comes back `unknown_person`, so
    truncating persons to make a smaller test run would poison the face batches
    that follow.  Persons are cheap anyway -- 448 of them is a single request.
    """
    with conn.cursor() as cur:
        cur.execute(PENDING_PERSONS_SQL)
        rows = cur.fetchall()

    if not rows:
        log.info("Persons: nothing pending")
        return {}

    by_id = {str(r["id"]): r for r in rows}
    totals: dict[str, int] = {}

    for batch in chunk(rows, PERSON_BATCH):
        results = client.post("/persons/sync", [person_payload(r) for r in batch])
        check_forbidden(results)

        for outcome, n in summarize(results).items():
            totals[outcome] = totals.get(outcome, 0) + n

        if not client.stamps:
            continue

        settled = [r["entityId"] for r in results if r.get("outcome") in SETTLED]
        stamp_published(
            conn, "person", [(pid, by_id[pid]["revision"]) for pid in settled]
        )
        log.info("Persons: %d/%d accepted", len(settled), len(batch))

    return totals


def publish_faces(conn, client: MawMediaClient, limit: int | None) -> dict:
    with conn.cursor() as cur:
        cur.execute(PENDING_FACES_SQL)
        rows = cur.fetchall()

    if limit:
        rows = rows[:limit]
    if not rows:
        log.info("Faces: nothing pending")
        return {}

    by_id = {str(r["id"]): r for r in rows}
    totals: dict[str, int] = {}
    done = 0

    for batch in chunk(rows, FACE_BATCH):
        try:
            payload = [face_payload(r) for r in batch]
        except ValueError as e:
            # to_web_path raises when a file sits outside ASSET_ROOT, which means
            # the configured root is wrong -- a per-row skip would quietly
            # publish a fraction of the library.
            raise PublishError(
                f"A face path is not under ASSET_ROOT ({ASSET_ROOT}): {e}"
            ) from e

        results = client.post("/faces/sync", payload)
        check_forbidden(results)

        done += len(batch)

        for outcome, n in summarize(results).items():
            totals[outcome] = totals.get(outcome, 0) + n

        if not client.stamps:
            continue

        settled = [r["entityId"] for r in results if r.get("outcome") in SETTLED]
        record_path_failures(conn, results, by_id)

        stamp_published(
            conn, "face_detection", [(fid, by_id[fid]["revision"]) for fid in settled]
        )
        mark_media_published(conn, [by_id[fid]["media_id"] for fid in settled])

        log.info("Faces: %d/%d rows processed", done, len(rows))

    return totals


def publish_removals(conn, client: MawMediaClient) -> dict:
    """Faces first, then persons.

    Ordering is not required by the API -- ON DELETE SET NULL on
    media.face.person_id means dropping a person only unassigns its faces -- but
    doing faces first means a failure halfway leaves no orphaned face rows
    pointing at a person that is already gone.
    """
    totals: dict[str, int] = {}

    for entity, endpoint, table in (
        ("face", "/faces/deletions", "face_detection"),
        ("person", "/persons/deletions", "person"),
    ):
        with conn.cursor() as cur:
            cur.execute(TOMBSTONES_SQL, (entity,))
            tombstones = {str(r["entity_id"]): r["revision"] for r in cur.fetchall()}
            cur.execute(
                RETRACT_FACES_SQL if entity == "face" else RETRACT_PERSONS_SQL
            )
            retractions = [str(r["id"]) for r in cur.fetchall()]

        ids = list(tombstones) + retractions
        if not ids:
            continue

        for batch in chunk(ids, DELETION_BATCH):
            results = client.post(endpoint, batch)
            check_forbidden(results)

            for outcome, n in summarize(results).items():
                key = f"{entity}:{outcome}"
                totals[key] = totals.get(key, 0) + n

            if not client.stamps:
                continue

            settled = [r["entityId"] for r in results if r.get("outcome") in SETTLED]

            done_tombstones = [(i, tombstones[i]) for i in settled if i in tombstones]
            done_retractions = [i for i in settled if i not in tombstones]

            if done_tombstones:
                stamp_tombstones(conn, entity, done_tombstones)
            if done_retractions:
                # Back to "never published".  If the person is named again later,
                # the outbox offers it up as a fresh publish.
                clear_published(conn, table, done_retractions)

        log.info("%ss removed: %d", entity, len(ids))

    return totals


# --- Local bookkeeping -------------------------------------------------------
def stamp_published(conn, table: str, pairs: list[tuple[str, int]]) -> None:
    """Record the revision that was actually accepted.

    The revision from the payload, never the row's current `revision`: if the row
    changed between the read and this write, published_revision < revision still
    holds and it republishes -- which is the correct outcome, and the reason a
    concurrent edit during a publish cannot be lost.
    """
    if not pairs:
        return

    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {table} t
            SET published_revision = v.rev
            FROM (SELECT unnest(%s::uuid[]) AS id, unnest(%s::bigint[]) AS rev) v
            WHERE t.id = v.id
            """,
            ([p[0] for p in pairs], [p[1] for p in pairs]),
        )
    conn.commit()


def stamp_tombstones(conn, entity: str, pairs: list[tuple[str, int]]) -> None:
    if not pairs:
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE deleted_entity d
            SET published_revision = v.rev
            FROM (SELECT unnest(%s::uuid[]) AS id, unnest(%s::bigint[]) AS rev) v
            WHERE d.entity_type = %s AND d.entity_id = v.id
            """,
            ([p[0] for p in pairs], [p[1] for p in pairs], entity),
        )
    conn.commit()


def clear_published(conn, table: str, ids: list[str]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET published_revision = NULL WHERE id = ANY(%s::uuid[])",
            (ids,),
        )
    conn.commit()


def mark_media_published(conn, media_ids: list) -> None:
    if not media_ids:
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE media
            SET last_published_at = now(), publish_error = NULL
            WHERE id = ANY(%s::uuid[])
            """,
            ([str(m) for m in media_ids],),
        )
    conn.commit()


def record_path_failures(conn, results: list[dict], by_id: dict) -> None:
    """A path key means a rename or recategorisation silently breaks the link, so
    make it loud: the API reports unresolved_path per row and it is recorded
    against the media locally (§4.3) for later inspection."""
    broken = [
        (by_id[r["entityId"]]["media_id"], r.get("detail"))
        for r in results
        if r.get("outcome") == "unresolved_path" and r.get("entityId") in by_id
    ]
    if not broken:
        return

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE media
            SET publish_error = v.err
            FROM (SELECT unnest(%s::uuid[]) AS id, unnest(%s::text[]) AS err) v
            WHERE media.id = v.id
            """,
            (
                [str(b[0]) for b in broken],
                [f"unresolved_path: {b[1]}" for b in broken],
            ),
        )
    conn.commit()

    log.warning(
        "%d face(s) referenced a path maw-media could not resolve; recorded in "
        "media.publish_error",
        len(broken),
    )


def announce(client: MawMediaClient) -> None:
    """Say plainly what is about to be written to, before anything is."""
    target = client.target

    log.info("Target: %s -> %s", client.environment.upper(), target["api_url"])
    log.info("        audience %s", target["audience"])
    log.info("        config   %s", target["config_path"])
    log.info("        ca       %s", target.get("ca_bundle") or "certifi (default)")

    for key in target.get("overrides", []):
        log.warning(
            "        %s comes from the config file, overriding the --%s default",
            key,
            client.environment,
        )

    if client.dry_run:
        log.info("DRY RUN -- payloads are built and validated, nothing is sent")
    elif not client.stamps:
        log.info(
            "Non-production: rows are sent but the outbox is NOT stamped, so "
            "this run neither affects nor is affected by production's queue."
        )


def cmd_publish(
    environment: str, dry_run: bool, limit: int | None, config: str | None = None
) -> None:
    started = time.time()
    client = MawMediaClient(
        resolve_target(environment, config, require_config=not dry_run),
        dry_run=dry_run,
    )

    announce(client)

    if not dry_run:
        client.login()

    with psycopg.connect(DB_DSN, row_factory=dict_row) as conn:
        # Order matters, and the caller owns it: each endpoint is its own
        # transaction.  Statuses before persons (status_code is a foreign key),
        # persons before faces (person_id is a foreign key), removals last.
        publish_statuses(conn, client)
        p_totals = publish_persons(conn, client)
        f_totals = publish_faces(conn, client, limit)
        r_totals = publish_removals(conn, client)

    log.info("Done in %.1fs", time.time() - started)
    for label, totals in (
        ("persons", p_totals),
        ("faces", f_totals),
        ("removals", r_totals),
    ):
        if totals:
            detail = ", ".join(f"{k}={v:,}" for k, v in sorted(totals.items()))
            log.info("  %-9s %s", label, detail)

    if not dry_run:
        log.info("Re-run `publish-faces.py status` to confirm the queue drained.")


def cmd_reset(persons: bool, faces: bool) -> None:
    """Forget what has been published, forcing a full republish.

    Only ever touches this side.  maw-media keeps its rows and its revision
    guard, so a republish of unchanged data lands as skipped_stale -- harmless,
    but it does mean this cannot be used to *repair* maw-media, only to resend.
    """
    with psycopg.connect(DB_DSN) as conn:
        with conn.cursor() as cur:
            if persons:
                cur.execute("UPDATE person SET published_revision = NULL")
                log.info("Persons reset: %d", cur.rowcount)
            if faces:
                cur.execute("UPDATE face_detection SET published_revision = NULL")
                log.info("Faces reset: %d", cur.rowcount)
            cur.execute("UPDATE deleted_entity SET published_revision = NULL")
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show what is pending; makes no network calls")

    p_pub = sub.add_parser("publish", help="Publish pending changes to maw-media")

    # Mutually exclusive and REQUIRED: there is no default target, so nothing
    # reaches the live site without the word `prod` appearing in the command.
    env_group = p_pub.add_mutually_exclusive_group(required=True)
    env_group.add_argument(
        "--prod",
        dest="environment",
        action="store_const",
        const="prod",
        help="Publish to production (media.mikeandwan.us) -- the live website",
    )
    env_group.add_argument(
        "--staging",
        dest="environment",
        action="store_const",
        const="staging",
        help="Publish to staging",
    )
    env_group.add_argument(
        "--dev",
        dest="environment",
        action="store_const",
        const="dev",
        help="Publish to dev; sends real data but never stamps the outbox",
    )

    p_pub.add_argument(
        "--config",
        help=f"Config file to use instead of {CONFIG_DIR}/<environment>/config.json",
    )
    p_pub.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate every payload but POST nothing",
    )
    p_pub.add_argument(
        "--limit",
        type=int,
        help="Publish at most this many FACES, for a careful first run. Persons "
        "are always sent in full, since a face whose person is missing is "
        "rejected as unknown_person.",
    )

    p_reset = sub.add_parser(
        "reset", help="Clear published_revision so everything republishes"
    )
    p_reset.add_argument("--persons", action="store_true")
    p_reset.add_argument("--faces", action="store_true")
    p_reset.add_argument("--all", action="store_true")

    args = parser.parse_args()

    try:
        if args.command == "status":
            cmd_status()
        elif args.command == "publish":
            cmd_publish(args.environment, args.dry_run, args.limit, args.config)
        elif args.command == "reset":
            if not (args.persons or args.faces or args.all):
                parser.error("reset needs --persons, --faces or --all")
            cmd_reset(args.persons or args.all, args.faces or args.all)
    except PublishError as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
