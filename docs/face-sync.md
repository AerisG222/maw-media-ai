# Face Sync Design

How face and person data produced by this project is published to
[maw-media](https://github.com/mmorano/maw-media), and how user-submitted
corrections make their way back here.

**Status:** design, not yet implemented.

---

## Goal

Let visitors to media.mikeandwan.us search for photos and videos containing
their friends and family, without moving any of the expensive work onto the
web server.

Two hard constraints shape everything below:

1. **No pgvector on the web server.** maw-media runs stock Postgres. Embeddings
   and similarity search stay here.
2. **No CPU-intensive work on the web server.** Detection, clustering,
   suggestion scoring and merging all run on the desktop, offline, on a GPU.

The consequence is that maw-media receives a *projection* — the conclusions of
the pipeline, not its inputs. No embedding ever crosses the wire.

---

## 1. Ownership model

The single most important decision, because everything else follows from it:
**each row has exactly one writer.**

| Data | Owner | The other side |
|------|-------|----------------|
| media, persons, clusters, faces, embeddings | maw-media-ai | maw-media holds a read-only projection |
| user suggestions and their review state | maw-media | maw-media-ai pulls, applies, acknowledges |

Users on the website never edit a person or a face. They file a **suggestion**,
which is a row in a table maw-media owns outright. The suggestion travels up to
this project, an operator reviews it, and the result comes back down as new
authoritative state.

This is deliberately *not* bidirectional sync. There is no merge logic, no
last-writer-wins, and no divergence to reconcile. Given that a single `cluster`
run can rewrite `person_id` on tens of thousands of rows at once, any design
where both sides could edit the same row would be painful and probably wrong.

---

## 2. Identity

### Persons and faces: shared UUIDs

`person.id` and `face_detection.id` are used verbatim as maw-media's
`media.person.id` and `media.face.id`. Both systems already use UUID primary
keys with no server-side default, so this costs nothing and means:

- every publish is a plain idempotent upsert keyed by primary key
- no mapping table exists, and none can drift

### Media: resolved by path, never stored

This project does **not** store maw-media's media id. Instead the sync payload
carries the file path, and maw-media resolves its own id inside the publish
transaction.

Rationale:

- A stored id would be a cache of derived data, and caches go stale. (See the
  `person_v` comment in `schema.sql` for the same lesson learned the hard way
  with `face_count`.)
- This project never needs to know maw-media exists until publish time. The
  scanner stays fully offline and there is no bootstrap ordering between the
  two systems.
- If maw-media's database is ever rebuilt with fresh media UUIDs, republishing
  fixes everything. A cached id would need an invalidation sweep.

The cost — an extra lookup per publish — is negligible when done set-based
(see §5).

> **Naming trap.** `face_detection.media_id` here refers to
> *this* database's local `media.id`. maw-media's `media.face.media_id` refers
> to *its* media id. Same column name, different values. They never meet,
> because the wire format carries `file_path` and never `media_id`.

### The path contract

maw-media builds `media.file.path` in `MawMediaPublisher`
(`Category.BuildMediaFilePath`) as:

```
/assets/{year}/{base-directory-name}/{scale-code}/{filename}
```

The scanner reads a local copy of that same tree, rooted differently:

```
local:   /data/maw-media-assets/{year}/{dir}/full/{filename}
stored:  /assets/{year}/{dir}/full/{filename}
```

So the transform is a root swap. Use `relative_to` rather than a string
replace, so a trailing slash on the configured root cannot produce a doubled
separator:

```python
ASSET_ROOT = os.getenv("ASSET_ROOT", "/data/maw-media-assets")
WEB_PREFIX = os.getenv("WEB_ASSET_PREFIX", "/assets")

def to_web_path(file_path: str) -> str:
    rel = Path(file_path).resolve().relative_to(Path(ASSET_ROOT).resolve())
    return f"{WEB_PREFIX}/{rel.as_posix()}"
```

The scale segment is pinned to `full` by `iter_images`. That filter should be
tightened from `"full" in p.parts` to `p.parent.name == "full"` — same result
today, but it enforces the structural guarantee the join key now depends on
rather than pattern-matching for it anywhere in the path.

**Whichever scale is scanned becomes part of the join key.** Switching to a
different scale later changes every path and forces a full republish. Not
fatal, but choose deliberately. Bounding boxes are normalised, so *rendering*
is scale-independent regardless; this affects only the join.

---

## 3. maw-media schema additions

New tables in the `media` schema, following that project's conventions
(`tables/media.*.sql`, snake_case, `TIMESTAMPTZ`, audit columns, named
constraints, `GRANT ... TO maw_media`).

### media.person_status

Mirrors this project's `person_status` lookup, same codes (`unknown`,
`not_a_person`). A lookup table rather than a `CHECK` so a new state costs an
upsert rather than a migration, and the API can expose the valid values.

**Not seeded on the maw-media side.** This project owns the codes, so they ride
along with a publish like everything else and the table starts empty. That keeps
a single source of truth — adding a status here needs no matching deploy over
there — at the cost of an ordering requirement on the sync: `media.person`
`status_code` is a foreign key, so statuses must be upserted **before** the
persons referencing them, in the same transaction. `POST /face/sync` should
therefore accept statuses as a third collection alongside `persons` and `faces`,
and apply them first.

### media.person

```sql
CREATE TABLE IF NOT EXISTS media.person (
    id UUID NOT NULL,                  -- same uuid as maw-media-ai person.id
    name TEXT,                         -- null until the operator labels the cluster
    slug TEXT,                         -- for /person/{slug} urls
    status_code TEXT,
    preferred_face_id UUID,            -- fk added after media.face exists
    face_count INTEGER NOT NULL DEFAULT 0,
    source_revision BIGINT NOT NULL,   -- monotonic revision from this project
    source_modified TIMESTAMPTZ,       -- our clock, informational only
    published TIMESTAMPTZ NOT NULL,    -- when the publish was accepted
    deleted TIMESTAMPTZ,               -- soft delete, keeps suggestions valid
    merged_into_id UUID,               -- set when a cluster is merged away

    CONSTRAINT pk_media_person PRIMARY KEY (id),
    CONSTRAINT uq_media_person$slug UNIQUE (slug),
    CONSTRAINT fk_media_person$media_person_status
        FOREIGN KEY (status_code) REFERENCES media.person_status(code),
    CONSTRAINT fk_media_person$media_person$merged
        FOREIGN KEY (merged_into_id) REFERENCES media.person(id)
);
```

Neither `media.person` nor `media.face` carries `created` / `modified` audit
columns, unlike most tables in that schema. Nothing in either is user authored —
every row is written by the same publisher service account, so `created_by` /
`modified_by` would record a constant. `published` (last accepted here) and
`source_modified` (last changed upstream) are the pair that actually matters
when reconciling the two systems.

### media.face

```sql
CREATE TABLE IF NOT EXISTS media.face (
    id UUID NOT NULL,                  -- same uuid as face_detection.id
    media_id UUID NOT NULL,            -- resolved from file_path at publish
    person_id UUID,
    -- normalised 0..1 so crops survive any scale; no embedding crosses over
    box_x NUMERIC(7,6) NOT NULL,
    box_y NUMERIC(7,6) NOT NULL,
    box_width NUMERIC(7,6) NOT NULL,
    box_height NUMERIC(7,6) NOT NULL,
    detection_score REAL NOT NULL,
    source_revision BIGINT NOT NULL,
    published TIMESTAMPTZ NOT NULL,
    deleted TIMESTAMPTZ,

    CONSTRAINT pk_media_face PRIMARY KEY (id),
    CONSTRAINT fk_media_face$media_media
        FOREIGN KEY (media_id) REFERENCES media.media(id),
    CONSTRAINT fk_media_face$media_person
        FOREIGN KEY (person_id) REFERENCES media.person(id)
);

CREATE INDEX ix_media_face$media_id ON media.face(media_id);
CREATE INDEX ix_media_face$person_id ON media.face(person_id)
    WHERE person_id IS NOT NULL;
```

### media.face_suggestion

The only table maw-media genuinely owns.

```sql
CREATE TABLE IF NOT EXISTS media.face_suggestion (
    id UUID NOT NULL,
    type_code TEXT NOT NULL,      -- name_person | assign_face | wrong_person | not_a_face
    status_code TEXT NOT NULL,    -- pending | claimed | applied | rejected | superseded
    face_id UUID,                 -- null for cluster-level suggestions
    person_id UUID,               -- the person the suggestion is about
    suggested_person_id UUID,     -- "this is actually <existing person>"
    suggested_name TEXT,          -- "...or this new name"
    note TEXT,
    created TIMESTAMPTZ NOT NULL,
    created_by UUID NOT NULL,
    modified TIMESTAMPTZ NOT NULL,
    modified_by UUID NOT NULL,
    claimed TIMESTAMPTZ,          -- when this project pulled it
    resolved TIMESTAMPTZ,         -- when this project reported back
    resolution_note TEXT,

    CONSTRAINT pk_media_face_suggestion PRIMARY KEY (id)
    -- fks to media.face, media.person, media.user, and the lookup tables
);

CREATE INDEX ix_media_face_suggestion$pending
    ON media.face_suggestion(created) WHERE status_code = 'pending';
```

Plus `media.face_suggestion_type` and `media.face_suggestion_status` lookup
tables, following the `person_status` precedent.

### Ordering note

`person.preferred_face_id → face.id` and `face.person_id → person.id` are
circular. Add the `preferred_face_id` foreign key in a `DO $$` block in
`tables/media.face.sql` after both tables exist, the way maw-media handles the
`media.location` unique constraint.

---

## 4. maw-media-ai schema changes

### 4.1 Publish queue (outbox)

```sql
CREATE SEQUENCE IF NOT EXISTS revision_seq;

ALTER TABLE person ADD COLUMN IF NOT EXISTS
    revision BIGINT NOT NULL DEFAULT nextval('revision_seq');
ALTER TABLE person ADD COLUMN IF NOT EXISTS published_revision BIGINT;

ALTER TABLE face_detection ADD COLUMN IF NOT EXISTS
    revision BIGINT NOT NULL DEFAULT nextval('revision_seq');
ALTER TABLE face_detection ADD COLUMN IF NOT EXISTS published_revision BIGINT;

-- "needs publishing" is simply:
--   published_revision IS NULL OR published_revision < revision
CREATE INDEX IF NOT EXISTS person_unpublished_idx ON person(revision)
    WHERE published_revision IS NULL OR published_revision < revision;
```

A trigger bumps `revision = nextval('revision_seq')` when a *meaningful* column
changes — `name`, `person_id`, `status_code`, `bounding_box`. Deliberately
**not** `embedding`, so a re-scan producing identical assignments does not
cause a republish.

### 4.2 Tombstones

`cluster` deletes and merges persons, and absence from a batch cannot express
that. Deletions must be explicit:

```sql
CREATE TABLE IF NOT EXISTS deleted_entity (
    entity_type TEXT NOT NULL,     -- 'person' | 'face'
    entity_id   UUID NOT NULL,
    merged_into UUID,              -- for cluster merges
    revision    BIGINT NOT NULL DEFAULT nextval('revision_seq'),
    published_revision BIGINT,
    deleted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),

    PRIMARY KEY (entity_type, entity_id)
);
```

### 4.3 Publish diagnostics

Deliberately *not* a cached media id — these are never read as truth, only for
observability:

```sql
ALTER TABLE media ADD COLUMN IF NOT EXISTS last_published_at TIMESTAMPTZ;
ALTER TABLE media ADD COLUMN IF NOT EXISTS publish_error TEXT;  -- e.g. 'path not found'
```

### 4.4 Inbound suggestions

```sql
CREATE TABLE IF NOT EXISTS inbound_suggestion (
    id UUID PRIMARY KEY,           -- maw-media's face_suggestion.id
    type_code TEXT NOT NULL,
    face_id UUID,
    person_id UUID,
    suggested_person_id UUID,
    suggested_name TEXT,
    note TEXT,
    submitted_by TEXT,
    submitted_at TIMESTAMPTZ,
    pulled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolution TEXT,               -- applied | rejected; null = awaiting operator
    resolved_at TIMESTAMPTZ,
    acked_at TIMESTAMPTZ           -- when the api was told the outcome
);
```

### 4.5 Confirmed labels

Applied suggestions must become **durable constraints**, not one-time edits —
otherwise the next `cluster` run discards the human's correction.

```sql
CREATE TABLE IF NOT EXISTS face_label (
    face_id   UUID PRIMARY KEY REFERENCES face_detection(id) ON DELETE CASCADE,
    person_id UUID NOT NULL REFERENCES person(id) ON DELETE CASCADE,
    source    TEXT NOT NULL,       -- 'operator' | 'suggestion'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

`cluster`, `suggest` and `merge-clusters` treat these as fixed points. This is
the piece that makes the feedback loop actually improve results rather than
just churn.

---

## 5. Sync mechanics

### Why not timestamps

An obvious approach is `WHERE modified > @last_seen` in both directions. It is
rejected here because:

- Two databases on two machines have two clocks. Comparing across that boundary
  is only safe if the watermark came from the same clock that writes the
  column, which is easy to get subtly wrong.
- Even within one database, a transaction that starts before the watermark and
  commits after it becomes invisible forever. That is silent data loss.
- Re-clustering updates enormous numbers of rows with near-identical
  timestamps, making boundary ties routine rather than theoretical.

Timestamps are kept for audit and display. They do not drive sync.

### Explicit queue state instead

Both directions use a queue that is self-correcting:

- **ai → api:** publish rows where `published_revision < revision`. The
  publisher claims a batch, POSTs it, and stamps `published_revision` only
  after a 2xx. A crash anywhere simply republishes.
- **api → ai:** `status_code = 'pending'` *is* the queue. Pull → mark `claimed`
  → operator resolves → ack.

Delivery is **at-least-once** and every handler must be idempotent. That is
satisfied by upserting on the shared primary key, with `source_revision` as an
idempotency guard: maw-media rejects any payload whose revision is `<=` what it
already stored, making retries and out-of-order batches safe without
coordination.

### Path resolution is set-based, inside the publish transaction

The expensive version of path resolution is one lookup per face over HTTP. The
correct version is a single join inside the publish function:

```sql
-- inside media.sync_faces(_user_id UUID, _payload JSONB)
WITH incoming AS (
    SELECT *
    FROM jsonb_to_recordset(_payload) AS x(
        id UUID,
        file_path TEXT,
        person_id UUID,
        box_x NUMERIC(7,6),
        -- ...
        source_revision BIGINT
    )
)
SELECT i.*, f.media_id
FROM incoming i
LEFT JOIN media.file f ON f.path = i.file_path
```

`ix_media_file$path` makes that one index scan per batch, microseconds against
the network round trip already being paid.

Resolving *inside* the publish transaction also closes a gap that a separate
`POST /media/resolve-paths` step would leave open: a file recategorised between
resolve and publish. It saves a round trip as well.

`media.file.path` has no unique constraint. Collisions should be impossible
given year + directory + scale + filename, but resolution must still handle a
multi-row match deterministically rather than silently fanning one face out
into several rows.

### Broken links must be loud

A path key means a rename or recategorisation silently breaks the link. Two
cheap mitigations:

1. `/face/sync` returns per-item results including an `unresolved` list, so a
   broken link is a number in the publisher's output rather than a gap.
2. `media.publish_error` (§4.3) records it locally for later inspection.

### Drift reconciliation

A periodic `GET /face/sync/state` returns counts and max revision per entity,
so divergence is detectable without a full republish.

---

## 6. API surface

Mirrors the existing location precedent (`missing-metadata` / `update-metadata`
plus machine-to-machine Auth0 identity per `docs/machine-to-machine-auth.sql`).

### Publisher-facing — new `face:publish` scope, new m2m app and `media.user` row

| Endpoint | Purpose |
|----------|---------|
| `POST /face/sync` | One transactional batch: `{ statuses, persons, faces, deletions }`, applied in that order |
| `GET /face/suggestions?status=pending&limit=n` | Pull the review queue |
| `POST /face/suggestions/resolve` | Report `applied` / `rejected` |
| `GET /face/sync/state` | Counts and max revision, for reconciliation |

`POST /face/sync` is implemented as a plpgsql function taking `_payload JSONB`
and using `jsonb_to_recordset`, consistent with maw-media's existing use of
jsonb.

### User-facing — `face:read` / `face:write`

| Endpoint | Purpose |
|----------|---------|
| `GET /media/{id}/faces` | Faces on a media item |
| `GET /person` | Browse named people |
| `GET /person/{id}/media` | Media containing a person |
| `POST /face/{id}/suggestion` | File a suggestion or correction |

### Security

`GET /person/{id}/media` **must** filter through the existing
`media.user_media` view. Without that join the person index becomes a side
channel revealing the existence and contents of categories the user has no role
for. The same applies to `face_count`, which should be a per-user count or it
leaks cardinality.

Face data is privacy-sensitive. Starting admin-only and widening later is the
cheaper mistake to unwind.

### Rendering

Only the normalised bounding box is published. Crops are drawn client-side
against the existing scaled files. This avoids an entire face-thumbnail asset
pipeline, and means a re-scan that shifts a box slightly is a metadata update
rather than a file regeneration.

---

## 7. Videos

The near-term goal is only **"is this person in this video"** — membership, not
temporal tracking.

That works today with no schema change at all, and the existing publisher
already lines this up:

- `ScaleSpec.AllScales` includes a poster variant at scale code `full`, so a
  video's poster is written to the same `full/` directory the scanner walks.
- `VideoScaler` names it `{name}.poster.avif`, and `.avif` is already in the
  scanner's `IMAGE_EXTENSIONS`. The sibling `{name}.mp4` is skipped.
- `SqlWriter.SqlFileMediaType` gives that row the video-poster `type_id` while
  keeping the *video's* `media_id`.

So the poster is scanned today, and its path resolves to the video's
`media_id`. Semantically that is "a face visible in the cover frame," which is
exactly right.

So videos get partial, zero-effort coverage in the first release: enough to
prove the search experience end to end before writing any ffmpeg frame
extraction.

Any UI assuming "faces belong to photos" needs to tolerate face rows pointing
at videos. If that is unwanted initially, the resolution join can filter on
`media.type`.

### When this needs revisiting

The trigger is **not** wanting to track people through a video. It is
**detecting faces on any frame other than the poster.**

`media.face` stores a normalised box with no image reference, so the crop is
drawn against the media's own scaled files. A box from the poster renders
correctly. A box from frame 4,213 renders nonsense against the poster, so
frame-sampled detection needs either a `frame_time NUMERIC(9,3)` column or a
published crop image.

Adding a nullable column later is an instant `ALTER TABLE`, and adding an
optional field to a JSON payload is additive and non-breaking. There is no
meaningful cost to deferring it, so it is deferred.

Deduplication for video — one row per person rather than one per detection — is
a publishing decision on this side, not a schema concern. `media.face` stays
one row per published detection either way.

---

## 8. Phasing

1. **maw-media tables** — `media.person`, `media.face`, `media.face_suggestion`
   and lookups, plus `media.sync_faces` with path resolution.
2. **Publish path** — outbox columns and triggers here, `POST /face/sync`, a
   publisher CLI subcommand. Read-only in maw-media, no UI yet.
3. **Read APIs and UI** — faces on media detail, person browse, person search.
4. **Suggestion loop** — suggestion tables, submit/pull/resolve, and the
   `face_label` constraint table that makes corrections stick.

---

## 9. Open questions

- **Publish policy.** All detections, or exclude `not_a_person` clusters?
  Leaning toward excluding those and publishing unnamed clusters, so visitors
  can help name them.
- **Who reviews?** The Streamlit app is where the operator already has crops,
  embeddings and merge tooling, so review belongs there and maw-media just
  collects. If non-admin suggestions need spam triage before reaching the
  desktop, an admin review state in maw-media is worth adding.
- **Visibility.** Admin-only to start, or all authenticated users?
- **Person slugs.** Generated here or in maw-media? Names are not unique, so
  the slug needs a disambiguation strategy.
