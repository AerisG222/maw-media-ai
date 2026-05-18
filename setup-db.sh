#!/usr/bin/env bash
# =============================================================================
# setup-db.sh
# Starts a Postgres 16 + pgvector container via Podman and applies the schema.
#
# Usage:
#   ./setup-db.sh            # start container + apply schema
#   ./setup-db.sh start      # start an already-created container
#   ./setup-db.sh stop       # stop the container (data is preserved)
#   ./setup-db.sh destroy    # stop + delete container AND volume (all data lost)
#   ./setup-db.sh logs       # tail container logs
#   ./setup-db.sh psql       # open an interactive psql session
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — edit these or override via environment variables
# ---------------------------------------------------------------------------
CONTAINER_NAME="${CONTAINER_NAME:-face_scanner_db}"
VOLUME_NAME="${VOLUME_NAME:-face_scanner_pgdata}"
PG_IMAGE="${PG_IMAGE:-docker.io/pgvector/pgvector:pg18-trixie}"
PG_PORT="${PG_PORT:-5433}"          # host port — 5433 avoids conflict with any local pg
PG_DB="${PG_DB:-face_scanner}"
PG_USER="${PG_USER:-face_scanner}"
PG_PASSWORD="${PG_PASSWORD:-face_scanner_secret}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEMA_FILE="${SCRIPT_DIR}/schema.sql"

DSN="postgresql://${PG_USER}:${PG_PASSWORD}@localhost:${PG_PORT}/${PG_DB}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()  { echo "  [INFO]  $*"; }
ok()    { echo "  [OK]    $*"; }
err()   { echo "  [ERROR] $*" >&2; exit 1; }

require_podman() {
    command -v podman &>/dev/null || err "podman is not installed or not in PATH."
}

require_psql() {
    command -v psql &>/dev/null || {
        echo ""
        echo "  [WARN]  psql not found. Schema will not be applied automatically."
        echo "          Install it with:  sudo apt install postgresql-client"
        echo "          Then run manually: psql '${DSN}' -f schema.sql"
        echo ""
        return 1
    }
    return 0
}

container_exists() {
    podman container exists "${CONTAINER_NAME}" 2>/dev/null
}

container_running() {
    [[ "$(podman inspect --format '{{.State.Status}}' "${CONTAINER_NAME}" 2>/dev/null)" == "running" ]]
}

wait_for_postgres() {
    local attempts=30
    info "Waiting for Postgres to be ready…"
    for i in $(seq 1 $attempts); do
        if podman exec "${CONTAINER_NAME}" pg_isready -U "${PG_USER}" -d "${PG_DB}" &>/dev/null; then
            ok "Postgres is ready."
            return 0
        fi
        sleep 1
    done
    err "Postgres did not become ready after ${attempts}s. Check logs: $0 logs"
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_start() {
    require_podman

    if container_running; then
        ok "Container '${CONTAINER_NAME}' is already running."
        print_env
        return
    fi

    if container_exists; then
        info "Starting existing container '${CONTAINER_NAME}'…"
        podman start "${CONTAINER_NAME}"
        wait_for_postgres
        print_env
        return
    fi

    info "Pulling image ${PG_IMAGE}…"
    podman pull "${PG_IMAGE}"

    info "Creating volume '${VOLUME_NAME}'…"
    podman volume create "${VOLUME_NAME}" &>/dev/null || true

    info "Creating container '${CONTAINER_NAME}'…"
    podman run -d \
        --name "${CONTAINER_NAME}" \
        --volume "${VOLUME_NAME}:/var/lib/postgresql:Z" \
        --publish "127.0.0.1:${PG_PORT}:5432" \
        --env "POSTGRES_DB=${PG_DB}" \
        --env "POSTGRES_USER=${PG_USER}" \
        --env "POSTGRES_PASSWORD=${PG_PASSWORD}" \
        --restart unless-stopped \
        "${PG_IMAGE}"

    wait_for_postgres
    apply_schema
    print_env
}

cmd_stop() {
    require_podman
    if container_exists; then
        info "Stopping container '${CONTAINER_NAME}'…"
        podman stop "${CONTAINER_NAME}"
        ok "Container stopped. Data is preserved in volume '${VOLUME_NAME}'."
    else
        info "Container '${CONTAINER_NAME}' does not exist."
    fi
}

cmd_destroy() {
    require_podman
    echo ""
    echo "  WARNING: This will permanently delete the container AND all data."
    read -rp "  Type 'yes' to confirm: " confirm
    [[ "${confirm}" == "yes" ]] || { info "Aborted."; exit 0; }

    podman stop "${CONTAINER_NAME}" 2>/dev/null || true
    podman rm   "${CONTAINER_NAME}" 2>/dev/null || true
    podman volume rm "${VOLUME_NAME}" 2>/dev/null || true
    ok "Container and volume deleted."
}

cmd_logs() {
    require_podman
    podman logs -f "${CONTAINER_NAME}"
}

cmd_psql() {
    require_podman
    container_running || err "Container '${CONTAINER_NAME}' is not running. Run: $0 start"
    podman exec -it "${CONTAINER_NAME}" \
        psql -U "${PG_USER}" -d "${PG_DB}"
}

apply_schema() {
    if [[ ! -f "${SCHEMA_FILE}" ]]; then
        echo "  [WARN]  schema.sql not found at ${SCHEMA_FILE} — skipping."
        return
    fi

    if require_psql; then
        info "Applying schema from ${SCHEMA_FILE}…"
        PGPASSWORD="${PG_PASSWORD}" psql \
            -h localhost -p "${PG_PORT}" \
            -U "${PG_USER}" -d "${PG_DB}" \
            -f "${SCHEMA_FILE}"
        ok "Schema applied."
    fi
}

print_env() {
    echo ""
    echo "  ┌─────────────────────────────────────────────────────────────┐"
    echo "  │  Postgres is running                                        │"
    echo "  │                                                             │"
    echo "  │  Host port : ${PG_PORT}                                          │"
    echo "  │  Database  : ${PG_DB}                                 │"
    echo "  │  User      : ${PG_USER}                               │"
    echo "  │                                                             │"
    echo "  │  DSN for scan_faces.py:                                     │"
    printf "  │  %-59s │\n" "FACE_SCANNER_DSN=\"${DSN}\""
    echo "  └─────────────────────────────────────────────────────────────┘"
    echo ""
    echo "  export FACE_SCANNER_DSN=\"${DSN}\""
    echo ""
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "${1:-}" in
    ""| start)  cmd_start   ;;
    stop)       cmd_stop    ;;
    destroy)    cmd_destroy ;;
    logs)       cmd_logs    ;;
    psql)       cmd_psql    ;;
    *)
        echo "Usage: $0 [start|stop|destroy|logs|psql]"
        exit 1
        ;;
esac
