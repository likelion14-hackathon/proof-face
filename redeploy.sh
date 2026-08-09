#!/usr/bin/env bash
#
# Tear down the running stack, rebuild the image, start it again, wait until the
# API answers.
#
#   ./redeploy.sh                     rebuild + restart (layer cache reused)
#   ./redeploy.sh --no-cache          rebuild from scratch (slow, ~5 min)
#   ./redeploy.sh --logs              follow the logs once it is healthy
#   SKIN_METRICS_PORT=8100 ./redeploy.sh    when 8000 is taken
#
# Only ever touches this compose project: containers come down via `docker
# compose down`, and image cleanup is filtered to the skin-metrics-api label.
# Other projects' containers, images and build cache are left alone.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${SKIN_METRICS_PORT:-8000}"
SERVICE="api"
LABEL="org.opencontainers.image.title=skin-metrics-api"
BUILD_ARGS=()
FOLLOW_LOGS=0
HEALTH_TIMEOUT=120

for arg in "$@"; do
    case "$arg" in
        --no-cache) BUILD_ARGS+=("--no-cache") ;;
        --logs)     FOLLOW_LOGS=1 ;;
        -h|--help)  sed -n '3,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
    esac
done

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
fail() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }

# --- preflight ------------------------------------------------------------
docker version --format '{{.Server.Version}}' > /dev/null 2>&1 \
    || fail "Docker daemon is not responding. Start Docker Desktop and retry."

# A port clash shows up as a confusing bind error halfway through `up`; catch it
# now. `skin-metrics serve` running locally is the usual culprit.
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN > /dev/null 2>&1; then
    holder=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -F c 2>/dev/null | sed -n 's/^c//p' | head -1)
    # Our own container holding the port is fine - `down` frees it below.
    if [ "${holder:-}" != "com.docker.backend" ] && [ "${holder:-}" != "docker" ]; then
        fail "Port $PORT is already in use by '${holder:-unknown}'.
       Stop it, or pick another port:  SKIN_METRICS_PORT=8100 $0"
    fi
fi

# --- 1. stop the previous stack -------------------------------------------
log "Stopping previous stack"
docker compose down --remove-orphans

# --- 2. build --------------------------------------------------------------
log "Building image${BUILD_ARGS[+]+ (${BUILD_ARGS[*]})}"
docker compose build "${BUILD_ARGS[@]+"${BUILD_ARGS[@]}"}" "$SERVICE"

# --- 3. start --------------------------------------------------------------
log "Starting"
docker compose up -d "$SERVICE"

# --- 4. wait for the health check to pass ----------------------------------
log "Waiting for health check (up to ${HEALTH_TIMEOUT}s)"
cid=$(docker compose ps -q "$SERVICE")
[ -n "$cid" ] || fail "Service '$SERVICE' did not start."

deadline=$((SECONDS + HEALTH_TIMEOUT))
while :; do
    status=$(docker inspect --format '{{.State.Health.Status}}' "$cid" 2>/dev/null || echo "gone")
    case "$status" in
        healthy) echo "healthy"; break ;;
        unhealthy)
            docker compose logs --tail 30 "$SERVICE"
            fail "Container reported unhealthy."
            ;;
        gone)
            docker compose logs --tail 30 "$SERVICE"
            fail "Container exited during startup."
            ;;
    esac
    [ "$SECONDS" -lt "$deadline" ] || {
        docker compose logs --tail 30 "$SERVICE"
        fail "Still '$status' after ${HEALTH_TIMEOUT}s."
    }
    sleep 3
done

# --- 5. drop images this rebuild orphaned (ours only) ----------------------
log "Removing images orphaned by this rebuild"
docker image prune -f --filter "label=$LABEL" | tail -1

# --- 6. report -------------------------------------------------------------
log "Ready"
docker compose ps --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}'
echo
curl -fsS "http://127.0.0.1:${PORT}/healthz" | python3 -m json.tool --no-ensure-ascii 2>/dev/null \
    || echo "(could not reach /healthz on port ${PORT})"
cat <<EOF

  API   http://127.0.0.1:${PORT}
  Docs  http://127.0.0.1:${PORT}/docs

  logs  docker compose logs -f ${SERVICE}
  stop  docker compose down
EOF

if [ "$FOLLOW_LOGS" -eq 1 ]; then
    log "Following logs (Ctrl-C to detach; container keeps running)"
    docker compose logs -f "$SERVICE"
fi
