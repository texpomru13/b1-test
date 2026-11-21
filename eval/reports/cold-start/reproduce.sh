#!/bin/sh
# Run from a clone containing the measured commit. Requires macOS, Docker
# Desktop, host Ollama with Metal, curl, nc and jq. Uses two free local ports.
set -eu
if [ "$#" != 2 ]; then
    echo "Usage: $0 <free Ollama port> <free API port>" >&2
    exit 2
fi
host_port=$1
api_port=$2
[ "$host_port" != "$api_port" ]
for port in "$host_port" "$api_port"; do
    case "$port" in ''|*[!0-9]*) exit 2 ;; esac
    [ "$port" -ge 1 ] && [ "$port" -le 65535 ] || exit 2
    if nc -z 127.0.0.1 "$port" 2>/dev/null; then
        echo "Port $port is occupied." >&2
        exit 2
    fi
done
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
out=$(mktemp -d "${TMPDIR:-/tmp}/b1-cold.XXXXXX")
echo "Results: $out"
export B1_PROJECT="b1-cold-$(date -u +%Y%m%d%H%M%S)-$$"
export B1_HOST_OLLAMA_PORT="$host_port" HF_HOME="$out/hf"
mkdir "$out/ollama" "$HF_HOME"
git clone -q --local "$repo" "$out/checkout"
cd "$out/checkout"
git checkout -q --detach f89180849615c5a74543b37eb460180380c2e6a4
test -z "$(git status --porcelain)"
test -z "$(docker volume ls -q --filter label=com.docker.compose.project="$B1_PROJECT")"
test -z "$(docker ps -aq --filter label=com.docker.compose.project="$B1_PROJECT")"
git rev-parse HEAD > "$out/commit"
OLLAMA_HOST="127.0.0.1:$host_port" OLLAMA_MODELS="$out/ollama" \
    ollama serve > "$out/daemon.log" 2>&1 &
daemon_pid=$!
trap 'kill "$daemon_pid" 2>/dev/null || true' 0
until curl -fsS "http://127.0.0.1:$host_port/api/tags" > "$out/empty-tags.json" 2>/dev/null; do
    kill -0 "$daemon_pid"
    sleep 1
done
jq -e '.models == []' "$out/empty-tags.json" >/dev/null
api="http://127.0.0.1:$api_port"
date -u +%FT%TZ > "$out/start"
./run --build --quality balanced --strict-profile --port "$api_port" > "$out/first.log" 2>&1
date -u +%FT%TZ > "$out/ready"
curl -fsS "$api/health/build" > "$out/build.json"
curl -fsS "$api/health/ready" > "$out/healthy.json"
docker logs --timestamps "$B1_PROJECT-models-init-1" > "$out/first-models-init.log" 2>&1
B1_DEMO_DIR="$out/demo" sh eval/demo.sh "$api" > "$out/demo.log" 2>&1
./run stop > "$out/stop.log" 2>&1
date -u +%FT%TZ > "$out/warm-start"
./run --offline > "$out/warm.log" 2>&1
date -u +%FT%TZ > "$out/warm-ready"
doc=$(jq -r .doc_id "$out/demo/01_upload_a.json")
curl -fsS "$api/api/v1/documents/$doc" > "$out/retained.json"
curl -fsS "$api/api/v1/documents?limit=100" > "$out/list-after.json"
jq -e '.ready and (.chunk_count > 0)' "$out/retained.json" >/dev/null
jq -e --arg id "$doc" 'any(.items[]; .doc_id == $id and .ready)' "$out/list-after.json" >/dev/null
du -sk "$out/ollama" > "$out/ollama-kib"
docker exec "$B1_PROJECT-worker-1" du -sk /models-cache /home/appuser/.cache > "$out/retrieval-kib"
docker ps -a --filter label=com.docker.compose.project="$B1_PROJECT" \
    --format '{{.ID}}' | while IFS= read -r id; do
    service=$(docker inspect -f '{{index .Config.Labels "com.docker.compose.service"}}' "$id")
    image_id=$(docker inspect -f '{{.Image}}' "$id")
    docker image inspect -f "$service {{.Id}} {{.Size}}" "$image_id"
done > "$out/images-by-service.txt"
docker system df -v --format json | jq --rawfile rows "$out/images-by-service.txt" \
    '[.Images[] | select(.ID as $id | $rows | contains($id))]' > "$out/docker-image-sizes.json"
date -u +%FT%TZ > "$out/verified"
./run stop > "$out/stopped.log" 2>&1
echo "Verified. Responses, timestamps and logs: $out"
