#!/bin/sh
# Host probe (stage 8.2): emits resolver facts as JSON. POSIX sh, stdlib tools
# only. Pure fact collection — NO decisions here (the resolver owns those).
set -eu

os_name=$(uname -s | tr '[:upper:]' '[:lower:]')
arch=$(uname -m)

docker_ok=false
docker_vm_mem=0
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    docker_ok=true
    docker_vm_mem=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)
fi

compose_ok=false
if docker compose version >/dev/null 2>&1; then
    compose_ok=true
fi

host_free_bytes=0
host_total_bytes=0
if [ "$os_name" = "darwin" ]; then
    host_total_bytes=$(sysctl -n hw.memsize)
    page_size=$(vm_stat | sed -n 's/.*page size of \([0-9]*\) bytes.*/\1/p')
    free_pages=$(vm_stat | awk '
        /Pages free/            {gsub("\\.",""); f=$3}
        /Pages inactive/        {gsub("\\.",""); i=$3}
        /Pages speculative/     {gsub("\\.",""); s=$3}
        END {print f+i+s}')
    host_free_bytes=$((free_pages * page_size))
elif [ -r /proc/meminfo ]; then
    host_total_bytes=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) * 1024 ))
    host_free_bytes=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) * 1024 ))
fi

ollama_reachable=false
if curl -s --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    ollama_reachable=true
fi

dmr_running=false
if docker model status >/dev/null 2>&1; then
    dmr_running=true
fi

nvidia_present=false
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia_present=true
fi
# A positive container GPU smoke is recorded ONLY by an actual smoke run
# (scripts/gpu_smoke.sh, launcher --device nvidia path); never inferred here.
nvidia_smoke=false
if [ "${B1_NVIDIA_SMOKE_OK:-}" = "1" ]; then
    nvidia_smoke=true
fi

free_disk_gb=$(df -g . 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)
[ -n "$free_disk_gb" ] || free_disk_gb=0

cat <<EOF
{
  "os": "$os_name",
  "arch": "$arch",
  "docker_ok": $docker_ok,
  "compose_ok": $compose_ok,
  "docker_vm_mem_gib": $(awk "BEGIN {printf \"%.2f\", $docker_vm_mem/1073741824}"),
  "host_total_ram_gib": $(awk "BEGIN {printf \"%.2f\", $host_total_bytes/1073741824}"),
  "host_free_ram_gib": $(awk "BEGIN {printf \"%.2f\", $host_free_bytes/1073741824}"),
  "host_ollama_reachable": $ollama_reachable,
  "dmr_running": $dmr_running,
  "dmr_gate_passed": false,
  "nvidia_smi_present": $nvidia_present,
  "nvidia_container_smoke_ok": $nvidia_smoke,
  "free_disk_gb": $free_disk_gb
}
EOF
