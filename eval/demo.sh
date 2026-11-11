#!/bin/sh
# Stage-9 API demo (plan §12): upload, poll, list, exact question, paraphrase,
# no-answer, debug, second-document isolation. Saves REAL outputs with a
# timestamp and version metadata (no secrets) to eval/reports/demo/<ts>/.
# Needs only curl + a running stack:  sh eval/demo.sh [API_BASE]
set -eu
API=${1:-http://127.0.0.1:8000}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT="$ROOT/eval/reports/demo/$TS"
mkdir -p "$OUT"
DOC_A="$ROOT/test_documents/04_Аренда_производственного_помещения.pdf"
DOC_B="$ROOT/test_documents/15_Лицензия_Litender.pdf"

save() { printf '%s\n' "$2" > "$OUT/$1.json"; printf '== %s\n%s\n\n' "$1" "$2" | head -c 1500; printf '\n'; }
json_field() { sed -n "s/.*\"$1\": *\"\([^\"]*\)\".*/\1/p" | head -1; }

{
    printf '{"timestamp_utc": "%s", "api": "%s", "git_head": "%s", ' "$TS" "$API" \
        "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
    printf '"ready": %s}\n' "$(curl -s "$API/health/ready")"
} > "$OUT/00_meta.json"
cat "$OUT/00_meta.json"

upload() {  # upload FILE -> prints body (202)
    curl -s -X POST "$API/api/v1/documents" -F "file=@$1;type=application/pdf"
}
wait_ready() {  # wait_ready JOB_ID -> final job JSON
    i=0
    while :; do
        body=$(curl -s "$API/api/v1/jobs/$1")
        case "$body" in *'"status":"succeeded"'*|*'"status": "succeeded"'*|*'"status":"failed"'*|*'"status": "failed"'*) printf '%s' "$body"; return 0 ;; esac
        i=$((i + 1)); [ $i -gt 1200 ] && { printf '%s' "$body"; return 1; }
        sleep 1
    done
}
ask() {  # ask DOC_ID QUESTION [debug]
    dbg=${3:-false}
    q=$(printf '%s' "$2" | sed 's/\\/\\\\/g; s/"/\\"/g')
    curl -s -X POST "$API/api/v1/documents/$1/questions" -H 'Content-Type: application/json' \
        -d "{\"question\": \"$q\", \"debug\": $dbg}"
}

A=$(upload "$DOC_A"); save 01_upload_a "$A"
JOB_A=$(printf '%s' "$A" | json_field job_id); DOC_ID_A=$(printf '%s' "$A" | json_field doc_id)
save 02_poll_a_first "$(curl -s "$API/api/v1/jobs/$JOB_A")"
save 03_poll_a_final "$(wait_ready "$JOB_A")"
B=$(upload "$DOC_B"); save 04_upload_b "$B"
JOB_B=$(printf '%s' "$B" | json_field job_id); DOC_ID_B=$(printf '%s' "$B" | json_field doc_id)
save 05_poll_b_final "$(wait_ready "$JOB_B")"
save 06_list "$(curl -s "$API/api/v1/documents?limit=5")"
save 07_exact "$(ask "$DOC_ID_A" 'Какой размер постоянной части арендной платы установлен п. 3.1.1 договора аренды нежилого помещения и включает ли он НДС?')"
save 08_paraphrase "$(ask "$DOC_ID_A" 'На какой срок здание передаётся в аренду и с какого момента этот срок отсчитывается?')"
save 09_no_answer "$(ask "$DOC_ID_A" 'Какой обеспечительный (гарантийный) платёж вносит арендатор по договору аренды нежилого помещения?')"
save 10_debug "$(ask "$DOC_ID_A" 'Какая неустойка предусмотрена п. 5.1 договора аренды при просрочке уплаты арендных платежей?' true)"
save 11_isolation_b_asked_about_a "$(ask "$DOC_ID_B" 'Какой размер постоянной части арендной платы установлен п. 3.1.1 договора аренды нежилого помещения и включает ли он НДС?')"
save 12_doc_b_own_question "$(ask "$DOC_ID_B" 'По какой ставке облагается НДС передача права использования Программы согласно п. 4.3 лицензионного договора и на каком основании?')"
echo "saved to $OUT"
