#!/bin/sh
# Public API acceptance. Requires curl and jq. Each step saves its actual response.
set -eu
API=${1:-http://127.0.0.1:8000}
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT=${B1_DEMO_DIR:-"$ROOT/eval/reports/demo/$TS"}
mkdir -p "$OUT"
DOC_A="$ROOT/test_documents/04_Аренда_производственного_помещения.pdf"
DOC_B="$ROOT/test_documents/15_Лицензия_Litender.pdf"
command -v jq >/dev/null || { echo "eval/demo.sh requires jq" >&2; exit 1; }

get() { curl --fail --silent --show-error --max-time 30 "$1"; }
upload() {
    filename=${1##*/}
    curl --fail --silent --show-error --max-time 60 "$API/api/v1/documents" \
        -F "file=@$1;type=application/pdf" --form-string "title=${filename%.*}" > "$OUT/$2.json"
    jq -er '.doc_id and .job_id and (.status == "accepted")' "$OUT/$2.json" >/dev/null
}
poll() {
    i=0
    while [ "$i" -lt 1200 ]; do
        get "$API/api/v1/jobs/$1" > "$OUT/$2.json"
        case $(jq -er '.status' "$OUT/$2.json") in
            succeeded) return 0 ;;
            failed|cleanup_pending) cat "$OUT/$2.json" >&2; return 1 ;;
        esac
        i=$((i+1))
        sleep 1
    done
    echo "Indexing timed out for job $1" >&2
    return 1
}
ask() {
    jq -n --arg q "$3" --argjson debug "${5:-false}" '{question:$q, debug:$debug}' |
        curl --fail --silent --show-error --max-time "${B1_QA_TEST_TIMEOUT:-960}" \
            "$API/api/v1/documents/$1/questions" -H 'Content-Type: application/json' \
            --data-binary @- > "$OUT/$2.json"
    if ! jq -e --argjson expected "$4" \
        '(.found == $expected) and (.answer | length > 0) and
         (if $expected then (.citations | length > 0) and
            all(.citations[]; (.text | length > 0) and .page >= 1 and has("section"))
          else .citations == [] end)' "$OUT/$2.json" >/dev/null; then
        cat "$OUT/$2.json" >&2
        echo "Unexpected response in $2" >&2
        return 1
    fi
}
get "$API/health/ready" > "$OUT/00_meta.json"
ready=$(cat "$OUT/00_meta.json")
jq -n --arg ts "$TS" --arg api "$API" --arg sha "$(git -C "$ROOT" rev-parse HEAD)" \
    --argjson ready "$ready" '{timestamp_utc:$ts,api:$api,git_head:$sha,ready:$ready}' > "$OUT/00_meta.json"
upload "$DOC_A" 01_upload_a
A=$(jq -er '.doc_id' "$OUT/01_upload_a.json")
JOB=$(jq -er '.job_id' "$OUT/01_upload_a.json")
get "$API/api/v1/jobs/$JOB" > "$OUT/02_poll_a_first.json"
poll "$JOB" 03_poll_a_final
upload "$DOC_B" 04_upload_b
B=$(jq -er '.doc_id' "$OUT/04_upload_b.json")
poll "$(jq -er '.job_id' "$OUT/04_upload_b.json")" 05_poll_b_final
get "$API/api/v1/documents?limit=100" > "$OUT/06_list.json"
EXACT='Какой размер постоянной части арендной платы установлен п. 3.1.1 договора аренды нежилого помещения и включает ли он НДС?'
ask "$A" 07_exact "$EXACT" true
jq -e '(.answer | test("158[[:space:] ]?400")) and (.answer | contains("НДС"))' "$OUT/07_exact.json" >/dev/null
ask "$A" 08_paraphrase 'На какой срок здание передаётся в аренду и с какого момента этот срок отсчитывается?' true
ask "$A" 09_no_answer 'Какой обеспечительный (гарантийный) платёж вносит арендатор по договору аренды нежилого помещения?' false
ask "$A" 10_debug 'Какая неустойка предусмотрена п. 5.1 договора аренды при просрочке уплаты арендных платежей?' true true
jq -e '.diagnostics | .dense_returned > 0 and .sparse_returned > 0 and
   .reranked_count >= .context_count and (.cited_chunk_ids | length) > 0 and
   ((.cited_chunk_ids - .context_chunk_ids) | length == 0)' "$OUT/10_debug.json" >/dev/null
ask "$B" 11_isolation_b_asked_about_a "$EXACT" false
ask "$B" 12_doc_b_own_question 'По какой ставке облагается НДС передача права использования Программы согласно п. 4.3 лицензионного договора и на каком основании?' true
echo "Проверка пройдена. Ответы API сохранены в $OUT"
