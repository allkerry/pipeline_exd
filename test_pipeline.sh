#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

COLLECTOR_URL="${COLLECTOR_URL:-http://localhost:9000}"
PASS=0
FAIL=0
RESULTS=()

log_pass() { RESULTS+=("✅ $1"); PASS=$((PASS+1)); }
log_fail() { RESULTS+=("❌ $1: $2"); FAIL=$((FAIL+1)); }

now_iso() { date -u +%Y-%m-%dT%H:%M:%S.000Z; }

post_store_item() {
    local payload="$1"
    curl -s -X POST "$COLLECTOR_URL/store_item" -H "Content-Type: application/json" -d "$payload"
}

assert_field() {
    local name="$1" response="$2" expected="$3"
    if echo "$response" | grep -q "\"message\":\"$expected\"" || echo "$response" | grep -q "\"message\": \"$expected\""; then
        log_pass "$name (message=$expected)"
    else
        log_fail "$name" "ожидалось message=$expected, получено: $response"
    fi
}

# ─── 1. Здоровье всех сервисов ─────────────────────────────────
test_health() {
    local services=("collector:9000" "upipe:5981" "bpipe:7995" "transactioneer:8002")
    for svc in "${services[@]}"; do
        name="${svc%%:*}"; port="${svc##*:}"
        if [ "$name" = "collector" ]; then
            code=$(curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health")
        else
            code=$(docker compose exec -T "$name" curl -s -o /dev/null -w "%{http_code}" "http://localhost:$port/health" 2>/dev/null)
        fi
        if [ "$code" = "200" ]; then
            log_pass "health $name"
        else
            log_fail "health $name" "http_code=$code"
        fi
    done

    vpn_status=$(docker compose ps vpn --format json 2>/dev/null | grep -o '"Health":"[a-z]*"' | cut -d'"' -f4)
    if [ "$vpn_status" = "healthy" ]; then
        log_pass "health vpn"
    else
        log_fail "health vpn" "status=$vpn_status"
    fi
}

# ─── 2. VPN реально проксирует ──────────────────────────────────
test_vpn_egress() {
    ip=$(curl -s -x socks5h://127.0.0.1:1080 --max-time 10 https://api.ipify.org)
    if [ -n "$ip" ]; then
        log_pass "vpn egress (ip=$ip)"
    else
        log_fail "vpn egress" "curl через socks5 не вернул IP"
    fi
}

# ─── 3. Нормальный текст проходит весь пайплайн (4.4) ──────────
test_normal_text() {
    local eid="test-normal-$(date +%s)"
    local resp
    resp=$(post_store_item "{
        \"content\": \"This is a completely normal test tweet about the weather being sunny today in California.\",
        \"external_id\": \"$eid\",
        \"created_at\": \"$(now_iso)\",
        \"domain\": \"twitter.com\",
        \"url\": \"https://twitter.com/test/status/1\"
    }")
    assert_field "normal_text store_item" "$resp" "OK"

    sleep 6
    if docker compose logs bpipe --since 30s 2>/dev/null | grep -q "Батч отправлен\|processed"; then
        log_pass "normal_text bpipe processed"
    else
        log_fail "normal_text bpipe processed" "нет строки обработки в логах bpipe за 30с"
    fi
}

# ─── 4. Слишком короткий текст (4.5) ───────────────────────────
test_short_text() {
    local resp
    resp=$(post_store_item "{\"content\": \"f\", \"external_id\": \"test-short-$(date +%s)\", \"created_at\": \"$(now_iso)\", \"domain\": \"twitter.com\", \"url\": \"https://twitter.com/test/status/2\"}")
    assert_field "short_text" "$resp" "skipped_short"
}

# ─── 5. Длинный текст не должен ронять батч (4.6) ──────────────
test_long_text() {
    local long_text
    long_text=$(python3 -c "print(('This is a very long repeated sentence about markets and finance and technology. ' * 400))")
    local eid="test-long-$(date +%s)"
    local resp
    resp=$(post_store_item "$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1], 'external_id': sys.argv[2], 'created_at': sys.argv[3], 'domain': 'twitter.com', 'url': 'https://twitter.com/test/status/3'}))" "$long_text" "$eid" "$(now_iso)")")
    assert_field "long_text store_item" "$resp" "OK"

    sleep 6
    if docker compose logs upipe --since 30s 2>/dev/null | grep -q "✂️"; then
        log_pass "long_text token truncation logged"
    else
        log_fail "long_text token truncation logged" "нет строки ✂️ в логах upipe"
    fi

    if docker compose logs bpipe --since 30s 2>/dev/null | grep -iq "CUDA OOM\|RuntimeError\|IndexError"; then
        log_fail "long_text bpipe no crash" "найдены ошибки в логах bpipe"
    else
        log_pass "long_text bpipe no crash"
    fi
}

# ─── 6. Мусорный текст после перевода (4.7) ────────────────────
test_junk_text() {
    local eid="test-junk-$(date +%s)"
    local resp
    resp=$(post_store_item "{\"content\": \"!!!!!!!!!! ################ 1234567890\", \"external_id\": \"$eid\", \"created_at\": \"$(now_iso)\", \"domain\": \"twitter.com\", \"url\": \"https://twitter.com/test/status/4\"}")
    assert_field "junk_text store_item" "$resp" "OK"

    sleep 3
    if docker compose logs upipe --since 20s 2>/dev/null | grep -q "No content to work with"; then
        log_pass "junk_text rejected in upipe"
    else
        log_fail "junk_text rejected in upipe" "нет ожидаемой ошибки в логах upipe"
    fi
}

# ─── 7. Дубликат по external_id ────────────────────────────────
test_duplicate() {
    local eid="test-dup-$(date +%s)"
    local payload="{\"content\": \"Duplicate detection test content that is long enough to pass filters.\", \"external_id\": \"$eid\", \"created_at\": \"$(now_iso)\", \"domain\": \"twitter.com\", \"url\": \"https://twitter.com/test/status/5\"}"
    post_store_item "$payload" > /dev/null
    local resp
    resp=$(post_store_item "$payload")
    assert_field "duplicate" "$resp" "duplicate"
}

# ─── 8. Устаревший текст (старше MAX_OLDNESS_SECONDS) ──────────
test_old_text() {
    local old_date
    old_date=$(date -u -d "@$(( $(date +%s) - 100000 ))" +%Y-%m-%dT%H:%M:%S.000Z 2>/dev/null || date -u -v-100000S +%Y-%m-%dT%H:%M:%S.000Z)
    local resp
    resp=$(post_store_item "{\"content\": \"This content is definitely old enough to be filtered by oldness check here.\", \"external_id\": \"test-old-$(date +%s)\", \"created_at\": \"$old_date\", \"domain\": \"twitter.com\", \"url\": \"https://twitter.com/test/status/6\"}")
    assert_field "old_text" "$resp" "filtered_old"
}

# ─── 9. Не-английский текст фильтруется ────────────────────────
test_non_english() {
    local resp
    resp=$(post_store_item "{\"content\": \"Это тестовый текст на русском языке, который должен быть отфильтрован языковым фильтром.\", \"external_id\": \"test-lang-$(date +%s)\", \"created_at\": \"$(now_iso)\", \"domain\": \"twitter.com\", \"url\": \"https://twitter.com/test/status/7\"}")
    assert_field "non_english" "$resp" "filtered_lang"
}

# ─── 10. Битый JSON ─────────────────────────────────────────────
test_bad_json() {
    local code
    code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$COLLECTOR_URL/store_item" -H "Content-Type: application/json" -d '{not valid json')
    if [ "$code" = "400" ]; then
        log_pass "bad_json returns 400"
    else
        log_fail "bad_json returns 400" "получен код $code"
    fi
}

# ─── 11. Текст длиннее MAX_TEXT_LEN обрезается, не дропается ──
test_over_max_len() {
    local huge_text
    huge_text=$(python3 -c "print('A' * 25000)")
    local eid="test-overmax-$(date +%s)"
    local resp
    resp=$(post_store_item "$(python3 -c "import json,sys; print(json.dumps({'content': sys.argv[1], 'external_id': sys.argv[2], 'created_at': sys.argv[3], 'domain': 'twitter.com', 'url': 'https://twitter.com/test/status/8'}))" "$huge_text" "$eid" "$(now_iso)")")
    assert_field "over_max_len store_item" "$resp" "OK"
}

# ─── 12. GPU память под нагрузкой не переполняется ─────────────
test_gpu_memory() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        log_fail "gpu_memory" "nvidia-smi недоступен на хосте"
        return
    fi
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -lt 8192 ]; then
        log_pass "gpu_memory (${used}MiB / 8192MiB)"
    else
        log_fail "gpu_memory" "используется ${used}MiB, упирается в лимит 8192MiB"
    fi
}

echo "▶ Запуск полного набора тестов пайплайна..."
echo

test_health
test_vpn_egress
test_bad_json
test_short_text
test_over_max_len
test_non_english
test_old_text
test_duplicate
test_junk_text
test_long_text
test_normal_text
test_gpu_memory

echo
echo "─────────────────────────────────────────"
for r in "${RESULTS[@]}"; do echo "$r"; done
echo "─────────────────────────────────────────"
echo "Пройдено: $PASS | Провалено: $FAIL"

[ "$FAIL" -eq 0 ]
