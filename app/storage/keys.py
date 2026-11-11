"""Redis keyspace and Lua scripts shared by async (API) and sync (worker/dispatcher) sides.

Keyspace per ADR 003 (verified in spike S5): single logical scope, prefix `rag:`.
"""

from __future__ import annotations

KEY_PREFIX = "rag"


def job_key(job_id: str) -> str:
    return f"{KEY_PREFIX}:job:{job_id}"


def idempotency_key(key: str) -> str:
    return f"{KEY_PREFIX}:idem:{key}"


def doc_key(doc_id: str) -> str:
    return f"{KEY_PREFIX}:doc:{doc_id}"


def outbox_pending_key() -> str:
    return f"{KEY_PREFIX}:outbox:pending"


def outbox_dispatching_key() -> str:
    return f"{KEY_PREFIX}:outbox:dispatching"


def jobs_by_created_key() -> str:
    return f"{KEY_PREFIX}:jobs:by_created"


def lease_key(job_id: str) -> str:
    return f"{KEY_PREFIX}:lease:{job_id}"


# Atomically: check idempotency key, create job hash + idem mapping + outbox intent
# + doc mapping + created-index entry. Returns {verdict, job_id, doc_id}.
# KEYS: 1=idem, 2=job, 3=outbox_pending, 4=jobs_by_created, 5=doc
# ARGV: 1=fingerprint, 2=job_id, 3=doc_id, 4=now_epoch, 5..=job hash field pairs
SUBMIT_LUA = """
local existing_fp = redis.call('HGET', KEYS[1], 'fingerprint')
if existing_fp then
    local existing_job = redis.call('HGET', KEYS[1], 'job_id')
    local existing_doc = redis.call('HGET', KEYS[1], 'doc_id')
    if existing_fp == ARGV[1] then
        return {'DUPLICATE', existing_job, existing_doc}
    end
    return {'CONFLICT', existing_job, existing_doc}
end
redis.call('HSET', KEYS[1], 'fingerprint', ARGV[1], 'job_id', ARGV[2], 'doc_id', ARGV[3])
redis.call('HSET', KEYS[2], unpack(ARGV, 5))
redis.call('RPUSH', KEYS[3], ARGV[2])
redis.call('ZADD', KEYS[4], ARGV[4], ARGV[2])
redis.call('SET', KEYS[5], ARGV[2])
return {'CREATED', ARGV[2], ARGV[3]}
"""

# Lease-gated publication CAS (ADR 003 §6, verified pattern in spike S5).
# KEYS: 1=lease, 2=job. ARGV: 1=token, 2=accepted_run_id, 3=updated_at iso.
# Returns {verdict, detail}.
PUBLISH_LUA = """
local tok = redis.call('GET', KEYS[1])
if tok ~= ARGV[1] then
    return {'STALE_TOKEN', tok or ''}
end
local st = redis.call('HGET', KEYS[2], 'status')
if st ~= 'running' and st ~= 'validating' then
    return {'BAD_STATE', st or ''}
end
redis.call('HSET', KEYS[2], 'status', 'succeeded', 'accepted_run_id', ARGV[2],
           'updated_at', ARGV[3], 'stage', 'publish',
           'error_code', '', 'error_message', '', 'error_retryable', '')
return {'OK', st}
"""

# Lease renewal, owner-only. KEYS: 1=lease. ARGV: 1=token, 2=ttl_ms.
RENEW_LUA = """
local tok = redis.call('GET', KEYS[1])
if tok ~= ARGV[1] then
    return 0
end
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 1
"""

# Guarded state transition. KEYS: 1=job. ARGV: 1=expected statuses csv (''=any),
# 2=new status, 3=updated_at iso, 4..=extra field pairs.
# Returns {verdict, previous_status}.
TRANSITION_LUA = """
local cur = redis.call('HGET', KEYS[1], 'status')
if not cur then
    return {'NOT_FOUND', ''}
end
if ARGV[1] ~= '' then
    local ok = false
    for s in string.gmatch(ARGV[1], '([^,]+)') do
        if s == cur then ok = true end
    end
    if not ok then
        return {'INVALID_STATE', cur}
    end
end
redis.call('HSET', KEYS[1], 'status', ARGV[2], 'updated_at', ARGV[3])
if #ARGV > 3 then
    redis.call('HSET', KEYS[1], unpack(ARGV, 4))
end
return {'OK', cur}
"""
