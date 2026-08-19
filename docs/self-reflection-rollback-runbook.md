# Self-reflection rollback runbook

Scope: the self-reflective answer critique/revision stage
(`app.rag_services.reflection`), admin-controlled via the RAG Operations
panel (`/admin/rag-ops`). This runbook covers rolling back a bad self-reflection
rollout - not a general incident response procedure.

Rehearsed end-to-end (not just described) in
`tests/api/test_rag_ops_sync.py::test_rollback_drill_emergency_disable_propagates_to_a_second_worker_and_forces_self_reflection_off`,
which proves the full chain - DB row -> cross-worker poller ->
`RagRuntimeConfigStore` -> `DynamicSelfReflectionEngine.plan()` - actually
converges every worker to "disabled" from one admin action, not merely that
individual fields copy correctly in isolation.

## When to roll back

Any of the triggers already listed in the feature's design doc, most
commonly checked via the RAG Operations panel's Self-Reflection metrics
card or a fleet-wide `SelfReflectionTelemetry` exporter if one is wired up
for this deployment:

- Fallback rate > 2% for 15 minutes (`self_reflection_metrics.fallback_rate`)
- p95 total request latency exceeds SLO by 20% since the rollout started
- Mean iterations trending above ~1.6 (`self_reflection_metrics.average_iterations`)
- Token cost per request tracking 20%+ over the pre-rollout baseline
  (`self_reflection_metrics.usage_tokens_total` rate of change)
- Any critical unsupported claim reaching a user despite reflection being on
- The bounded evidence-retrieval step firing more than the configured
  `reflection_max_additional_retrievals` ceiling would suggest (a sign the
  gate in `SelfReflectionPlan.allow_retrieval` isn't being respected -
  should never happen; treat as a code-level bug, not just a rollout tune)

## Step 1 - Immediate stop: emergency disable (seconds, fleet-wide)

This is the fastest lever and affects every RAG Ops-controlled feature at
once (reranking, semantic cache, HyDE, CRAG, self-reflection, SQL routing),
not just self-reflection - use it when you need traffic back to the
baseline pipeline *right now* and can sort out which feature caused it
afterward.

```
POST /admin/rag-ops/emergency-disable
{"reason": "self-reflection fallback rate spike", "confirm": true}
```

- Takes effect immediately on the worker that received the request.
- Every other worker converges within one `RagOpsConfigPoller` interval
  (default 5s - see `app.api.rag_ops_sync`) without a restart.
- Verify convergence: `GET /admin/rag-ops/status` on more than one
  worker/pod if your deployment fronts multiple, and confirm
  `emergency_disabled: true` on each.
- Self-reflection's own effect: `DynamicSelfReflectionEngine.plan()`
  returns `cohort="disabled", bypass_reason="emergency_disabled"` for every
  request regardless of `self_reflective_enabled`/rollout percentage/shadow
  state - no self-reflection call happens at all, treatment or shadow.

To resume everything once the cause is understood and fixed:

```
POST /admin/rag-ops/emergency-enable
{"reason": "root cause fixed, resuming"}
```

This restores every RAG Ops feature to whatever its individual
`*_enabled`/rollout fields were already set to - it does not by itself
re-promote self-reflection if you want it to stay off pending step 2 below.

## Step 2 - Scoped rollback: self-reflection only

Prefer this over emergency-disable when only self-reflection is
misbehaving and the other RAG Ops features (reranking, HyDE, CRAG, SQL
routing) are healthy - it avoids reverting unrelated, working features.

```
PATCH /admin/rag-ops/config
{"self_reflective_enabled": false, "reason": "rolling back self-reflection"}
```

Or step back one stage instead of going fully dark:

- **Pull back to revision-only** (drop the additional-retrieval canary,
  keep critique/revision running):
  `{"self_reflective_retrieval_enabled": false}`
- **Pull back to shadow** (keep observing real traffic, stop serving
  reflected answers):
  `{"self_reflective_shadow_enabled": true}` - note this does *not* also
  set `self_reflective_rollout_percentage` to 0; shadow mode is checked
  before the rollout-percentage sample in `DynamicSelfReflectionEngine.plan`,
  so it takes priority regardless of that value.
- **Reduce rollout percentage** rather than an all-or-nothing toggle:
  `{"self_reflective_rollout_percentage": 5}`

All of the above are partial updates - omitted fields are left unchanged,
and the same cross-worker poller convergence in Step 1 applies here too
(no restart needed, converges within one poll interval fleet-wide).

## Step 3 - Confirm the rollback took effect

1. `GET /admin/rag-ops/status` - confirm `self_reflective_enabled`/
   `self_reflective_rollout_percentage`/`self_reflective_shadow_enabled`/
   `self_reflective_retrieval_enabled` match what you just set.
2. Watch `self_reflection_metrics.sample_count` stop climbing (or
   `self_reflection_shadow_metrics.sample_count` start climbing instead, if
   you stepped back to shadow) over the next few minutes of real traffic.
3. Check `GET /admin/rag-ops/audit` - the change should appear as a
   `config_update` entry with your `reason` text, actor, and a diff of
   exactly the fields that changed (see `_diff` in
   `app.repositories.rag_ops_repository` - only fields that actually
   changed value are recorded, so an unexpectedly large diff is itself a
   signal something else moved too).

## Notes

- Every mutating endpoint above requires the admin role and is audited -
  there is no silent/unaudited rollback path.
- A response served under self-reflection's fallback or abstain outcome is
  never cached (see `RAGService.answer`'s `pipeline_fallback` computation),
  so rolling back does not require a separate cache-invalidation step for
  those specific responses. A rollback also does not retroactively affect
  already-cached *accepted* self-reflection answers already served under
  the old configuration's cache namespace - those simply age out normally,
  or call `POST /admin/cache/clear` (see `AdminController.cache_clear`) if
  you need them gone immediately.
- If the DB row itself is suspected corrupted or stuck (e.g. a constraint
  violation is somehow being bypassed), migrations 010/012's CHECK
  constraints are the final backstop - see
  `tests/repositories/test_rag_ops_self_reflection_integration.py` for what
  they guarantee even against a writer that bypasses the application layer
  entirely.
