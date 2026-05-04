---
phase: 04
slug: storage-layer-recording
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-05-04
---

# Phase 04 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest (existing in project) |
| **Config file** | `pytest.ini` or `pyproject.toml` — check existing |
| **Quick run command** | `pytest tests/test_history.py -x` |
| **Full suite command** | `pytest tests/ -x` |
| **Estimated runtime** | ~10 seconds |

---

## Sampling Rate

- **After every task commit:** Run `pytest tests/test_history.py -x`
- **After every plan wave:** Run `pytest tests/ -x`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 15 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 04-01-01 | 01 | 1 | STOR-01 | — | N/A | unit | `pytest tests/test_history.py::test_record_created -x` | ❌ W0 | ⬜ pending |
| 04-01-02 | 01 | 1 | STOR-02 | — | N/A | unit | `pytest tests/test_history.py::test_file_location -x` | ❌ W0 | ⬜ pending |
| 04-01-03 | 01 | 1 | STOR-03 | — | N/A | unit | `pytest tests/test_history.py::test_unique_ids -x` | ❌ W0 | ⬜ pending |
| 04-01-04 | 01 | 1 | STOR-04 | — | N/A | unit | `pytest tests/test_history.py::test_record_schema -x` | ❌ W0 | ⬜ pending |
| 04-01-05 | 01 | 1 | STOR-05 | — | N/A | unit | `pytest tests/test_history.py::test_fifo_eviction -x` | ❌ W0 | ⬜ pending |
| 04-02-01 | 02 | 2 | STOR-01 | — | N/A | unit | `pytest tests/test_history.py::test_copy_hooks -x` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `tests/test_history.py` — stubs for STOR-01 through STOR-05
- [ ] `tests/conftest.py` — shared `HistoryRecorder` fixture

*If none: "Existing infrastructure covers all phase requirements."*

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Concurrent copy operations | STOR-01 | Parallel processes race condition | Run two `elm copy` in parallel; verify both appear in history |

*If none: "All phase behaviors have automated verification."*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 15s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
