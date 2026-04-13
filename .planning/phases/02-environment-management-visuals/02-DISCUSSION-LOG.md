# Phase 2: Environment Management Visuals - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-14
**Phase:** 02-Environment Management Visuals
**Areas discussed:** UI Layout Strategy, Sensitive Data Handling, Connection Verification Flow

---

## UI Layout Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Split Pane | List of environments on the left, a details/edit form on the right. | ✓ |
| List + Modals | A full-width list of environments, with separate popup windows. | |
| Accordion/expandable | A list where clicking an environment expands it inline. | |

**User's choice:** Split Pane

---

## Sensitive Data Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Masked with Toggle | Passwords appear as `****` with an "eye" icon button. | |
| Always Masked | Passwords are write-only. If you need to change it, type a new one. | ✓ |
| Clear Text | Show them plainly since this is a local CLI tool. | |

**User's choice:** Always Masked

---

## Connection Verification Flow

| Option | Description | Selected |
|--------|-------------|----------|
| On Demand | User explicitly clicks a "Test Connection" button on the form. | ✓ |
| Automatically on Save | Every time an environment is saved/created, automatically run a test. | |
| Strict Verification | Automatically test, and block saving if the connection fails. | |

**User's choice:** On demand

---

## the agent's Discretion

- Precise widget layout inside the split pane and validation error formatting.

## Deferred Ideas

None
