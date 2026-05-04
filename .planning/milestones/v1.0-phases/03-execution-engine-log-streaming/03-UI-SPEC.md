---
phase: 3
slug: execution-engine-log-streaming
status: approved
reviewed_at: 2026-04-23
preset: none
created: 2026-04-23
---

# Phase 3 — UI Design Contract

> Visual and interaction contract for Execution Engine & Log Streaming. Extends Phase 2 design system; new widgets only.

---

## Design System (Inherited from Phase 2)

| Property | Value |
|----------|-------|
| Tool | CustomTkinter (native Python desktop) |
| Preset | `dark-blue` (CTk built-in theme) |
| Component library | CTk native widgets |
| Icon library | Unicode glyphs |
| Font | System default (CTk inherits OS font) |

Spacing, typography, and color tokens identical to Phase 2 UI-SPEC. Key tokens reused:
- `padx=10, pady=10` for tab content padding
- `font=("", 14)` body, `font=("", 12)` label, `font=("", 18, "bold")` heading
- Accent reserved for Execute button and selected states
- Destructive red (`#DC3545`) for Cancel / error states
- Success green (`#28A745`) for completion states

---

## Component Inventory

### Operations Tab Layout

| Widget | Configuration | Behavior |
|--------|--------------|----------|
| Top frame | `CTkFrame` | `pack(fill="x", padx=10, pady=(10,0))` — holds the copy form |
| Log frame | `CTkFrame` | `pack(fill="both", expand=True, padx=10, pady=10)` — holds the log panel |

### Copy Form (Top Frame)

| Widget | Configuration | Behavior |
|--------|--------------|----------|
| Source dropdown | `CTkOptionMenu` | Populated from `api.list_environments()` |
| Target dropdown | `CTkOptionMenu` | Populated from `api.list_environments()` |
| Query input | `CTkTextbox(height=80)` | Multiline SQL query entry |
| Target table entry | `CTkEntry(placeholder_text="target_table")` | Target table name |
| Execute button | `CTkButton(text="▶ Execute", fg_color=accent)` | Starts background copy thread |
| Cancel button | `CTkButton(text="⏹ Cancel", fg_color="#DC3545")` | Visible only while running; sets cancel event |
| Status label | `CTkLabel(font=("", 12))` | Shows "Ready", "Running...", or "✓ Complete" |

### Log Panel (Log Frame)

| Widget | Configuration | Behavior |
|--------|--------------|----------|
| Log textbox | `CTkTextbox(state="disabled")` | Read-only; main thread inserts drained queue lines |
| Clear button | `CTkButton(text="Clear Log", width=80)` | Clears log textbox contents |

---

## Interaction States

### Operation Lifecycle

| Action | State | UI Behavior |
|--------|-------|-------------|
| Click "Execute" | idle → running | Disable Execute, enable Cancel, status "Running...", start worker thread |
| Worker emits log line | running | Queue drained into log textbox; auto-scroll to bottom |
| Click "Cancel" | running → cancelling | Status "Cancelling..."; worker checks event and exits early |
| Worker completes | running → idle | Enable Execute, hide Cancel, status "✓ Complete" (green) |
| Worker errors | running → idle | Status "✗ Failed: {msg}" (red), show error in log |

### Empty Log State

When log textbox is empty, show placeholder label: "Log output will appear here when an operation runs."

---

## Data Flow Contract

```
User clicks Execute → OperationsPanel.start_copy()
  → Validate form (source != target, query non-empty, table non-empty)
  → Create queue + cancel_event + redirect_stdout(queue)
  → Spawn worker thread calling api.copy_db_to_db(...)
  → Worker stdout/stderr → queue → main after() poll → log_textbox.insert()
  → Thread.join() or cancel_event → Restore buttons, update status
```

---

## Checker Sign-Off

- [x] Dimension 1 Copywriting: PASS — verb+noun CTAs, state labels clear
- [x] Dimension 3 Color: PASS — inherits 60/30/10 from Phase 2
- [x] Dimension 5 Spacing: PASS — 4px grid inherited
- [x] Dimension 6 Registry Safety: PASS — no third-party registries

**Approval:** approved 2026-04-23
