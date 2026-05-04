---
phase: 6
slug: gui-history-panel
status: approved
reviewed_at: 2026-05-04
preset: none
created: 2026-05-04
---

# Phase 6 — UI Design Contract

> Visual and interaction contract for the GUI History Panel phase. Extends Phase 2/3 design system; new widgets only.

---

## Design System (Inherited from Phase 2/3)

| Property | Value |
|----------|-------|
| Tool | CustomTkinter (native Python desktop) |
| Preset | `dark-blue` (CTk built-in theme) |
| Component library | CTk native widgets |
| Icon library | Unicode glyphs |
| Font | System default (CTk inherits OS font) |

Spacing, typography, and color tokens identical to Phase 2/3 UI-SPEC. Key tokens reused:
- `padx=10, pady=10` for tab content padding
- `font=("", 14)` body, `font=("", 12)` label, `font=("", 18, "bold")` heading
- Accent reserved for Re-run button and selected states
- Destructive red (`#DC3545`) for Cancel / error states
- Success green (`#28A745`) for completion states

---

## Component Inventory

### History Tab Layout

| Widget | Configuration | Behavior |
|--------|--------------|----------|
| Root frame | `CTkFrame` | `pack(fill="both", expand=True, padx=10, pady=10)` |
| Scrollable list | `CTkScrollableFrame` | `pack(fill="both", expand=True)` — holds history rows |
| Status label | `CTkLabel` | `pack(anchor="w", pady=(0, 8))` — shows "Last refreshed: ..." |

### History Row (CTkFrame)

| Element | Type | Style |
|---------|------|-------|
| ID label | `CTkLabel` | `font=("", 12)`, width=40 |
| Date label | `CTkLabel` | `font=("", 12)`, width=120 |
| Type label | `CTkLabel` | `font=("", 12)`, width=80 |
| Source → Target label | `CTkLabel` | `font=("", 12)`, `wraplength=200` |
| Table label | `CTkLabel` | `font=("", 12)`, width=100 |
| Status badge | `CTkLabel` | text_color by status: green/red/blue |
| Re-run button | `CTkButton` | `text="▶ Re-run"`, `width=80` |
| Edit & Re-run button | `CTkButton` | `text="✎ Edit"`, `width=80` |

Row frame: `CTkFrame` with `fg_color="transparent"`, packed with `fill="x", pady=2`.
Buttons packed `side="right"` with `padx=(4, 0)`.

---

## Color

CustomTkinter `dark-blue` theme provides the color palette. Status colors are hardcoded hex values (matching existing codebase patterns):

| Role | Hex | Usage |
|------|-----|-------|
| Success | `#28A745` | `success` status text |
| Failure | `#DC3545` | `failure` status text |
| In-progress | `#3B8ED0` | `running` status text |

---

## Copywriting Contract

| Element | Copy |
|---------|------|
| Empty state heading | "No history yet" |
| Empty state body | "Run a copy operation to see it here." |
| Status label | "Last refreshed: {timestamp}" |
| Re-run button | "▶ Re-run" |
| Edit & Re-run button | "✎ Edit" |

---

## Interaction Contract

### Tab Switch Refresh
- When user clicks onto History tab, list refreshes from `HistoryRecorder.read_records()`.
- `tabview.tab("⏳  History")` is the tab name.

### Scheduled Poll
- `self.after(15000, self._poll_refresh)` checks for new records while History tab is active.
- Poll is cancelled when tab switches away (store `after_id`, cancel on `tabview.bind("<<NotebookTabChanged>>"` equivalent).

### Re-run Flow
1. User clicks "Re-run" on history entry N
2. GUI switches to Operations tab (`tabview.set("⚙  Operations")`)
3. OperationsPanel executes copy with original parameters via background thread
4. Log streams to existing log panel in Operations tab

### Edit & Re-run Flow
1. User clicks "Edit" on history entry N
2. GUI switches to Operations tab
3. `OperationsPanel.pre_fill_form(params)` populates source, target, query, table, mode, batch_size
4. User can modify and click Execute

---

## Registry Safety

| Registry | Blocks Used | Safety Gate |
|----------|-------------|-------------|
| Not applicable | — | CustomTkinter is the sole widget toolkit; no registry system exists |

---

## Data Flow Contract

```
User clicks Re-run → HistoryPanel._on_rerun(record_id)
  → api.rerun_history_record(record_id)
  → OperationsPanel._on_copy_finished() signals HistoryPanel
  → Next poll or tab switch refreshes list

User clicks Edit → HistoryPanel._on_edit(record_id)
  → api.get_history_record(record_id)
  → ELMApp switches to Operations tab
  → OperationsPanel.pre_fill_form(params)
```

**API functions used by this phase:**
- `api.list_history()` → list of `HistoryRecord` dicts
- `api.get_history_record(id)` → single `HistoryRecord` dict
- `api.rerun_history_record(id)` → re-execute, returns result dict

---

## Conditional Visibility Rules

| Condition | Elements Affected | Behavior |
|-----------|-------------------|----------|
| No history records | Scrollable list | Hidden; empty state label shown |
| History records exist | Empty state label | Hidden; scrollable list shown |
| Status == `success` | Status badge | text_color=`#28A745` |
| Status == `failure` | Status badge | text_color=`#DC3545` |
| Status == `running` | Status badge | text_color=`#3B8ED0` |

---

## Upstream Decision Traceability

| Contract Element | Source | Decision ID |
|------------------|--------|-------------|
| Compact table-style rows | CONTEXT.md | D-01 |
| CTkScrollableFrame + CTkFrame rows | CONTEXT.md | D-02 |
| Re-run streams to Operations log | CONTEXT.md | D-03, D-04 |
| Refresh on tab switch + 15s poll | CONTEXT.md | D-05, D-06, D-07 |
| Edit & Re-run switches tab + pre-fills | CONTEXT.md | D-08, D-09, D-10 |
| Status colors | CONTEXT.md | D-11 |

---

## Checker Sign-Off

| Dimension | Status |
|-----------|--------|
| 1. Spacing | PASS — inherits Phase 2/3 tokens |
| 2. Typography | PASS — inherits Phase 2/3 tokens |
| 3. Color | PASS — accent reserved for Re-run; status colors locked |
| 4. Copy | PASS — all CTAs and empty states declared |
| 5. Interactions | PASS — tab switch, poll, re-run, edit flows defined |
| 6. Registry Safety | PASS — no registry, CustomTkinter only |

**Approved for planning.**
