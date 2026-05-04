# Phase 02: Environment Management Visuals — Research

**Researched:** 2026-04-14
**Status:** Complete

## 1. Codebase Architecture Analysis

### API Layer (`elm/api.py`)

The API module provides the following environment functions that the GUI must connect to:

| Function | Signature | Returns | Notes |
|----------|-----------|---------|-------|
| `list_environments(show_all)` | `show_all: bool = False` | `List[Dict]` | Each dict has `name` key; with `show_all=True` includes `host`, `port`, `user`, `password` (masked as `********`), `service`, `type`, `is_encrypted` |
| `get_environment(name, encryption_key)` | `name: str, encryption_key: Optional[str]` | `Optional[Dict]` | Returns full env dict or `None` |
| `create_environment(...)` | `name, host, port, user, password, service, db_type, encrypt, encryption_key, overwrite, connection_type` | `bool` | Returns `True` on success |
| `update_environment(...)` | `name, host, port, user, password, service, db_type, encrypt, encryption_key` | `bool` | Partial updates supported (only non-`None` fields changed) |
| `delete_environment(name)` | `name: str` | `bool` | Returns `True` on success |
| `test_environment(name, encryption_key)` | `name: str, encryption_key: Optional[str]` | `Dict` | Returns `OperationResult.to_dict()` with `success`, `message` keys |

### Core Layer (`elm/core/environment.py`)

Returns `OperationResult` objects (not raw booleans). The `api.py` layer unwraps these. For the GUI we can use **either** layer, but `api.py` is cleaner since it returns simple Python types.

**Key data model fields per environment:**
- `name` (section header in INI)
- `host`, `port`, `user`, `password`, `service`
- `type` (enum: `ORACLE`, `POSTGRES`, `MYSQL`, `MSSQL`)
- `is_encrypted` (`True`/`False` string)
- `connection_type` (Oracle-only: `service_name` | `sid`)

### Storage Format

Environments are stored in `environments.ini` (INI/ConfigParser format) at the path defined by `elm.elm_utils.variables.ENVS_FILE`. File locking is handled by `elm/core/utils.py` via `file_lock`.

### Existing GUI (`elm/gui/app.py`)

- `ELMApp(ctk.CTk)` — main window, 800×600, centered, dark-blue theme
- `CTkTabview` with tabs: `"🌐  Environments"` and `"⚙  Operations"`
- Default tab: `"🌐  Environments"`
- Currently contains placeholder `CTkLabel` in each tab
- Integration point: replace placeholder in `self.tabview.tab("🌐  Environments")` with the environment manager widget

## 2. CustomTkinter Widget Patterns

### Split Pane Layout

CustomTkinter **does not** have a native `PanedWindow` widget. Two approaches:

**Approach A: Fixed-ratio `CTkFrame` side-by-side (Recommended)**
- Use two `CTkFrame` instances inside a container using `pack(side="left")` and `pack(side="right", fill="both", expand=True)`
- Left panel gets a fixed `width` value; right panel expands
- Simpler, consistent with CTk theming, no theme mismatch

**Approach B: `tk.PanedWindow` + `CTkFrame` children**
- Resizable sash, but visually inconsistent with CTk dark theme
- Requires manual background/sash color tuning

**Decision: Approach A** — fixed-width left panel (`~220px`) with right panel expanding. Matches CONTEXT.md D-01 requirement without introducing theme styling complexity.

### Scrollable Environment List

CustomTkinter has no built-in listbox. Best pattern:

**`CTkScrollableFrame` with `CTkButton` items:**
- Each environment rendered as a flat `CTkButton(fg_color="transparent", anchor="w")`
- On click: highlight selected button, populate form on the right
- Selection tracked via instance variable (`self.selected_env`)
- Supports dynamic add/remove of items

This pattern avoids third-party dependencies (no `CTkListbox` pip package needed).

### Form Fields

| Field | Widget | Notes |
|-------|--------|-------|
| Name | `CTkEntry(placeholder_text="Environment Name")` | Disabled during edit mode |
| Host | `CTkEntry(placeholder_text="hostname or IP")` | |
| Port | `CTkEntry(placeholder_text="1521")` | Numeric validation on submit |
| User | `CTkEntry(placeholder_text="username")` | |
| Password | `CTkEntry(show="*")` | Write-only per D-02: never populated from existing env |
| Service | `CTkEntry(placeholder_text="service name or SID")` | |
| DB Type | `CTkOptionMenu(values=["ORACLE","POSTGRES","MYSQL","MSSQL"])` | Dropdown |
| Connection Type | `CTkOptionMenu(values=["service_name","sid"])` | Only shown when DB Type = ORACLE |
| Encrypt | `CTkCheckBox(text="Encrypt")` | Toggles encryption key field visibility |
| Encryption Key | `CTkEntry(show="*")` | Only visible when Encrypt is checked |

### Password Handling (D-02: Always Masked)

Per context decision D-02: passwords are **write-only**. When loading an existing environment for editing:
- All fields are populated **except** password
- Password entry shows placeholder text: `"Enter new password to change"`
- On save/update: if password field is empty, pass `None` to `update_environment()` (field is skipped)
- On create: password is required (validation enforced)

### Connection Test (D-03: On Demand)

A `CTkButton(text="Test Connection")` in the form area calls `api.test_environment(name, encryption_key)`.
- On success: show green status label `"✓ Connected"` 
- On failure: show red status label `"✗ Connection failed: {message}"`
- Button disabled while test is in progress (optional: run in thread to avoid UI freeze — but since test_environment is typically fast, can be synchronous for this phase)

## 3. Form Validation Strategy

**Batch validation on submit** (not real-time per-field):

Required fields for **create**: `name`, `host`, `port`, `user`, `password`, `service`, `db_type`
Required fields for **update**: only the `name` must exist; all other fields are optional

Validation rules:
1. `name` — non-empty, not `"*"`
2. `port` — must be a valid integer
3. `password` — required for create, optional for update
4. If `encrypt` checked → `encryption_key` must be non-empty
5. If `db_type` is `ORACLE` → `connection_type` must be set

Error display: use a `CTkLabel(text_color="red")` below the form buttons.

## 4. Architectural Decisions

### Module Organization

Create a new file `elm/gui/environment_manager.py` containing:
- `EnvironmentManagerFrame(ctk.CTkFrame)` — the main split-pane widget
  - Contains `EnvironmentListPanel(ctk.CTkFrame)` — left side
  - Contains `EnvironmentFormPanel(ctk.CTkFrame)` — right side

This keeps `app.py` clean — it just instantiates `EnvironmentManagerFrame` inside the tab.

### Data Flow

```
User action → EnvironmentFormPanel
  → calls elm.api.* functions
  → on success: refreshes EnvironmentListPanel
  → on error: shows error label in form
```

### State Management

- `self.environments: List[Dict]` — cached from `api.list_environments(show_all=True)`
- `self.selected_env: Optional[str]` — name of currently selected environment
- `self.mode: str` — `"idle"` | `"create"` | `"edit"` (controls form state)

### Form Mode Transitions

| Action | From | To | Form State |
|--------|------|----|------------|
| Click "New" | idle | create | Clear all fields, enable name field |
| Click list item | idle/create | edit | Populate fields (except password), disable name field |
| Click "Save" (success) | create/edit | idle | Clear form, refresh list |
| Click "Cancel" | create/edit | idle | Clear form |
| Click "Delete" | edit | idle | Confirm dialog, delete, refresh list, clear form |

### Delete Confirmation

Use `CTkInputDialog` or a simple `messagebox.askyesno` (standard tkinter, works fine with CTk) for delete confirmation.

## 5. Integration with `app.py`

Current placeholder in `_build_tabs()`:
```python
env_label = ctk.CTkLabel(
    self.tabview.tab("🌐  Environments"),
    text="Environment management will appear here.",
    font=("", 14),
)
env_label.pack(expand=True)
```

Replace with:
```python
from elm.gui.environment_manager import EnvironmentManagerFrame

self.env_manager = EnvironmentManagerFrame(self.tabview.tab("🌐  Environments"))
self.env_manager.pack(fill="both", expand=True)
```

## 6. Risk Assessment

| Risk | Mitigation |
|------|------------|
| CTkScrollableFrame performance with many environments | Unlikely — typical user has <50 environments; CTkButton items are lightweight |
| Thread safety for connection test | Phase 2 can use synchronous calls; threading is Phase 3 scope |
| Oracle `connection_type` conditional visibility | Show/hide via `pack_forget()` / `pack()` based on DB Type selection callback |
| Encrypted environment editing | Require encryption key entry; pass to `get_environment()` for field population |
| INI file concurrent access | Already handled by core `file_lock` utility |

## RESEARCH COMPLETE
