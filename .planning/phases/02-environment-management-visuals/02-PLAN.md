---
phase: 2
plan: 1
type: feature
wave: 1
depends_on: [01-foundation-gui-bootstrap]
files_modified:
  - elm/gui/environment_manager.py
  - elm/gui/app.py
autonomous: true
requirements: [ENV-01, ENV-02, ENV-03, ENV-04]
---

<objective>
Build the Environment Management GUI inside the "🌐  Environments" tab. Replace the placeholder label with a split-pane widget: a scrollable list of environments on the left (220px fixed width) and a full form on the right for creating, editing, and deleting environments. Connect all form actions to `elm.api` CRUD functions. Implement write-only password handling, on-demand connection testing, batch validation on submit, and a three-mode state machine (idle / create / edit).
</objective>

<tasks>

### Task 1: Create elm/gui/environment_manager.py

<type>create</type>
<files>elm/gui/environment_manager.py</files>

<read_first>
- elm/gui/app.py (understand CTkTabview tab structure and how to pack a widget into the Environments tab)
- elm/api.py (list_environments, get_environment, create_environment, update_environment, delete_environment, test_environment signatures)
- elm/core/environment.py (understand OperationResult, config storage via INI, encryption behavior)
- .planning/phases/02-environment-management-visuals/02-UI-SPEC.md (full design contract: spacing, typography, colors, component inventory, interaction states, copywriting, validation rules, conditional visibility)
- .planning/phases/02-environment-management-visuals/02-CONTEXT.md (D-01 split pane, D-02 write-only password, D-03 on-demand test)
</read_first>

<action>
Create `elm/gui/environment_manager.py` containing three nested CustomTkinter widget classes. The file must be importable without launching a GUI (no `mainloop` or window creation at module level).

**Class 1: `EnvironmentManagerFrame(ctk.CTkFrame)`**
- `__init__(self, master)` creates a horizontal split pane:
  - Left `CTkFrame(width=220, fg_color="transparent")` packed `side="left", fill="y"`
  - Right `CTkFrame(fg_color="transparent")` packed `side="right", fill="both", expand=True`
- Instantiate `EnvironmentListPanel` in the left frame and `EnvironmentFormPanel` in the right frame.
- Expose `refresh_list()` that delegates to the list panel.

**Class 2: `EnvironmentListPanel(ctk.CTkFrame)`**
- `__init__(self, master, on_select_callback)`
- Top header bar: `CTkFrame` with a right-aligned `CTkButton(text="＋ New", width=80, command=self._on_new)`
- Below header: `CTkScrollableFrame` packed `fill="both", expand=True`
- `refresh()` clears scrollable frame, calls `api.list_environments(show_all=True)`, and for each env creates a `CTkButton` inside the scrollable frame with:
  - `text=env['name']`, `fg_color="transparent"`, `anchor="w"`, `height=36`
  - Command=lambda n=env['name']: self._select_env(n)
- `_select_env(name)` highlights the selected button (set its `fg_color` to the CTk accent color via `ctk.ThemeManager.theme["CTkButton"]["fg_color"][1]` or by using the master frame's default button color), deselects others, then calls `on_select_callback(name)`.
- `_on_new()` calls `on_select_callback(None)` to signal "new environment" mode.

**Class 3: `EnvironmentFormPanel(ctk.CTkFrame)`**
- `__init__(self, master, on_refresh_callback)`
- Maintain state: `self.mode` ∈ `{"idle", "create", "edit"}`, `self.selected_env_name = None`
- Build the form fields vertically with `pady=(0, 16)` between fields (md spacing token):
  1. Title label: `CTkLabel(font=("", 18, "bold"))` — text is "Environment Details" or "New Environment"
  2. Name: `CTkEntry(placeholder_text="Environment Name")`
  3. Host: `CTkEntry(placeholder_text="hostname or IP")`
  4. Port: `CTkEntry(placeholder_text="1521")`
  5. User: `CTkEntry(placeholder_text="username")`
  6. Password: `CTkEntry(show="*", placeholder_text="Enter new password to change")` — write-only per D-02
  7. Service: `CTkEntry(placeholder_text="service name or SID")`
  8. DB Type: `CTkOptionMenu(values=["ORACLE","POSTGRES","MYSQL","MSSQL"], command=self._on_db_type_change)`
  9. Connection Type: `CTkOptionMenu(values=["service_name","sid"])` — only shown when DB Type == "ORACLE"; initially hidden via `pack_forget()`
  10. Encrypt: `CTkCheckBox(text="Encrypt credentials", command=self._on_encrypt_toggle)`
  11. Encryption Key: `CTkEntry(show="*", placeholder_text="encryption key")` — only shown when Encrypt is checked; initially hidden via `pack_forget()`
- Button row (horizontal `CTkFrame`):
  - "Save Environment" — accent color, calls `_on_save()`
  - "Test Connection" — accent color, calls `_on_test()`; visible only in edit mode
  - "Delete" — `fg_color="#DC3545"`, `hover_color="#C82333"`, calls `_on_delete()`; visible only in edit mode
  - "Discard Changes" — `fg_color="transparent"`, `border_width=1`, calls `_on_discard()`
- Error label: `CTkLabel(text_color="#DC3545", font=("", 12))` below buttons
- Status label: `CTkLabel(font=("", 12))` below error label

**State Machine Methods:**
- `set_mode(mode, env_name=None, env_data=None)`
  - `"idle"`: clear all entries, disable them, hide all buttons, clear title to "Environment Details"
  - `"create"`: clear entries, enable all, show Save + Discard, hide Test + Delete, title = "New Environment"
  - `"edit"`: populate fields from `env_data` (except password which stays empty/placeholder per D-02), disable name entry, show all four buttons, title = env_name
- `_on_db_type_change(value)` — show/hide Connection Type dropdown and its label via `pack()` / `pack_forget()`
- `_on_encrypt_toggle()` — show/hide Encryption Key field and label

**API Integration:**
- `_on_save()` — batch validate, then:
  - If mode == "create": call `api.create_environment(...)` with all field values; on success call `on_refresh_callback()`, call `set_mode("idle")`
  - If mode == "edit": call `api.update_environment(name=self.selected_env_name, ...)` passing only non-empty fields (None for empty strings so partial update works); on success same refresh+idle
  - On any failure: show error text in the error label
- `_on_test()` — call `api.test_environment(name=self.selected_env_name, encryption_key=...)`. Button text changes to "Testing..." and `state="disabled"` during call. On success show green "✓ Connected successfully"; on failure show red "✗ Connection failed: {message}" in status label. Re-enable button after.
- `_on_delete()` — `messagebox.askyesno("Delete '{name}'", "Are you sure you want to delete this environment? This action cannot be undone.")`; if yes, call `api.delete_environment(name)`, on success refresh list and set idle.
- `_on_discard()` — call `set_mode("idle")`

**Validation (`_validate()`):**
- Create mode: `name` non-empty and not `"*"`, `host` non-empty, `port` valid int, `user` non-empty, `password` non-empty, `service` non-empty, `db_type` selected. If encrypt checked → `encryption_key` non-empty. If db_type == ORACLE → `connection_type` selected.
- Edit mode: only validate that at least one of host/port/user/password/service/db_type is non-empty (partial updates). If encrypt checked → `encryption_key` non-empty. If db_type == ORACLE → `connection_type` selected.
- Return list of error strings; if non-empty, join with ", " and display in error label.

**Copywriting:** use exact strings from UI-SPEC Copywriting Contract for all labels, placeholders, button text, success/error messages, and dialog text.
</action>

<acceptance_criteria>
- File `elm/gui/environment_manager.py` exists and is importable via `python -c "from elm.gui.environment_manager import EnvironmentManagerFrame; print('ok')"`
- Contains `class EnvironmentManagerFrame(ctk.CTkFrame)`
- Contains `class EnvironmentListPanel(ctk.CTkFrame)` with `refresh()` method
- Contains `class EnvironmentFormPanel(ctk.CTkFrame)` with `set_mode(self, mode, env_name=None, env_data=None)` method signature
- Password entry uses `show="*"` and placeholder text includes `"Enter new password to change"`
- DB Type dropdown values are exactly `["ORACLE","POSTGRES","MYSQL","MSSQL"]`
- Connection Type dropdown values are exactly `["service_name","sid"]`
- Delete button has `fg_color="#DC3545"` and `hover_color="#C82333"`
- Error label has `text_color="#DC3545"`
- Uses `messagebox.askyesno` for delete confirmation
- Imports `elm.api` and `customtkinter as ctk`
</acceptance_criteria>

---

### Task 2: Wire EnvironmentManagerFrame into app.py

<type>modify</type>
<files>elm/gui/app.py</files>

<read_first>
- elm/gui/app.py (MUST read current content before editing)
- elm/gui/environment_manager.py (the file created in Task 1 — verify it exists and is importable)
</read_first>

<action>
Replace the placeholder label in the `_build_tabs` method of `ELMApp` with the live `EnvironmentManagerFrame`.

Find these lines in `elm/gui/app.py`:
```python
        env_label = ctk.CTkLabel(
            self.tabview.tab("🌐  Environments"),
            text="Environment management will appear here.",
            font=("", 14),
        )
        env_label.pack(expand=True)
```

Replace them with:
```python
        from elm.gui.environment_manager import EnvironmentManagerFrame
        self.env_manager = EnvironmentManagerFrame(self.tabview.tab("🌐  Environments"))
        self.env_manager.pack(fill="both", expand=True)
```

The import is placed inside `_build_tabs` (not at top of file) to prevent circular import issues during module-level import of `app.py`. `EnvironmentManagerFrame` is a `ctk.CTkFrame` subclass that fills its parent tab via `pack(fill="both", expand=True)`.
</action>

<acceptance_criteria>
- `elm/gui/app.py` no longer contains the placeholder text `"Environment management will appear here."`
- `elm/gui/app.py` contains `from elm.gui.environment_manager import EnvironmentManagerFrame`
- `elm/gui/app.py` contains `self.env_manager = EnvironmentManagerFrame(...)`
- `elm/gui/app.py` contains `self.env_manager.pack(fill="both", expand=True)`
- Running `python -c "from elm.gui.app import ELMApp; print('ok')"` succeeds (no import errors)
</acceptance_criteria>

</tasks>

<verification>
## Verification Steps

1. **Module importable (no GUI launch):**
   ```bash
   python -c "from elm.gui.environment_manager import EnvironmentManagerFrame, EnvironmentListPanel, EnvironmentFormPanel; print('ok')"
   ```
   Expected: prints `ok`, exit code 0.

2. **App module still importable after wiring:**
   ```bash
   python -c "from elm.gui.app import ELMApp; print('ok')"
   ```
   Expected: prints `ok`, exit code 0.

3. **GUI launches without crash (manual verification):**
   ```bash
   python -m elm.elm
   ```
   Expected: CustomTkinter window opens. The "🌐  Environments" tab shows a left panel (≈220px wide) with a "＋ New" button and a right panel with form fields. No placeholder text is visible.

4. **Form fields present (code verification):**
   ```bash
   grep -E "CTkEntry|CTkOptionMenu|CTkCheckBox|CTkButton" elm/gui/environment_manager.py | wc -l
   ```
   Expected: ≥ 10 widget creation lines.

5. **API integration present:**
   ```bash
   grep -E "api\.(create|update|delete|test|list)_environment" elm/gui/environment_manager.py
   ```
   Expected: matches found for create, update, delete, test, and list.

6. **Password write-only enforced:**
   ```bash
   grep -n "password" elm/gui/environment_manager.py
   ```
   Expected: no line populates the password entry from `env_data`; only `show="*"` and placeholder text references.
</verification>

<must_haves>
- [ ] Environment list is scrollable and displays all configured environments (ENV-01)
- [ ] Form supports creating a new environment with all required fields (ENV-02)
- [ ] Form supports editing an existing environment; name field is disabled in edit mode (ENV-03)
- [ ] Delete button with confirmation dialog removes the environment (ENV-04)
- [ ] Password field is write-only: never populated from stored data, always shows placeholder (D-02)
- [ ] Test Connection button calls `api.test_environment` and shows success/failure status (D-03)
- [ ] Split pane layout: fixed 220px left list panel, expanding right form panel (D-01)
- [ ] Form state machine supports idle / create / edit modes with correct button visibility
- [ ] Conditional visibility: Connection Type only shown for ORACLE; Encryption Key only shown when Encrypt checked
- [ ] Batch validation on Save with clear error messages in red label
- [ ] All UI text matches the Copywriting Contract from UI-SPEC exactly
</must_haves>

## PLANNING COMPLETE
