"""
Environment management GUI for ELM Tool.

Provides a split-pane widget for listing, creating, editing, and deleting
database environments. Uses elm.api for CRUD operations.
"""

import customtkinter as ctk
from tkinter import messagebox
from elm import api


class EnvironmentManagerFrame(ctk.CTkFrame):
    """Main split-pane environment manager widget."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")

        # Left panel (fixed 220px)
        self.left_frame = ctk.CTkFrame(self, width=220, fg_color="transparent")
        self.left_frame.pack(side="left", fill="y")
        self.left_frame.pack_propagate(False)

        # Right panel (expanding)
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.pack(side="right", fill="both", expand=True)

        # Sub-panels
        self.list_panel = EnvironmentListPanel(
            self.left_frame, on_select_callback=self._on_env_selected
        )
        self.list_panel.pack(fill="both", expand=True)

        self.form_panel = EnvironmentFormPanel(
            self.right_frame, on_refresh_callback=self.refresh_list
        )
        self.form_panel.pack(fill="both", expand=True)

        self.refresh_list()

    def refresh_list(self):
        """Refresh the environment list and reset form to idle."""
        self.list_panel.refresh()
        self.form_panel.set_mode("idle")

    def _on_env_selected(self, name):
        """Handle environment selection from list."""
        if name is None:
            self.form_panel.set_mode("create")
        else:
            env_data = api.get_environment(name)
            if env_data:
                self.form_panel.set_mode("edit", env_name=name, env_data=env_data)
            else:
                self.form_panel.set_mode("idle")


class EnvironmentListPanel(ctk.CTkFrame):
    """Scrollable list of environments with New button."""

    def __init__(self, master, on_select_callback):
        super().__init__(master, fg_color="transparent")
        self.on_select_callback = on_select_callback
        self.selected_env = None
        self.env_buttons = {}

        # Header with New button
        self.header = ctk.CTkFrame(self, fg_color="transparent")
        self.header.pack(fill="x", padx=8, pady=8)

        self.new_btn = ctk.CTkButton(
            self.header,
            text="＋ New",
            width=80,
            command=self._on_new,
        )
        self.new_btn.pack(side="right")

        # Scrollable list container
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    def refresh(self):
        """Clear and repopulate the environment list."""
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.env_buttons.clear()
        self.selected_env = None

        envs = api.list_environments(show_all=True)
        for env in envs:
            name = env["name"]
            btn = ctk.CTkButton(
                self.scroll_frame,
                text=name,
                fg_color="transparent",
                anchor="w",
                height=36,
                command=lambda n=name: self._select_env(n),
            )
            btn.pack(fill="x", pady=2)
            self.env_buttons[name] = btn

    def _select_env(self, name):
        """Highlight selected environment and notify callback."""
        if self.selected_env and self.selected_env in self.env_buttons:
            self.env_buttons[self.selected_env].configure(fg_color="transparent")

        self.selected_env = name

        if name in self.env_buttons:
            try:
                accent = ctk.ThemeManager.theme["CTkButton"]["fg_color"][1]
            except Exception:
                accent = "#3B8ED0"
            self.env_buttons[name].configure(fg_color=accent)

        self.on_select_callback(name)

    def _on_new(self):
        """Signal new environment creation."""
        if self.selected_env and self.selected_env in self.env_buttons:
            self.env_buttons[self.selected_env].configure(fg_color="transparent")
        self.selected_env = None
        self.on_select_callback(None)


class EnvironmentFormPanel(ctk.CTkFrame):
    """Environment details form with create/edit/delete capabilities."""

    def __init__(self, master, on_refresh_callback):
        super().__init__(master, fg_color="transparent")
        self.on_refresh_callback = on_refresh_callback
        self.mode = "idle"
        self.selected_env_name = None

        # Title
        self.title_label = ctk.CTkLabel(
            self, text="Environment Details", font=("", 18, "bold")
        )
        self.title_label.pack(anchor="w", padx=16, pady=(16, 8))

        # Form container
        self.form_container = ctk.CTkFrame(self, fg_color="transparent")
        self.form_container.pack(fill="both", expand=True, padx=16, pady=8)

        # Build form fields
        self.fields = {}
        self._build_fields()

        # Button row
        self.button_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.button_frame.pack(fill="x", padx=16, pady=(8, 0))
        self._build_buttons()

        # Error label
        self.error_label = ctk.CTkLabel(
            self, text="", text_color="#DC3545", font=("", 12)
        )
        self.error_label.pack(anchor="w", padx=16, pady=(8, 0))

        # Status label
        self.status_label = ctk.CTkLabel(self, text="", font=("", 12))
        self.status_label.pack(anchor="w", padx=16, pady=(4, 16))

        self.set_mode("idle")

    def _build_fields(self):
        """Create all form field widgets."""
        field_configs = [
            ("name", "Name", "Environment Name"),
            ("host", "Host", "hostname or IP"),
            ("port", "Port", "1521"),
            ("user", "User", "username"),
            ("password", "Password", "Enter new password to change"),
            ("service", "Service", "service name or SID"),
        ]

        for key, label_text, placeholder in field_configs:
            lbl = ctk.CTkLabel(
                self.form_container, text=label_text, font=("", 12)
            )
            lbl.pack(anchor="w", pady=(0, 4))
            entry = ctk.CTkEntry(
                self.form_container, placeholder_text=placeholder
            )
            if key == "password":
                entry.configure(show="*")
            entry.pack(fill="x", pady=(0, 16))
            self.fields[key] = entry

        # DB Type
        self.db_type_label = ctk.CTkLabel(
            self.form_container, text="Database Type", font=("", 12)
        )
        self.db_type_label.pack(anchor="w", pady=(0, 4))
        self.db_type_var = ctk.StringVar(value="ORACLE")
        self.db_type_menu = ctk.CTkOptionMenu(
            self.form_container,
            values=["ORACLE", "POSTGRES", "MYSQL", "MSSQL"],
            variable=self.db_type_var,
            command=self._on_db_type_change,
        )
        self.db_type_menu.pack(fill="x", pady=(0, 16))

        # Connection Type (Oracle only)
        self.conn_type_label = ctk.CTkLabel(
            self.form_container, text="Connection Type", font=("", 12)
        )
        self.conn_type_var = ctk.StringVar(value="service_name")
        self.conn_type_menu = ctk.CTkOptionMenu(
            self.form_container,
            values=["service_name", "sid"],
            variable=self.conn_type_var,
        )

        # Encrypt
        self.encrypt_check = ctk.CTkCheckBox(
            self.form_container,
            text="Encrypt credentials",
            command=self._on_encrypt_toggle,
        )
        self.encrypt_check.pack(anchor="w", pady=(0, 8))

        # Encryption Key
        self.encrypt_key_label = ctk.CTkLabel(
            self.form_container, text="Encryption Key", font=("", 12)
        )
        self.encrypt_key_entry = ctk.CTkEntry(
            self.form_container, show="*", placeholder_text="encryption key"
        )

    def _build_buttons(self):
        """Create action buttons."""
        self.save_btn = ctk.CTkButton(
            self.button_frame, text="Save Environment", command=self._on_save
        )
        self.test_btn = ctk.CTkButton(
            self.button_frame, text="Test Connection", command=self._on_test
        )
        self.delete_btn = ctk.CTkButton(
            self.button_frame,
            text="Delete",
            fg_color="#DC3545",
            hover_color="#C82333",
            command=self._on_delete,
        )
        self.discard_btn = ctk.CTkButton(
            self.button_frame,
            text="Discard Changes",
            fg_color="transparent",
            border_width=1,
            command=self._on_discard,
        )

    def set_mode(self, mode, env_name=None, env_data=None):
        """Switch form state machine: idle, create, or edit."""
        self.mode = mode
        self.selected_env_name = env_name
        self.error_label.configure(text="")
        self.status_label.configure(text="")

        for btn in (self.save_btn, self.test_btn, self.delete_btn, self.discard_btn):
            btn.pack_forget()

        self.conn_type_label.pack_forget()
        self.conn_type_menu.pack_forget()
        self.encrypt_key_label.pack_forget()
        self.encrypt_key_entry.pack_forget()

        if mode == "idle":
            self.title_label.configure(text="Environment Details")
            self._clear_fields()
            self._set_fields_state("disabled")
        elif mode == "create":
            self.title_label.configure(text="New Environment")
            self._clear_fields()
            self._set_fields_state("normal")
            self.fields["name"].configure(state="normal")
            self.save_btn.pack(side="left", padx=(0, 8))
            self.discard_btn.pack(side="left", padx=(0, 8))
        elif mode == "edit":
            self.title_label.configure(text=env_name or "Environment Details")
            self._clear_fields()
            self._populate_fields(env_data)
            self._set_fields_state("normal")
            self.fields["name"].configure(state="disabled")
            self.save_btn.pack(side="left", padx=(0, 8))
            self.test_btn.pack(side="left", padx=(0, 8))
            self.delete_btn.pack(side="left", padx=(0, 8))
            self.discard_btn.pack(side="left", padx=(0, 8))

    def _clear_fields(self):
        """Reset all form fields to defaults."""
        for entry in self.fields.values():
            entry.delete(0, "end")
        self.db_type_menu.set("ORACLE")
        self.conn_type_menu.set("service_name")
        self.encrypt_check.deselect()
        self.encrypt_key_entry.delete(0, "end")
        self._on_db_type_change("ORACLE")
        self._on_encrypt_toggle()

    def _populate_fields(self, env_data):
        """Fill form from environment data. Password remains empty (write-only)."""
        if not env_data:
            return

        self.fields["name"].insert(0, env_data.get("name", ""))
        self.fields["host"].insert(0, env_data.get("host", ""))
        port_val = env_data.get("port", "")
        self.fields["port"].insert(0, str(port_val) if port_val is not None else "")
        self.fields["user"].insert(0, env_data.get("user", ""))
        # Password is intentionally left empty — write-only per D-02
        self.fields["service"].insert(0, env_data.get("service", ""))

        db_type = env_data.get("type", "ORACLE")
        self.db_type_menu.set(db_type)
        self._on_db_type_change(db_type)

        if db_type == "ORACLE":
            self.conn_type_menu.set(env_data.get("connection_type", "service_name"))

        is_encrypted = env_data.get("is_encrypted", "False") == "True"
        if is_encrypted:
            self.encrypt_check.select()
        else:
            self.encrypt_check.deselect()
        self._on_encrypt_toggle()

    def _set_fields_state(self, state):
        """Enable or disable all form fields."""
        for entry in self.fields.values():
            entry.configure(state=state)
        self.db_type_menu.configure(state=state)
        self.conn_type_menu.configure(state=state)
        self.encrypt_check.configure(state=state)
        self.encrypt_key_entry.configure(state=state)

    def _on_db_type_change(self, value):
        """Show/hide Connection Type for Oracle."""
        if value == "ORACLE":
            self.conn_type_label.pack(anchor="w", pady=(0, 4))
            self.conn_type_menu.pack(fill="x", pady=(0, 16))
        else:
            self.conn_type_label.pack_forget()
            self.conn_type_menu.pack_forget()

    def _on_encrypt_toggle(self):
        """Show/hide Encryption Key field."""
        if self.encrypt_check.get():
            self.encrypt_key_label.pack(anchor="w", pady=(0, 4))
            self.encrypt_key_entry.pack(fill="x", pady=(0, 16))
        else:
            self.encrypt_key_label.pack_forget()
            self.encrypt_key_entry.pack_forget()

    def _validate(self):
        """Batch validation on submit."""
        errors = []
        name = self.fields["name"].get().strip()
        host = self.fields["host"].get().strip()
        port_str = self.fields["port"].get().strip()
        user = self.fields["user"].get().strip()
        password = self.fields["password"].get()
        service = self.fields["service"].get().strip()
        db_type = self.db_type_var.get()
        encrypt = self.encrypt_check.get() == 1
        encryption_key = self.encrypt_key_entry.get()

        if self.mode == "create":
            if not name:
                errors.append("Name is required")
            if name == "*":
                errors.append("Name cannot be '*'")
            if not host:
                errors.append("Host is required")
            if not port_str:
                errors.append("Port is required")
            else:
                try:
                    int(port_str)
                except ValueError:
                    errors.append("Port must be a valid integer")
            if not user:
                errors.append("User is required")
            if not password:
                errors.append("Password is required")
            if not service:
                errors.append("Service is required")
            if not db_type:
                errors.append("Database Type is required")
            if encrypt and not encryption_key:
                errors.append("Encryption Key is required when encryption is enabled")
            if db_type == "ORACLE" and not self.conn_type_var.get():
                errors.append("Connection Type is required for Oracle")
        elif self.mode == "edit":
            if not any([host, port_str, user, password, service, db_type]):
                errors.append("At least one field must be provided to update")
            if port_str:
                try:
                    int(port_str)
                except ValueError:
                    errors.append("Port must be a valid integer")
            if encrypt and not encryption_key:
                errors.append("Encryption Key is required when encryption is enabled")
            if db_type == "ORACLE" and not self.conn_type_var.get():
                errors.append("Connection Type is required for Oracle")

        return errors

    def _on_save(self):
        """Handle Save Environment button."""
        errors = self._validate()
        if errors:
            self.error_label.configure(text=", ".join(errors))
            return

        self.error_label.configure(text="")

        name = self.fields["name"].get().strip()
        host = self.fields["host"].get().strip() or None
        port_str = self.fields["port"].get().strip()
        port = int(port_str) if port_str else None
        user = self.fields["user"].get().strip() or None
        password = self.fields["password"].get() or None
        service = self.fields["service"].get().strip() or None
        db_type = self.db_type_var.get() or None
        encrypt = self.encrypt_check.get() == 1
        encryption_key = self.encrypt_key_entry.get() or None
        connection_type = self.conn_type_var.get() if db_type == "ORACLE" else None

        if self.mode == "create":
            success = api.create_environment(
                name=name,
                host=host,
                port=port,
                user=user,
                password=password,
                service=service,
                db_type=db_type,
                encrypt=encrypt,
                encryption_key=encryption_key,
                connection_type=connection_type,
            )
            if success:
                self.status_label.configure(
                    text=f"Environment '{name}' saved.", text_color="#28A745"
                )
                self.on_refresh_callback()
                self.set_mode("idle")
            else:
                self.error_label.configure(text="Failed to create environment")
        elif self.mode == "edit":
            kwargs = {"name": self.selected_env_name}
            if host:
                kwargs["host"] = host
            if port is not None:
                kwargs["port"] = port
            if user:
                kwargs["user"] = user
            if password:
                kwargs["password"] = password
            if service:
                kwargs["service"] = service
            if db_type:
                kwargs["db_type"] = db_type
            if encrypt is not None:
                kwargs["encrypt"] = encrypt
            if encryption_key:
                kwargs["encryption_key"] = encryption_key

            success = api.update_environment(**kwargs)
            if success:
                self.status_label.configure(
                    text=f"Environment '{self.selected_env_name}' saved.",
                    text_color="#28A745",
                )
                self.on_refresh_callback()
                self.set_mode("idle")
            else:
                self.error_label.configure(text="Failed to update environment")

    def _on_test(self):
        """Handle Test Connection button."""
        if not self.selected_env_name:
            return

        self.test_btn.configure(text="Testing...", state="disabled")
        self.status_label.configure(text="")

        encryption_key = self.encrypt_key_entry.get() or None
        result = api.test_environment(
            self.selected_env_name, encryption_key=encryption_key
        )

        self.test_btn.configure(text="Test Connection", state="normal")

        if result.get("success"):
            self.status_label.configure(
                text="Connected successfully", text_color="#28A745"
            )
        else:
            msg = result.get("message", "Unknown error")
            self.status_label.configure(
                text=f"Connection failed: {msg}", text_color="#DC3545"
            )

    def _on_delete(self):
        """Handle Delete button with confirmation."""
        if not self.selected_env_name:
            return

        confirmed = messagebox.askyesno(
            f"Delete '{self.selected_env_name}'",
            "Are you sure you want to delete this environment? This action cannot be undone.",
        )
        if confirmed:
            success = api.delete_environment(self.selected_env_name)
            if success:
                self.status_label.configure(
                    text=f"Environment '{self.selected_env_name}' deleted.",
                    text_color="#28A745",
                )
                self.on_refresh_callback()
                self.set_mode("idle")
            else:
                self.error_label.configure(text="Failed to delete environment")

    def _on_discard(self):
        """Handle Discard Changes button."""
        self.set_mode("idle")
