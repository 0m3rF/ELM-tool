"""
History panel for ELM Tool GUI.

Displays a scrollable list of past copy operations with re-run and edit actions.
"""

import customtkinter as ctk
from elm import api


class HistoryPanel(ctk.CTkFrame):
    """Panel displaying copy operation history with re-run and edit actions."""

    def __init__(self, master, app_ref):
        super().__init__(master, fg_color="transparent")
        self.app = app_ref
        self._poll_after_id = None

        # Status label
        self.status_label = ctk.CTkLabel(self, text="Last refreshed: never", font=("", 12))
        self.status_label.pack(anchor="w", pady=(0, 8))

        # Empty state
        self.empty_label = ctk.CTkLabel(
            self, text="No history yet\nRun a copy operation to see it here.",
            font=("", 14)
        )

        # Scrollable list container
        self.list_container = ctk.CTkScrollableFrame(self)

        self._refresh_list()

    def _refresh_list(self):
        """Fetch history records and rebuild the scrollable list."""
        for widget in self.list_container.winfo_children():
            widget.destroy()

        records = api.list_history()

        if not records:
            self.list_container.pack_forget()
            self.empty_label.pack(expand=True)
            self.status_label.configure(text="Last refreshed: " + self._now())
            return

        self.empty_label.pack_forget()
        self.list_container.pack(fill="both", expand=True)

        for record in records:
            self._build_row(record)

        self.status_label.configure(text="Last refreshed: " + self._now())

    def _build_row(self, record):
        """Create a single history entry row inside the scrollable frame."""
        row = ctk.CTkFrame(self.list_container, fg_color="transparent")
        row.pack(fill="x", pady=2)

        status = record.get("status", "unknown")
        color = {"success": "#28A745", "failure": "#DC3545"}.get(status, "#3B8ED0")

        # Compact labels
        id_text = str(record.get("id", "?"))
        date_text = record.get("date", "")[:19].replace("T", " ")  # trim to yyyy-mm-dd hh:mm:ss
        type_text = record.get("operation_type", "")
        src_text = record.get("source", "") or ""
        tgt_text = record.get("target", "") or ""
        table_text = record.get("table", "") or ""

        route = f"{src_text} → {tgt_text}" if src_text or tgt_text else ""

        ctk.CTkLabel(row, text=id_text, font=("", 12), width=40).pack(side="left")
        ctk.CTkLabel(row, text=date_text, font=("", 12), width=120).pack(side="left")
        ctk.CTkLabel(row, text=type_text, font=("", 12), width=80).pack(side="left")
        ctk.CTkLabel(row, text=route, font=("", 12), wraplength=200).pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(row, text=table_text, font=("", 12), width=100).pack(side="left")
        ctk.CTkLabel(row, text=status, font=("", 12), text_color=color, width=80).pack(side="left")

        rerun_btn = ctk.CTkButton(
            row, text="▶ Re-run", width=80,
            command=lambda rid=record["id"]: self._on_rerun(rid)
        )
        rerun_btn.pack(side="right", padx=(4, 0))

        edit_btn = ctk.CTkButton(
            row, text="✎ Edit", width=80,
            command=lambda rid=record["id"]: self._on_edit(rid)
        )
        edit_btn.pack(side="right", padx=(4, 0))

    def _on_rerun(self, record_id):
        """Fetch record and execute via OperationsPanel."""
        record = api.get_history_record(record_id)
        if not record:
            return
        self.app.tabview.set("⚙  Operations")
        self.app.ops_panel.run_history_record(record)

    def _on_edit(self, record_id):
        """Fetch record, pre-fill Operations form, switch to Operations tab."""
        record = api.get_history_record(record_id)
        if not record:
            return
        self.app.tabview.set("⚙  Operations")
        if record.get("operation_type") == "db2db":
            self.app.ops_panel.pre_fill_form(record)
        else:
            # For non-db2db, run directly since the form is db2db-oriented
            self.app.ops_panel.run_history_record(record)

    def _now(self):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def start_polling(self):
        """Begin 15-second refresh poll."""
        self._refresh_list()
        self._poll_after_id = self.after(15000, self._poll_refresh)

    def stop_polling(self):
        """Cancel pending poll."""
        if self._poll_after_id is not None:
            self.after_cancel(self._poll_after_id)
            self._poll_after_id = None

    def _poll_refresh(self):
        """Refresh and reschedule."""
        self._refresh_list()
        self._poll_after_id = self.after(15000, self._poll_refresh)
