"""
Operations GUI panel for ELM Tool.

Provides a copy-operation form and a persistent log panel.
Background threads redirect stdout/stderr into a queue that the
main thread drains into a read-only CTkTextbox.
"""

import queue
import sys
import threading

import customtkinter as ctk

from elm import api


class QueueStream:
    """File-like object that writes text into a queue."""

    def __init__(self, log_queue):
        self.log_queue = log_queue

    def write(self, text):
        if isinstance(text, str) and text:
            self.log_queue.put(text)

    def flush(self):
        pass


class OperationsPanel(ctk.CTkFrame):
    """Panel containing the copy form and log display."""

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")

        # Threading state
        self.worker_thread = None
        self.log_queue = queue.Queue()
        self.cancel_event = threading.Event()
        self.on_copy_complete_callback = None

        # ---- Top: Copy Form ----
        self.form_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.form_frame.pack(fill="x", padx=10, pady=(10, 0))

        # Source Environment
        ctk.CTkLabel(
            self.form_frame, text="Source Environment", font=("", 12)
        ).pack(anchor="w")
        self.source_var = ctk.StringVar()
        self.source_menu = ctk.CTkOptionMenu(
            self.form_frame, variable=self.source_var, values=[]
        )
        self.source_menu.pack(fill="x", pady=(0, 8))

        # Target Environment
        ctk.CTkLabel(
            self.form_frame, text="Target Environment", font=("", 12)
        ).pack(anchor="w")
        self.target_var = ctk.StringVar()
        self.target_menu = ctk.CTkOptionMenu(
            self.form_frame, variable=self.target_var, values=[]
        )
        self.target_menu.pack(fill="x", pady=(0, 8))

        # SQL Query
        ctk.CTkLabel(self.form_frame, text="SQL Query", font=("", 12)).pack(anchor="w")
        self.query_input = ctk.CTkTextbox(self.form_frame, height=80)
        self.query_input.pack(fill="x", pady=(0, 8))

        # Target Table
        ctk.CTkLabel(self.form_frame, text="Target Table", font=("", 12)).pack(anchor="w")
        self.table_entry = ctk.CTkEntry(
            self.form_frame, placeholder_text="target_table_name"
        )
        self.table_entry.pack(fill="x", pady=(0, 8))

        # Button row
        self.btn_frame = ctk.CTkFrame(self.form_frame, fg_color="transparent")
        self.btn_frame.pack(fill="x")

        self.execute_btn = ctk.CTkButton(
            self.btn_frame,
            text="▶ Execute",
            command=self._on_execute,
        )
        self.execute_btn.pack(side="left", padx=(0, 8))

        self.cancel_btn = ctk.CTkButton(
            self.btn_frame,
            text="⏹ Cancel",
            fg_color="#DC3545",
            hover_color="#C82333",
            command=self._on_cancel,
            state="disabled",
        )
        self.cancel_btn.pack(side="left")

        # Status label
        self.status_label = ctk.CTkLabel(
            self.form_frame, text="Ready", font=("", 12)
        )
        self.status_label.pack(anchor="w", pady=(4, 0))

        # ---- Bottom: Log Panel ----
        self.log_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.log_frame.pack(fill="both", expand=True, padx=10, pady=10)

        ctk.CTkLabel(
            self.log_frame, text="Execution Log", font=("", 18, "bold")
        ).pack(anchor="w", pady=(0, 8))

        self.log_textbox = ctk.CTkTextbox(self.log_frame, state="disabled")
        self.log_textbox.pack(fill="both", expand=True)

        self.clear_btn = ctk.CTkButton(
            self.log_frame,
            text="Clear Log",
            width=80,
            command=self._clear_log,
        )
        self.clear_btn.pack(anchor="e", pady=(8, 0))

        self._refresh_environments()

    def _refresh_environments(self):
        """Populate source/target dropdowns from API."""
        envs = api.list_environments(show_all=True)
        names = [env["name"] for env in envs if "name" in env]

        if not names:
            self.source_menu.configure(values=["(no environments)"])
            self.target_menu.configure(values=["(no environments)"])
            self.source_var.set("(no environments)")
            self.target_var.set("(no environments)")
            self.execute_btn.configure(state="disabled")
        else:
            self.source_menu.configure(values=names)
            self.target_menu.configure(values=names)
            self.source_var.set(names[0])
            self.target_var.set(names[1] if len(names) > 1 else names[0])
            self.execute_btn.configure(state="normal")

    def _on_execute(self):
        """Validate form and launch background copy thread."""
        source = self.source_var.get().strip()
        target = self.target_var.get().strip()
        query = self.query_input.get("1.0", "end-1c").strip()
        table = self.table_entry.get().strip()

        if not source or not target or not query or not table:
            self.status_label.configure(
                text="Please fill all fields", text_color="#DC3545"
            )
            return

        if source == target:
            self.status_label.configure(
                text="Source and target must differ", text_color="#DC3545"
            )
            return

        self.status_label.configure(text="Running...")
        self.execute_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        self._clear_log()

        record = {
            "operation_type": "db2db",
            "source": source,
            "target": target,
            "query": query,
            "table": table,
            "mode": "APPEND",
            "batch_size": None,
            "parallel_workers": 1,
        }

        self.log_queue = queue.Queue()
        self.cancel_event = threading.Event()

        self.worker_thread = threading.Thread(
            target=self._copy_worker,
            args=(record, self.log_queue, self.cancel_event),
            daemon=True,
        )
        self.worker_thread.start()
        self.after(100, self._poll_queue)

    def _copy_worker(self, record, log_queue, cancel_event):
        """Run copy in background, redirecting stdout/stderr to the queue."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = QueueStream(log_queue)
        sys.stderr = QueueStream(log_queue)

        try:
            op_type = record.get("operation_type", "db2db")
            if op_type == "db2db":
                result = api.copy_db_to_db(
                    source_env=record.get("source"),
                    target_env=record.get("target"),
                    query=record.get("query"),
                    table=record.get("table"),
                    mode=record.get("mode") or "APPEND",
                    batch_size=record.get("batch_size"),
                    parallel_workers=record.get("parallel_workers") or 1,
                )
            elif op_type == "db2file":
                result = api.copy_db_to_file(
                    source_env=record.get("source"),
                    query=record.get("query"),
                    file_path=record.get("target"),
                    mode=record.get("mode") or "REPLACE",
                    batch_size=record.get("batch_size"),
                    parallel_workers=record.get("parallel_workers") or 1,
                )
            elif op_type == "file2db":
                result = api.copy_file_to_db(
                    file_path=record.get("source"),
                    target_env=record.get("target"),
                    table=record.get("table"),
                    mode=record.get("mode") or "APPEND",
                    batch_size=record.get("batch_size"),
                    parallel_workers=record.get("parallel_workers") or 1,
                )
            else:
                result = {"success": False, "message": f"Unknown operation type: {op_type}"}

            if result.get("success"):
                log_queue.put("\n✓ Copy completed successfully.\n")
            else:
                log_queue.put(
                    f"\n✗ Copy failed: {result.get('error_details', result.get('message', 'Unknown error'))}\n"
                )
        except Exception as e:
            log_queue.put(f"\n✗ Error: {str(e)}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def pre_fill_form(self, record):
        """Populate the copy form from a history record (db2db best-effort)."""
        self._refresh_environments()

        envs = api.list_environments(show_all=True)
        env_names = {e["name"] for e in envs if "name" in e}

        missing = []
        src = record.get("source")
        tgt = record.get("target")

        if src and src in env_names:
            self.source_var.set(src)
        elif src:
            missing.append(f"source '{src}'")
            self.source_var.set(src)

        if tgt and tgt in env_names:
            self.target_var.set(tgt)
        elif tgt:
            missing.append(f"target '{tgt}'")
            self.target_var.set(tgt)

        if missing:
            self.status_label.configure(
                text="Warning: missing environments " + ", ".join(missing),
                text_color="#DC3545",
            )

        if record.get("query"):
            self.query_input.delete("1.0", "end")
            self.query_input.insert("1.0", record["query"])
        if record.get("table"):
            self.table_entry.delete(0, "end")
            self.table_entry.insert(0, record["table"])

    def _validate_record_envs(self, record):
        """Check that source/target environments in a record still exist."""
        op_type = record.get("operation_type", "db2db")
        envs = api.list_environments(show_all=True)
        env_names = {e["name"] for e in envs if "name" in e}

        missing = []
        if op_type in ("db2db", "db2file"):
            src = record.get("source")
            if src and src not in env_names:
                missing.append(f"source '{src}'")
        if op_type in ("db2db", "file2db"):
            tgt = record.get("target")
            if tgt and tgt not in env_names:
                missing.append(f"target '{tgt}'")
        return missing

    def run_history_record(self, record):
        """Execute a copy operation from a history record without form interaction.

        The record dict must contain an operation_type key (db2db, db2file, file2db).
        """
        missing_envs = self._validate_record_envs(record)
        if missing_envs:
            self.status_label.configure(
                text="Missing environments: " + ", ".join(missing_envs),
                text_color="#DC3545",
            )
            return

        self.status_label.configure(text="Running...")
        self.execute_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        self._clear_log()

        self.log_queue = queue.Queue()
        self.cancel_event = threading.Event()

        self.worker_thread = threading.Thread(
            target=self._copy_worker,
            args=(record, self.log_queue, self.cancel_event),
            daemon=True,
        )
        self.worker_thread.start()
        self.after(100, self._poll_queue)

    def _poll_queue(self):
        """Drain log queue into the textbox; reschedule while worker lives."""
        while True:
            try:
                text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", text)
            self.log_textbox.configure(state="disabled")
            self.log_textbox.see("end")

        if self.worker_thread and self.worker_thread.is_alive():
            self.after(100, self._poll_queue)
        else:
            self._on_copy_finished()

    def _on_copy_finished(self):
        """Reset UI after worker terminates and notify listeners."""
        self.execute_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.status_label.configure(text="Ready")
        self._drain_remaining_queue()
        if self.on_copy_complete_callback:
            try:
                self.on_copy_complete_callback(force=True)
            except Exception:
                pass

    def _drain_remaining_queue(self):
        """Flush any leftover queue items (no rescheduling)."""
        while True:
            try:
                text = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", text)
            self.log_textbox.configure(state="disabled")
            self.log_textbox.see("end")

    def _on_cancel(self):
        """Request cooperative cancellation."""
        self.cancel_event.set()
        self.status_label.configure(text="Cancelling...", text_color="#DC3545")

    def _clear_log(self):
        """Clear the read-only log textbox."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")
