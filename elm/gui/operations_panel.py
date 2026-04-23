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

        self.status_label.configure(text="Running...", text_color=None)
        self.execute_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")

        self._clear_log()

        self.log_queue = queue.Queue()
        self.cancel_event = threading.Event()

        self.worker_thread = threading.Thread(
            target=self._copy_worker,
            args=(source, target, query, table, self.log_queue, self.cancel_event),
            daemon=True,
        )
        self.worker_thread.start()
        self.after(100, self._poll_queue)

    def _copy_worker(self, source_env, target_env, query, table, log_queue, cancel_event):
        """Run copy in background, redirecting stdout/stderr to the queue."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = QueueStream(log_queue)
        sys.stderr = QueueStream(log_queue)

        try:
            result = api.copy_db_to_db(
                source_env=source_env,
                target_env=target_env,
                query=query,
                table=table,
            )
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
        """Reset UI after worker terminates."""
        self.execute_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")
        self.status_label.configure(text="Ready")
        self._drain_remaining_queue()

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
