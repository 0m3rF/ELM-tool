"""
Main GUI application window for ELM Tool.

Uses CustomTkinter with CTkTabview for tabbed navigation.
Launched by elm.py when CLI is invoked without arguments.
"""

import customtkinter as ctk
import sys


class ELMApp(ctk.CTk):
    """Main ELM Tool GUI application window."""

    WINDOW_TITLE = "ELM Tool"
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 600

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title(self.WINDOW_TITLE)
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}")
        self.minsize(640, 480)

        # Center window on screen
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = (screen_w - self.WINDOW_WIDTH) // 2
        y = (screen_h - self.WINDOW_HEIGHT) // 2
        self.geometry(f"{self.WINDOW_WIDTH}x{self.WINDOW_HEIGHT}+{x}+{y}")

        # Appearance: system theme, dark-blue accent
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("dark-blue")

        # Clean shutdown on window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Build the tabbed layout
        self._build_tabs()

    def _build_tabs(self):
        """Create the CTkTabview with Environments, Operations, and History tabs."""
        self.tabview = ctk.CTkTabview(self, anchor="nw")
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs with Unicode icons
        self.tabview.add("🌐  Environments")
        self.tabview.add("⚙  Operations")
        self.tabview.add("⏳  History")

        # Set default active tab
        self.tabview.set("🌐  Environments")

        # Environment Manager (Phase 2)
        from elm.gui.environment_manager import EnvironmentManagerFrame
        self.env_manager = EnvironmentManagerFrame(self.tabview.tab("🌐  Environments"))
        self.env_manager.pack(fill="both", expand=True)

        from elm.gui.operations_panel import OperationsPanel
        self.ops_panel = OperationsPanel(self.tabview.tab("⚙  Operations"))
        self.ops_panel.pack(fill="both", expand=True)

        from elm.gui.history_panel import HistoryPanel
        self.history_panel = HistoryPanel(self.tabview.tab("⏳  History"), app_ref=self)
        self.history_panel.pack(fill="both", expand=True)

        # Wire copy-completion callback so History panel refreshes immediately
        self.ops_panel.on_copy_complete_callback = self.history_panel._refresh_list

        # Bind tab switching to start/stop history polling
        self.tabview.configure(command=self._on_tab_change)

    def _on_tab_change(self):
        """Start/stop history polling based on active tab."""
        current = self.tabview.get()
        if current == "⏳  History":
            self.history_panel.start_polling()
        else:
            self.history_panel.stop_polling()

    def _on_close(self):
        """Handle window close: destroy the window and exit the process."""
        self.destroy()
        sys.exit(0)


def launch():
    """Create and run the ELM GUI application. Blocks until window is closed."""
    app = ELMApp()
    app.mainloop()
