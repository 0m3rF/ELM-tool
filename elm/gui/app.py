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
        """Create the CTkTabview with Environments and Operations tabs."""
        self.tabview = ctk.CTkTabview(self, anchor="nw")
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Add tabs with Unicode icons
        self.tabview.add("🌐  Environments")
        self.tabview.add("⚙  Operations")

        # Set default active tab
        self.tabview.set("🌐  Environments")

        # Placeholder labels (will be replaced in Phase 2 and 3)
        env_label = ctk.CTkLabel(
            self.tabview.tab("🌐  Environments"),
            text="Environment management will appear here.",
            font=("", 14),
        )
        env_label.pack(expand=True)

        ops_label = ctk.CTkLabel(
            self.tabview.tab("⚙  Operations"),
            text="Copy operations will appear here.",
            font=("", 14),
        )
        ops_label.pack(expand=True)

    def _on_close(self):
        """Handle window close: destroy the window and exit the process."""
        self.destroy()
        sys.exit(0)


def launch():
    """Create and run the ELM GUI application. Blocks until window is closed."""
    app = ELMApp()
    app.mainloop()
