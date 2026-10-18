"""Launch Success / Failure Compare inside Houdini (same style as launch_ui.py)."""
from __future__ import annotations

import runpy


tool = runpy.run_path(r"C:\Users\username\Documents\houdinitotext\houdini_success_failure_compare.py")
tool["show_compare_ui"]()
