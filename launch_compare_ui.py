"""Run inside Houdini; keep this file beside houdini_success_failure_compare.py."""
from __future__ import annotations

import runpy
import os


tool = runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "houdini_success_failure_compare.py"))
tool["show_compare_ui"]()
