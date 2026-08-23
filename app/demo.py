"""
app/demo.py
Deprecated entry point — kept for backward compatibility.

The full, production Gradio KYC app now lives in app/gradio_app.py. This shim
simply launches it so existing `python app/demo.py` commands keep working.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.gradio_app import launch

if __name__ == "__main__":
    launch()
