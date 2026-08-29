"""Ensures `import shared...` / `import services...` resolve when running
pytest from the repo root, without needing an editable pip install for this
research-project-sized codebase."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
