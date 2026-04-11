
from pathlib import Path
import sys

_PKG_DIR = Path(__file__).resolve().parent
_SRC_DIR = _PKG_DIR.parent / 'src'
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
__path__ = [str(_SRC_DIR)]
