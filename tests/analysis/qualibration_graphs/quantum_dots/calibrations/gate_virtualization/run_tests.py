"""Run gate virtualization analysis tests."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
TEST_DIR = Path(__file__).resolve().parent
WORKING_DIR = REPO_ROOT / "qualibration_graphs" / "quantum_dots"

result = subprocess.run(
    [sys.executable, "-m", "pytest", str(TEST_DIR), "-m", "analysis", "-v"],
    cwd=WORKING_DIR,
)
sys.exit(result.returncode)
