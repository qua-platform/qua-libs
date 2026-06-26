"""Save QUAM states from the test factory for sharing.

Generates both the Stage-1 (BaseQuamQD) and Stage-2 (LossDiVincenzoQuam)
machines and writes their serialised state as JSON files.

Usage::

    python -m tests.qualibration_graphs.quantum_dots.calibrations.save_quam_state

Or run directly::

    python tests/qualibration_graphs/quantum_dots/calibrations/save_quam_state.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

OUTPUT_DIR = Path(__file__).resolve().parent / "quam_state"

# build_quam_wiring / reference resolution internally calls get_quam_config()
# which triggers a config migration that is broken in the current env
# (quam 0.4.2 ships v1_v2 but qualibrate-config expects v2_v3).
# Patching it out at every import site lets the factory run purely in-memory.
_tmpdir = tempfile.mkdtemp(prefix="quam_state_")
os.environ["QUAM_STATE_PATH"] = _tmpdir


def _fake_get_quam_config():
    from types import SimpleNamespace

    return SimpleNamespace(
        state_path=_tmpdir,
        raise_error_missing_reference=False,
    )


_patches = [
    patch("quam.config.resolvers.get_quam_config", _fake_get_quam_config),
    patch("quam.config.get_quam_config", _fake_get_quam_config),
    patch("quam.core.quam_classes.get_quam_config", _fake_get_quam_config),
    patch("quam.serialisation.json.get_quam_config", _fake_get_quam_config),
]
for p in _patches:
    p.start()

from tests.qualibration_graphs.quantum_dots.calibrations.quam_factory import (  # noqa: E402
    create_ld_quam,
    create_qd_quam,
)


def _save_machine(machine, output_path: Path) -> None:
    state = machine.to_dict(follow_references=False, include_defaults=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def main() -> None:
    stage1_path = OUTPUT_DIR / "stage1_base_quam.json"
    stage2_path = OUTPUT_DIR / "stage2_loss_divincenzo.json"

    print("Building Stage-1 BaseQuamQD …")
    qd_machine = create_qd_quam()
    _save_machine(qd_machine, stage1_path)
    print(f"  Saved to {stage1_path}")

    print("Building Stage-2 LossDiVincenzoQuam …")
    ld_machine = create_ld_quam()
    _save_machine(ld_machine, stage2_path)
    print(f"  Saved to {stage2_path}")

    print(f"\nDone. Share the {OUTPUT_DIR.name}/ directory with your colleague.")
    for p in _patches:
        p.stop()


if __name__ == "__main__":
    main()
