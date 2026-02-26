"""Generate the simulated video mode QuAM state from the local video-mode factory."""

from __future__ import annotations

from pathlib import Path
import sys

SIMULATED_VIDEO_MODE_DIR = Path(__file__).resolve().parent
SIMULATED_VIDEO_MODE_QUAM_PATH = SIMULATED_VIDEO_MODE_DIR / "quam_state"
QUANTUM_DOTS_DIR = SIMULATED_VIDEO_MODE_DIR.parents[3]


def _load_create_minimal_quam():
    if str(QUANTUM_DOTS_DIR) not in sys.path:
        sys.path.insert(0, str(QUANTUM_DOTS_DIR))

    from calibration_utils.run_video_mode.simulated_video_mode.quam_factory import (
        create_minimal_quam,
    )

    return create_minimal_quam


def generate_simulated_video_mode_quam():
    """Create the QuAM object used by simulated video mode."""
    return _load_create_minimal_quam()()


def save_simulated_video_mode_quam(
    quam_state_path: str | Path = SIMULATED_VIDEO_MODE_QUAM_PATH,
) -> Path:
    """Generate the simulated QuAM and save it into the local quam_state directory."""
    output_path = Path(quam_state_path)
    output_path.mkdir(parents=True, exist_ok=True)

    machine = generate_simulated_video_mode_quam()
    machine.save(output_path, include_defaults=False)
    return output_path


def main() -> None:
    output_path = save_simulated_video_mode_quam()
    print(output_path / "state.json")


if __name__ == "__main__":
    main()
