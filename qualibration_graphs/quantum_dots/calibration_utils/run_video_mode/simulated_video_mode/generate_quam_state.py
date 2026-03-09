"""Generate the simulated video mode QuAM state from the local video-mode factory."""

from __future__ import annotations

import importlib.util
from pathlib import Path

SIMULATED_VIDEO_MODE_DIR = Path(__file__).resolve().parent
SIMULATED_VIDEO_MODE_QUAM_PATH = SIMULATED_VIDEO_MODE_DIR / "quam_state"


def _load_local_quam_factory():
    module_path = SIMULATED_VIDEO_MODE_DIR / "quam_factory.py"
    spec = importlib.util.spec_from_file_location(
        "simulated_video_mode_quam_factory",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def generate_simulated_video_mode_quam():
    """Create the QuAM object used by simulated video mode."""
    return _load_local_quam_factory().create_minimal_quam()


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
