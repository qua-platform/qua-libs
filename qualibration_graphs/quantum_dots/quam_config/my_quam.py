from pathlib import Path
from typing import Optional, Union

from quam.core import quam_dataclass
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD, LossDiVincenzoQuam

DIR = Path(__file__).resolve().parent
DEFAULT_QUAM_STATE_DIR = DIR / "quam_state"

# Define the QUAM class that will be used in all calibration nodes
# Should inherit from BaseQuamQD
@quam_dataclass
class Quam(LossDiVincenzoQuam):
    pass


@quam_dataclass
class QubitQuam(LossDiVincenzoQuam):
    pass
