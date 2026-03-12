from quam.core import quam_dataclass
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD, LossDiVincenzoQuam
from calibration_utils.run_video_mode.simulated_video_mode.demo_files.demo_quam_ld import DemoQuamLD


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from BaseQuamQD
@quam_dataclass
class Quam(DemoQuamLD):
    pass


@quam_dataclass
class QubitQuam(DemoQuamLD):
    pass
