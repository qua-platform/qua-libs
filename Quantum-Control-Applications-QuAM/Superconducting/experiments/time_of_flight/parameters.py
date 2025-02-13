from typing import Optional, List

from experiments.node_parameters import SimulatableNodeParameters, QubitsExperimentNodeParameters


class Parameters(QubitsExperimentNodeParameters, SimulatableNodeParameters):
    num_averages: int = 100
    time_of_flight_in_ns: Optional[int] = 24
    intermediate_frequency_in_mhz: Optional[float] = 50
    readout_amplitude_in_v: Optional[float] = 0.1
    readout_length_in_ns: Optional[int] = None
    timeout: int = 100


