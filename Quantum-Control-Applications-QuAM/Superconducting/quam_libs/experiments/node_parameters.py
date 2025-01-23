from typing import Optional, List, Literal

from pydantic import Field
from qualibrate.parameters import RunnableParameters


class QubitsExperimentNodeParameters(RunnableParameters):
    qubits: Optional[List[str]] = None


class SimulatableNodeParameters(RunnableParameters):
    simulate: bool = Field(False, description="Simulate the waveforms on the OPX instead of executing the program")
    simulation_duration_ns: int = Field(2500, description="Duration over which the simulation will collect samples")
    use_waveform_report: bool = Field(False, description="Whether to use the interactive waveform report in simulation")


class FluxControlledNodeParameters(RunnableParameters):
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"


class MultiplexableNodeParameters(RunnableParameters):
    multiplexed: bool = Field(False, description="Play control, readout and reset aligned and in parallel for all qubits")


class DataLoadableNodeParameters(RunnableParameters):
    load_data_id: Optional[int] = None


class QmSessionNodeParameters(RunnableParameters):
    timeout: int = 100