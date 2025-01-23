from typing import Optional, List, Literal

from pydantic import Field
from qualibrate.parameters import RunnableParameters


class QubitsExperimentNodeParameters(RunnableParameters):
    qubits: Optional[List[str]] = None


class SimulatableNodeParameters(RunnableParameters):
    simulate: bool = Field(False, description="Simulate the waveforms on the OPX instead of executing the program")
    simulation_duration_ns: int = 2500


class FluxControlledNodeParameters(RunnableParameters):
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"


class MultiplexableNodeParameters(RunnableParameters):
    multiplexed: bool = False


class DataLoadableNodeParameters(RunnableParameters):
    load_data_id: Optional[int] = None


class QmSessionNodeParameters(RunnableParameters):
    timeout: int = 100