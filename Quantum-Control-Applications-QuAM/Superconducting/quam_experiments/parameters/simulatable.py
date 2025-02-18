from pydantic import Field
from qualibrate.parameters import RunnableParameters

class SimulatableNodeParameters(RunnableParameters):
    simulate: bool = Field(
        default=False,
        description="Simulate the waveforms on the OPX instead of executing the program"
    )
    simulation_duration_ns: int = Field(
        default=2500,
        description="Duration over which the simulation will collect samples"
    )
    use_waveform_report: bool = Field(
        default=False,
        description="Whether to use the interactive waveform report in simulation"
    )
