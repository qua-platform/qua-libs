from typing import Optional
from pydantic import Field
from qualibrate.parameters import RunnableParameters


class CommonNodeParameters(RunnableParameters):
    simulate: bool = Field(
        default=False, description="Simulate the waveforms on the OPX instead of executing the program."
    )
    simulation_duration_ns: int = Field(
        default=50_000, description="Duration over which the simulation will collect samples (in nanoseconds)."
    )
    use_waveform_report: bool = Field(
        default=True, description="Whether to use the interactive waveform report in simulation."
    )
    timeout: int = Field(
        default=120,
        description="Waiting time for the OPX resources to become available before giving up (in seconds).",
    )
    load_data_id: Optional[int] = Field(
        default=None, description="Qualibrate node run index for loading historical data."
    )
