from typing import Optional
from pydantic import Field
from qualibrate.parameters import RunnableParameters


class CommonNodeParameters(RunnableParameters):
    """Common parameters for configuring a node in a quantum machine simulation or execution."""

    simulate: bool = False
    """Simulate the waveforms on the OPX instead of executing the program. Default is False."""
    simulation_duration_ns: int = Field(default=50_000, gt=16, lt=1_000_000)
    """Duration over which the simulation will collect samples (in nanoseconds). Default is 50_000 ns."""
    use_waveform_report: bool = True
    """Whether to use the interactive waveform report in simulation. Default is True."""
    timeout: int = 120
    """Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 s."""
    load_data_id: Optional[int] = None
    """Optional QUAlibrate node run index for loading historical data. Default is None."""
