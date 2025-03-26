from typing import Optional
from pydantic import Field
from qualibrate.parameters import RunnableParameters


class CommonNodeParameters(RunnableParameters):
    """
    Common parameters for configuring a node in a quantum machine simulation or execution.

    Attributes:
    simulate (bool): Simulate the waveforms on the OPX instead of executing the program. Default is False.
    simulation_duration_ns (int): Duration over which the simulation will collect samples (in nanoseconds). Default is 50,000 ns.
    use_waveform_report (bool): Whether to use the interactive waveform report in simulation. Default is True.
    timeout (int): Waiting time for the OPX resources to become available before giving up (in seconds). Default is 120 seconds.
    load_data_id (Optional[int]): Qualibrate node run index for loading historical data. Default is None.
    """

    simulate: bool = Field(
        default=False,
        description="Simulate the waveforms on the OPX instead of executing the program.",
    )
    simulation_duration_ns: int = Field(default=50_000, gt=16, lt=1_000_000)
    use_waveform_report: bool = Field(
        default=True,
        description="Whether to use the interactive waveform report in simulation.",
    )
    timeout: int = Field(
        default=120,
        description="Waiting time for the OPX resources to become available before giving up (in seconds).",
    )
    load_data_id: Optional[int] = Field(
        default=None,
        description="Qualibrate node run index for loading historical data.",
    )
