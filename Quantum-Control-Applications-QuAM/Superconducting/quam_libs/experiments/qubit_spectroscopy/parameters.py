from typing import Optional, Literal

from quam_libs.experiments.node import SimulatableNodeParameters


class Parameters(SimulatableNodeParameters):
    """This is a test class for dataclasses.

    This is the body of the docstring description.

    Attributes:
        num_averages (int): Number of averaging iterations.
        operation (str): The qubit drive operation.
        operation_amplitude_factor (float): The qubit pulse amplitude scale factor relative to the amplitude define in the state.
        operation_len_in_ns (int): The qubit pulse duration.
        frequency_span_in_mhz (float): to do.
        frequency_step_in_mhz (float): to do.
        flux_point_joint_or_independent (str): to do.
        target_peak_width (float): The desired width of the response to the saturation pulse (including saturation amp), in Hz.
        timeout (int): to do.
        load_data_id (int): to do.
        multiplexed (bool): to do.

    """

    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.5
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 100
    frequency_step_in_mhz: float = 0.25
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    target_peak_width: Optional[float] = 3e6
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
