from dataclasses import field
from typing import Dict, Any, Union, Optional, Literal, Callable

import numpy as np
from qm import QuantumMachine, logger
from qm.qua import align, wait, declare, fixed, assign, Cast, StreamType, while_, save, if_, update_frequency
from qualang_tools.octave_tools import octave_calibration_tool
from quam.components import Qubit
from quam.components.channels import IQChannel, Pulse
from quam.core import quam_dataclass

from .flux_line import FluxLine
from .readout_resonator import ReadoutResonator

__all__ = ["Transmon"]


@quam_dataclass
class Transmon(Qubit):
    """
    Example QuAM component for a transmon qubit.

    Args:
        id (str, int): The id of the Transmon, used to generate the name.
            Can be a string, or an integer in which case it will add`Channel._default_label`.
        xy (IQChannel): The xy drive component.
        z (FluxLine): The z drive component.
        resonator (ReadoutResonator): The readout resonator component.
        T1 (float): The transmon T1 in s.
        T2ramsey (float): The transmon T2* in s.
        T2echo (float): The transmon T2 in s.
        thermalization_time_factor (int): thermalization time in units of T1.
        anharmonicity (int, float): the transmon anharmonicity in Hz.
        freq_vs_flux_01_quad_term (float):
        arbitrary_intermediate_frequency (float):
        sigma_time_factor:
        phi0_current (float):
        phi0_voltage (float):
        GEF_frequency_shift (int):
        chi (float):
        grid_location (str): qubit location in the plot grid as "(column, row)"
    """

    id: Union[int, str]

    xy: IQChannel = None
    z: FluxLine = None
    resonator: ReadoutResonator = None

    f_01: float = None
    f_12: float = None
    anharmonicity: int = 150e6
    freq_vs_flux_01_quad_term: float = 0.0
    arbitrary_intermediate_frequency: float = 0.0

    T1: float = 10e-6
    T2ramsey: float = None
    T2echo: float = None
    thermalization_time_factor: int = 5
    sigma_time_factor: int = 5
    phi0_current: float = 0.0
    phi0_voltage: float = 0.0

    GEF_frequency_shift: int = 10
    chi: float = 0.0
    grid_location: str = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def get_output_power(self, operation, Z=50) -> float:
        power = self.xy.opx_output.full_scale_power_dbm
        amplitude = self.xy.operations[operation].amplitude
        x_mw = 10 ** (power / 10)
        x_v = amplitude * np.sqrt(2 * Z * x_mw / 1000)
        return 10 * np.log10(((x_v / np.sqrt(2)) ** 2 * 1000) / Z)

    @property
    def inferred_f_12(self) -> float:
        """The 0-2 (e-f) transition frequency in Hz, derived from f_01 and anharmonicity"""
        name = getattr(self, "name", self.__class__.__name__)
        if not isinstance(self.f_01, (float, int)):
            raise AttributeError(f"Error inferring f_12 for channel {name}: {self.f_01=} is not a number")
        if not isinstance(self.anharmonicity, (float, int)):
            raise AttributeError(f"Error inferring f_12 for channel {name}: {self.anharmonicity=} is not a number")
        return self.f_01 + self.anharmonicity

    @property
    def inferred_anharmonicity(self) -> float:
        """The transmon anharmonicity in Hz, derived from f_01 and f_12."""
        name = getattr(self, "name", self.__class__.__name__)
        if not isinstance(self.f_01, (float, int)):
            raise AttributeError(f"Error inferring anharmonicity for channel {name}: {self.f_01=} is not a number")
        if not isinstance(self.f_12, (float, int)):
            raise AttributeError(f"Error inferring anharmonicity for channel {name}: {self.f_12=} is not a number")
        return self.f_12 - self.f_01

    #@property
    def sigma(self, operation: Pulse):
        return operation.length / self.sigma_time_factor

    @property
    def thermalization_time(self):
        """The transmon thermalization time in ns."""
        return int(self.thermalization_time_factor * self.T1 * 1e9 / 4) * 4

    def calibrate_octave(
        self, QM: QuantumMachine, calibrate_drive: bool = True, calibrate_resonator: bool = True
    ) -> None:
        """Calibrate the Octave channels (xy and resonator) linked to this transmon for the LO frequency, intermediate
        frequency and Octave gain as defined in the state.

        Args:
            QM (QuantumMachine): the running quantum machine.
            calibrate_drive (bool): flag to calibrate xy line.
            calibrate_resonator (bool): flag to calibrate the resonator line.
        """
        if calibrate_resonator and self.resonator is not None:
            logger.info(f"Calibrating {self.resonator.name}")
            octave_calibration_tool(
                QM,
                self.resonator.name,
                lo_frequencies=self.resonator.frequency_converter_up.LO_frequency,
                intermediate_frequencies=self.resonator.intermediate_frequency,
            )

        if calibrate_drive and self.xy is not None:
            logger.info(f"Calibrating {self.xy.name}")
            octave_calibration_tool(
                QM,
                self.xy.name,
                lo_frequencies=self.xy.frequency_converter_up.LO_frequency,
                intermediate_frequencies=self.xy.intermediate_frequency,
            )

    def set_gate_shape(self, gate_shape: str) -> None:
        """Set the shape fo the single qubit gates defined as ["x180", "x90" "-x90", "y180", "y90", "-y90"]"""
        for gate in ["x180", "x90", "-x90", "y180", "y90", "-y90"]:
            self.xy.operations[gate] = f"#./{gate}_{gate_shape}"

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"

    def __matmul__(self, other):
        if not isinstance(other, Transmon):
            raise ValueError(
                "Cannot create a qubit pair (q1 @ q2) with a non-qubit object, " f"where q1={self} and q2={other}"
            )

        if self is other:
            raise ValueError("Cannot create a qubit pair with same qubit (q1 @ q1), where q1={self}")

        for qubit_pair in self.get_root().qubit_pairs.values():
            if qubit_pair.qubit_control is self and qubit_pair.qubit_target is other:
                return qubit_pair
        else:
            raise ValueError("Qubit pair not found: qubit_control={self.name}, " "qubit_target={other.name}")

    def align(self, other = None):
        channels = [self.xy.name, self.resonator.name, self.z.name]

        if other is not None:
            channels += [other.xy.name, other.resonator.name, other.z.name]

        align(*channels)

    def wait(self, duration):
        wait(duration, self.xy.name, self.z.name, self.resonator.name)

    def readout_state(self, state, pulse_name: str = "readout", threshold: float = None):
        """
        Perform a readout of the qubit state using the specified pulse.

        This function measures the qubit state using the specified readout pulse and assigns the result to the given state variable.
        If no threshold is provided, the default threshold for the specified pulse is used.

        Args:
            state: The variable to assign the readout result to.
            pulse_name (str): The name of the readout pulse to use. Default is "readout".
            threshold (float, optional): The threshold value for the readout. If None, the default threshold for the pulse is used.

        Returns:
            None

        The function declares fixed variables I and Q, measures the qubit state using the specified pulse, and assigns the result to the state variable based on the threshold.
        It then waits for the resonator depletion time.
        """
        I = declare(fixed)
        Q = declare(fixed)
        if threshold is None:
            threshold = self.resonator.operations[pulse_name].threshold
        self.resonator.measure(pulse_name, qua_vars=(I, Q))
        assign(state, Cast.to_int(I > threshold))
        wait(self.resonator.depletion_time // 4, self.resonator.name)


    def reset(
        self,
        reset_type: Literal["thermal", "active", "active_gef"] = "thermal",
        simulate: bool = False,
        log_callable: Optional[Callable] = None,
        **kwargs,
    ):
        """
        Reset the qubit with the specified method.

        This function resets the qubit using the specified method: thermal reset, active reset, or active GEF reset.
        When simulating the QUA program, the qubit reset is skipped to save simulated samples.

        Args:
            reset_type (Literal["thermal", "active", "active_gef"]): The type of reset to perform. Default is "thermal".
            simulate (bool): If True, the qubit reset is skipped for simulation purposes. Default is False.
            log_callable (optional): Logger instance to log warnings. If None, a default logger is used.
            **kwargs: Additional keyword arguments passed to the active reset methods.

        Returns:
            None

        Raises:
            Warning: If the function is called in simulation mode, a warning is issued indicating
                     that the qubit reset has been skipped.
        """
        if not simulate:
            if reset_type == "thermal":
                self.reset_qubit_thermal()
            elif reset_type == "active":
                self.reset_qubit_active(**kwargs)
            elif reset_type == "active_gef":
                self.reset_qubit_active_gef(**kwargs)
        else:
            logger.warning(
                "For simulating the QUA program, the qubit reset has been skipped."
            )

    def reset_qubit_thermal(self):
        """
        Perform a thermal reset of the qubit.

        This function waits for a duration specified by the thermalization time
        to allow the qubit to return to its ground state through natural thermal
        relaxation.
        """
        self.wait(self.thermalization_time // 4)

    def reset_qubit_active(
        self,
        save_qua_var: Optional[StreamType] = None,
        pi_pulse_name: str = "x180",
        readout_pulse_name: str = "readout",
        max_attempts: int = 15,
    ):
        """
        Perform an active reset of the qubit.

        This function performs an active reset of the qubit by repeatedly measuring the qubit state and applying a pi pulse
        until the qubit is in the ground state or the maximum number of attempts is reached.

        Args:
            save_qua_var (Optional[StreamType]): The QUA variable to save the number of attempts to.
            pi_pulse_name (str): The name of the pi pulse to use for the reset. Default is "x180".
            readout_pulse_name (str): The name of the readout pulse to use for measuring the qubit state. Default is "readout".
            max_attempts (int): The maximum number of attempts to reset the qubit. Default is 15.

        Returns:
            None

        The function measures the qubit state using the specified readout pulse, applies a pi pulse if the qubit is not in the ground state,
        and repeats this process until the qubit is in the ground state or the maximum number of attempts is reached.
        If `save_qua_var` is provided, the number of attempts is saved to this variable.
        """
        pulse = self.resonator.operations[readout_pulse_name]

        I = declare(fixed)
        Q = declare(fixed)
        state = declare(bool)
        attempts = declare(int, value=1)
        assign(attempts, 1)
        self.align()
        self.resonator.measure("readout", qua_vars=(I, Q))
        assign(state, I > pulse.threshold)
        wait(self.resonator.depletion_time // 2, self.resonator.name)
        self.xy.play(pi_pulse_name, condition=state)
        self.align()
        with while_((I > pulse.rus_exit_threshold) & (attempts < max_attempts)):
            self.align()
            self.resonator.measure("readout", qua_vars=(I, Q))
            assign(state, I > pulse.threshold)
            wait(self.resonator.depletion_time // 2, self.resonator.name)
            self.xy.play(pi_pulse_name, condition=state)
            self.align()
            assign(attempts, attempts + 1)
        wait(500, self.xy.name)
        self.align()
        if save_qua_var is not None:
            save(attempts, save_qua_var)

    def reset_qubit_active_gef(
        self,
        readout_pulse_name: str = "readout",
        pi_01_pulse_name: str = "x180",
        pi_12_pulse_name: str = "EF_x180",
    ):
        """
        Reset the qubit to the ground state ('g') using active reset with GEF state readout.

        This function performs an active reset of the qubit by repeatedly measuring its state
        and applying appropriate pulses to bring it back to the ground state ('g'). The process
        continues until the qubit is measured in the ground state twice in a row to ensure high
        confidence in the reset.

        Args:
            readout_pulse_name (str, optional): The name of the pulse to use for the readout. Defaults to "readout".
            pi_01_pulse_name (str, optional): The name of the pulse to use for the 0-1 transition. Defaults to "x180".
            pi_12_pulse_name (str, optional): The name of the pulse to use for the 1-2 transition. Defaults to "EF_x180".

        Returns:
            None
        """
        res_ar = declare(int)
        success = declare(int)
        assign(success, 0)
        attempts = declare(int)
        assign(attempts, 0)
        self.align()
        with while_(success < 2):
            self.readout_state_gef(res_ar, readout_pulse_name)
            wait(self.rr.res_deplete_time // 4, self.xy.name)
            self.align()
            with if_(res_ar == 0):
                assign(
                    success, success + 1
                )  # we need to measure 'g' two times in a row to increase our confidence
            with if_(res_ar == 1):
                update_frequency(self.xy.name, int(self.xy.intermediate_frequency))
                self.xy.play(pi_01_pulse_name)
                assign(success, 0)
            with if_(res_ar == 2):
                update_frequency(
                    self.xy.name,
                    int(self.xy.intermediate_frequency - self.anharmonicity),
                )
                self.xy.play(pi_12_pulse_name)
                update_frequency(self.xy.name, int(self.xy.intermediate_frequency))
                self.xy.play(pi_01_pulse_name)
                assign(success, 0)
            self.align()
            assign(attempts, attempts + 1)
