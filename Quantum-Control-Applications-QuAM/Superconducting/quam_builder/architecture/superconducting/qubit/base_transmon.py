from typing import Dict, Any, Union, Optional, Literal
from dataclasses import field
from logging import getLogger

from quam.core import quam_dataclass
from quam.components.channels import IQChannel, MWChannel, Pulse
from quam import QuamComponent
from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorIQ,
    ReadoutResonatorMW,
)

from qualang_tools.octave_tools import octave_calibration_tool
from qm import QuantumMachine, logger
from qm.qua import (
    save,
    declare,
    fixed,
    assign,
    wait,
    while_,
    StreamType,
    if_,
    update_frequency,
    QuaVariableType,
    Math,
    Cast,
)


__all__ = ["BaseTransmon"]


@quam_dataclass
class BaseTransmon(QuamComponent):
    """
    Example QuAM component for a transmon qubit.

    Attributes:
        id (Union[int, str]): The id of the Transmon, used to generate the name.
            Can be a string, or an integer in which case it will add `Channel._default_label`.
        xy (Union[MWChannel, IQChannel]): The xy drive component.
        resonator (Union[ReadoutResonatorIQ, ReadoutResonatorMW]): The readout resonator component.
        f_01 (float): The 0-1 transition frequency in Hz.
        f_12 (float): The 1-2 transition frequency in Hz.
        anharmonicity (int): The transmon anharmonicity in Hz. Default is 150e6.
        T1 (float): The transmon T1 in seconds. Default is 10e-6.
        T2ramsey (float): The transmon T2* in seconds.
        T2echo (float): The transmon T2 in seconds.
        thermalization_time_factor (int): Thermalization time in units of T1. Default is 5.
        sigma_time_factor (int): Sigma time factor for pulse shaping. Default is 5.
        GEF_frequency_shift (int): The frequency shift for the GEF states. Default is 10.
        chi (float): The dispersive shift in Hz. Default is 0.0.
        grid_location (str): Qubit location in the plot grid as "(column, row)".
        extras (Dict[str, Any]): Additional attributes for the transmon.

    Methods:
        name: Returns the name of the transmon.
        inferred_f_12: Returns the 0-2 (e-f) transition frequency in Hz, derived from f_01 and anharmonicity.
        inferred_anharmonicity: Returns the transmon anharmonicity in Hz, derived from f_01 and f_12.
        sigma: Returns the sigma value for a given pulse.
        thermalization_time: Returns the transmon thermalization time in ns.
        calibrate_octave: Calibrates the Octave channels (xy and resonator) linked to this transmon.
        set_gate_shape: Sets the shape of the single qubit gates.
        readout_state: Performs a readout of the qubit state using the specified pulse.
        reset_qubit: Reset the qubit to the ground state ('g') with the specified method.
        reset_qubit_thermal: Reset the qubit to the ground state ('g') using thermalization.
        reset_qubit_active: Reset the qubit to the ground state ('g') using active reset.
        reset_qubit_active_gef: Reset the qubit to the ground state ('g') using active reset with GEF state readout.
        readout_state_gef: Perform a GEF state readout using the specified pulse and update the state variable.
    """

    id: Union[int, str]

    xy: Union[MWChannel, IQChannel] = None
    resonator: Union[ReadoutResonatorIQ, ReadoutResonatorMW] = None

    f_01: float = None
    f_12: float = None
    anharmonicity: float = None

    T1: float = None
    T2ramsey: float = None
    T2echo: float = None
    thermalization_time_factor: int = 5
    sigma_time_factor: int = 5

    GEF_frequency_shift: int = None
    chi: float = None
    grid_location: str = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @property
    def name(self):
        """The name of the transmon"""
        return self.id if isinstance(self.id, str) else f"q{self.id}"

    def __matmul__(self, other):
        if not isinstance(other, BaseTransmon):
            raise ValueError(
                "Cannot create a qubit pair (q1 @ q2) with a non-qubit object, "
                f"where q1={self} and q2={other}"
            )

        if self is other:
            raise ValueError(
                "Cannot create a qubit pair with same qubit (q1 @ q1), where q1={self}"
            )

        for qubit_pair in self._root.qubit_pairs.values():
            if qubit_pair.qubit_control is self and qubit_pair.qubit_target is other:
                return qubit_pair
        else:
            raise ValueError(
                "Qubit pair not found: qubit_control={self.name}, "
                "qubit_target={other.name}"
            )

    @property
    def inferred_f_12(self) -> float:
        """The 0-2 (e-f) transition frequency in Hz, derived from f_01 and anharmonicity"""
        name = getattr(self, "name", self.__class__.__name__)
        if not isinstance(self.f_01, (float, int)):
            raise AttributeError(
                f"Error inferring f_12 for channel {name}: {self.f_01=} is not a number"
            )
        if not isinstance(self.anharmonicity, (float, int)):
            raise AttributeError(
                f"Error inferring f_12 for channel {name}: {self.anharmonicity=} is not a number"
            )
        return self.f_01 + self.anharmonicity

    @property
    def inferred_anharmonicity(self) -> float:
        """The transmon anharmonicity in Hz, derived from f_01 and f_12."""
        name = getattr(self, "name", self.__class__.__name__)
        if not isinstance(self.f_01, (float, int)):
            raise AttributeError(
                f"Error inferring anharmonicity for channel {name}: {self.f_01=} is not a number"
            )
        if not isinstance(self.f_12, (float, int)):
            raise AttributeError(
                f"Error inferring anharmonicity for channel {name}: {self.f_12=} is not a number"
            )
        return self.f_12 - self.f_01

    def sigma(self, operation: Pulse):
        # todo: check if really needed
        return operation.length / self.sigma_time_factor

    @property
    def thermalization_time(self):
        """The transmon thermalization time in ns."""
        if self.T1 is not None:
            return int(self.thermalization_time_factor * self.T1 * 1e9 / 4) * 4
        else:
            return int(self.thermalization_time_factor * 10e-6 * 1e9 / 4) * 4

    # todo: do we really want this one here?
    def calibrate_octave(
        self,
        QM: QuantumMachine,
        calibrate_drive: bool = True,
        calibrate_resonator: bool = True,
    ) -> None:
        """Calibrate the Octave channels (xy and resonator) linked to this transmon for the LO frequency, intermediate
        frequency and Octave gain as defined in the state.

        Args:
            QM (QuantumMachine): the running quantum machine.
            calibrate_drive (bool): flag to calibrate xy line.
            calibrate_resonator (bool): flag to calibrate the resonator line.
        """
        if calibrate_resonator and self.resonator is not None:
            if hasattr(self.resonator, "frequency_converter_up"):
                logger.info(f"Calibrating {self.resonator.name}")
                octave_calibration_tool(
                    QM,
                    self.resonator.name,
                    lo_frequencies=self.resonator.frequency_converter_up.LO_frequency,
                    intermediate_frequencies=self.resonator.intermediate_frequency,
                )
            else:
                raise RuntimeError(
                    f"{self.resonator.name} doesn't have a 'frequency_converter_up' attribute, it is thus most likely not connected to an Octave."
                )
        if calibrate_drive and self.xy is not None:
            if hasattr(self.xy, "frequency_converter_up"):
                logger.info(f"Calibrating {self.xy.name}")
                octave_calibration_tool(
                    QM,
                    self.xy.name,
                    lo_frequencies=self.xy.frequency_converter_up.LO_frequency,
                    intermediate_frequencies=self.xy.intermediate_frequency,
                )
            else:
                raise RuntimeError(
                    f"{self.xy.name} doesn't have a 'frequency_converter_up' attribute, it is thus most likely not connected to an Octave."
                )

    def set_gate_shape(self, gate_shape: str) -> None:
        """Set the shape fo the single qubit gates defined as ["x180", "x90" "-x90", "y180", "y90", "-y90"]"""
        for gate in ["x180", "x90", "-x90", "y180", "y90", "-y90"]:
            if f"{gate}_{gate_shape}" in self.xy.operations:
                self.xy.operations[gate] = f"#./{gate}_{gate_shape}"
            else:
                raise AttributeError(
                    f"The gate '{gate}_{gate_shape}' is not part of the existing operations for {self.xy.name} --> {self.xy.operations.keys()}."
                )

    def readout_state(
        self, state, pulse_name: str = "readout", threshold: float = None
    ):
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

    def reset_qubit(
        self,
        reset_type: Literal["thermal", "active", "active_gef"] = "thermal",
        simulate: bool = False,
        logger=None,
        **kwargs,
    ):
        """
        Reset the qubit with the specified method.

        This function resets the qubit using the specified method: thermal reset, active reset, or active GEF reset.
        When simulating the QUA program, the qubit reset is skipped to save simulated samples.

        Args:
            reset_type (Literal["thermal", "active", "active_gef"]): The type of reset to perform. Default is "thermal".
            simulate (bool): If True, the qubit reset is skipped for simulation purposes. Default is False.
            logger (optional): Logger instance to log warnings. If None, a default logger is used.
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
            if logger is None:
                logger = getLogger(__name__)
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

    def readout_state_gef(self, state: QuaVariableType, pulse_name: str = "readout"):
        """
        Perform a GEF state readout using the specified pulse and update the state variable.

        This function measures the 'I' and 'Q' quadrature components of the resonator's response
        to a given pulse, calculates the squared Euclidean distance between the measured
        (I, Q) values and the predefined GEF state centers, and assigns the state variable
        to the index of the closest GEF state.

        Args:
            state (QuaVariableType): The variable to store the readout state (0 for 'g', 1 for 'e', 2 for 'f').
            pulse_name (str, optional): The name of the pulse to use for the readout. Defaults to "readout".

        Returns:
            None
        """
        I = declare(fixed)
        Q = declare(fixed)
        diff = declare(fixed, size=3)

        self.resonator.update_frequency(
            self.resonator.intermediate_frequency + self.resonator.GEF_frequency_shift
        )
        self.resonator.measure(pulse_name, qua_vars=(I, Q))
        self.resonator.update_frequency(self.resonator.intermediate_frequency)

        gef_centers = [
            self.resonator.gef_centers[0],
            self.resonator.gef_centers[1],
            self.resonator.gef_centers[2],
        ]
        for p in range(3):
            assign(
                diff[p],
                (I - gef_centers[p][0]) * (I - gef_centers[p][0])
                + (Q - gef_centers[p][1]) * (Q - gef_centers[p][1]),
            )
        assign(state, Math.argmin(diff))
        wait(self.resonator.depletion_time // 4, self.resonator.name)
