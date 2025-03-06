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
import warnings
from typing import Dict, Any, Union, Optional, Literal
from dataclasses import field

__all__ = ["BaseTransmon"]


@quam_dataclass
class BaseTransmon(QuamComponent):
    """
    Example QuAM component for a transmon qubit.

    Args:
        id (str, int): The id of the Transmon, used to generate the name.
            Can be a string, or an integer in which case it will add`Channel._default_label`.
        xy (IQChannel): The xy drive component.
        resonator (ReadoutResonator): The readout resonator component.
        T1 (float): The transmon T1 in s.
        T2ramsey (float): The transmon T2* in s.
        T2echo (float): The transmon T2 in s.
        thermalization_time_factor (int): thermalization time in units of T1.
        anharmonicity (int, float): the transmon anharmonicity in Hz.
        sigma_time_factor:
        GEF_frequency_shift (int):
        chi (float):
        grid_location (str): qubit location in the plot grid as "(column, row)"
    """

    id: Union[int, str]

    xy: Union[MWChannel, IQChannel] = None
    resonator: Union[ReadoutResonatorIQ, ReadoutResonatorMW] = None

    f_01: float = None
    f_12: float = None
    anharmonicity: int = 150e6

    T1: float = 10e-6
    T2ramsey: float = None
    T2echo: float = None
    thermalization_time_factor: int = 5
    sigma_time_factor: int = 5

    GEF_frequency_shift: int = 10
    chi: float = 0.0
    grid_location: str = None
    extras: Dict[str, Any] = field(default_factory=dict)

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
        if not isinstance(other, BaseTransmon):
            raise ValueError(
                "Cannot create a qubit pair (q1 @ q2) with a non-qubit object, " f"where q1={self} and q2={other}"
            )

        if self is other:
            raise ValueError("Cannot create a qubit pair with same qubit (q1 @ q1), where q1={self}")

        for qubit_pair in self._root.qubit_pairs.values():
            if qubit_pair.qubit_control is self and qubit_pair.qubit_target is other:
                return qubit_pair
        else:
            raise ValueError("Qubit pair not found: qubit_control={self.name}, " "qubit_target={other.name}")

    def readout_state(
        self, state, pulse_name: str = "readout", threshold: float = None, save_qua_var: StreamType = None
    ):
        I = declare(fixed)
        Q = declare(fixed)
        if threshold is None:
            threshold = self.resonator.operations[pulse_name].threshold
        self.resonator.measure(pulse_name, qua_vars=(I, Q))
        assign(state, Cast.to_int(I > threshold))
        wait(self.resonator.depletion_time // 4, self.resonator.name)

    def reset_qubit(self, reset_type: Literal["thermal", "active", "active_gef"]="thermal", simulate: bool=False, **kwargs):
        """
        todo: update the docstring
        Reset the qubit with the specified method based on the node parameters.

        This function resets the qubit using the method specified in the node parameters.
        It supports thermal reset, active reset, and active GEF reset. When simulating the
        QUA program, the qubit reset is skipped to save simulated samples.

        Args:
            node_parameters (Union[QubitsExperimentNodeParameters, CommonNodeParameters]):
                The parameters defining the qubit reset method and simulation mode.
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
            warnings.warn("For simulating the QUA program, the qubit reset has been skipped.")

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
                assign(success, success + 1)  # we need to measure 'g' two times in a row to increase our confidence
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

        self.resonator.update_frequency(self.resonator.intermediate_frequency - self.resonator.GEF_frequency_shift)
        self.resonator.measure(pulse_name, qua_vars=(I, Q))
        self.resonator.update_frequency(self.resonator.intermediate_frequency)

        gef_centers = [self.resonator.gef_centers.g, self.resonator.gef_centers.e, self.resonator.gef_centers.f]
        for p in range(3):
            assign(
                diff[p],
                (I - gef_centers[p][0]) * (I - gef_centers[p][0]) + (Q - gef_centers[p][1]) * (Q - gef_centers[p][1]),
            )
        assign(state, Math.argmin(diff))
        wait(self.resonator.depletion_time // 4, self.resonator.name)
