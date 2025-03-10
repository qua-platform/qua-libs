from quam.components import pulses
from qualang_tools.units import unit
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon, FluxTunableTransmon
from quam_builder.architecture.superconducting.qubit_pair import FixedFrequencyTransmonPair, FluxTunableTransmonPair
import numpy as np
from typing import Union

# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)


def add_DragGaussian_pulses(
    transmon: Union[FixedFrequencyTransmon, FluxTunableTransmon],
    amplitude: float,
    length: int,
    sigma: int,
    alpha: float,
    detuning: float,
    anharmonicity: float = None,
    subtracted: bool = True,
    digital_marker: str = None,
):
    """
    Adds a set of DragGaussian pulses to a transmon qubit and sets the gate shape as 'DragGaussian'.
    The function will create the 6 operations corresponding to the set of single qubit gates:
    ["x180_DragGaussian", "x90_DragGaussian", "-x90_DragGaussian", "y180_DragGaussian", "y90_DragGaussian", "-y90_DragGaussian"].

    The specified parameters are the parameters for x180 and will be propagated to the other gates, except for the
    amplitude which is halved for the "x90", "-x90", "y90" and "-y90" gates.

    Parameters:
    transmon (Union[FixedFrequencyTransmon, FluxTunableTransmon]): The transmon qubit to which the pulses will be added.
    amplitude (float): The amplitude of the pulses in Volts.
    length (int): The length of the pulses in ns. Must be above 16ns and a multiple of 4ns.
    sigma (int): The gaussian standard deviation.
    alpha (float): The alpha parameter (DRAG coefficient) for the DragGaussian pulses.
    detuning (float): The detuning frequency for the pulses in Hz.
    anharmonicity (float): The anharmonicity of the qubit in Hz. Defaults to None which means that the anharmonicity is set to `transmon.anharmonicity`.
    digital_marker (str, optional): The digital marker for the pulses. Defaults to None. Can be set to "ON".
    subtracted (bool):  If true, returns a subtracted Gaussian, such that the first and last points will be at 0 volts.
    This reduces high-frequency components due to the initial and final points offset. Default is true.
    """
    if transmon.xy is not None:
        if anharmonicity is None:
            anharmonicity = f"#/qubits/{transmon.name}/anharmonicity"
        transmon.xy.operations["x180_DragGaussian"] = pulses.DragGaussianPulse(
            length=length,
            sigma=sigma,
            amplitude=amplitude,
            alpha=alpha,
            anharmonicity=anharmonicity,
            detuning=detuning,
            digital_marker=digital_marker,
            subtracted=subtracted,
            axis_angle=0,
        )
        transmon.xy.operations["x90_DragGaussian"] = pulses.DragGaussianPulse(
            length="#../x180_DragGaussian/length",
            sigma="#../x180_DragGaussian/sigma",
            amplitude=amplitude / 2,
            alpha="#../x180_DragGaussian/alpha",
            anharmonicity="#../x180_DragGaussian/anharmonicity",
            detuning="#../x180_DragGaussian/detuning",
            digital_marker="#../x180_DragGaussian/digital_marker",
            subtracted="#../x180_DragGaussian/subtracted",
            axis_angle=0,
        )
        transmon.xy.operations["-x90_DragGaussian"] = pulses.DragGaussianPulse(
            length="#../x180_DragGaussian/length",
            sigma="#../x180_DragGaussian/sigma",
            amplitude="#../x90_DragGaussian/amplitude",
            alpha="#../x180_DragGaussian/alpha",
            anharmonicity="#../x180_DragGaussian/anharmonicity",
            detuning="#../x180_DragGaussian/detuning",
            digital_marker="#../x180_DragGaussian/digital_marker",
            subtracted="#../x180_DragGaussian/subtracted",
            axis_angle=np.pi,
        )
        transmon.xy.operations["y180_DragGaussian"] = pulses.DragGaussianPulse(
            length="#../x180_DragGaussian/length",
            sigma="#../x180_DragGaussian/sigma",
            amplitude="#../x180_DragGaussian/amplitude",
            alpha="#../x180_DragGaussian/alpha",
            anharmonicity="#../x180_DragGaussian/anharmonicity",
            detuning="#../x180_DragGaussian/detuning",
            digital_marker="#../x180_DragGaussian/digital_marker",
            subtracted="#../x180_DragGaussian/subtracted",
            axis_angle=np.pi / 2,
        )
        transmon.xy.operations["y90_DragGaussian"] = pulses.DragGaussianPulse(
            length="#../x180_DragGaussian/length",
            sigma="#../x180_DragGaussian/sigma",
            amplitude="#../x90_DragGaussian/amplitude",
            alpha="#../x90_DragGaussian/alpha",
            anharmonicity="#../x180_DragGaussian/anharmonicity",
            detuning="#../x90_DragGaussian/detuning",
            digital_marker="#../x180_DragGaussian/digital_marker",
            subtracted="#../x180_DragGaussian/subtracted",
            axis_angle=np.pi / 2,
        )
        transmon.xy.operations["-y90_DragGaussian"] = pulses.DragGaussianPulse(
            length="#../x180_DragGaussian/length",
            sigma="#../x180_DragGaussian/sigma",
            amplitude="#../x90_DragGaussian/amplitude",
            alpha="#../x90_DragGaussian/alpha",
            anharmonicity="#../x180_DragGaussian/anharmonicity",
            detuning="#../x90_DragGaussian/detuning",
            digital_marker="#../x180_DragGaussian/digital_marker",
            subtracted="#../x180_DragGaussian/subtracted",
            axis_angle=-np.pi / 2,
        )
        transmon.set_gate_shape("DragGaussian")


def add_DragCosine_pulses(
    transmon: Union[FixedFrequencyTransmon, FluxTunableTransmon],
    amplitude: float,
    length: int,
    alpha: float,
    detuning: float,
    anharmonicity: float = None,
    digital_marker: str = None,
):
    """
    Adds a set of DragCosine pulses to a transmon qubit and sets the gate shape as 'DragCosine'.
    The function will create the 6 operations corresponding to the set of single qubit gates:
    ["x180_DragCosine", "x90_DragCosine", "-x90_DragCosine", "y180_DragCosine", "y90_DragCosine", "-y90_DragCosine"].

    The specified parameters are the parameters for x180 and will be propagated to the other gates, except for the
    amplitude which is halved for the "x90", "-x90", "y90" and "-y90" gates.

    Parameters:
    transmon (Union[FixedFrequencyTransmon, FluxTunableTransmon]): The transmon qubit to which the pulses will be added.
    amplitude (float): The amplitude of the pulses in Volts.
    length (int): The length of the pulses in ns. Must be above 16ns and a multiple of 4ns.
    alpha (float): The alpha parameter (DRAG coefficient) for the DragCosine pulses.
    detuning (float): The detuning frequency for the pulses in Hz.
    anharmonicity (float): The anharmonicity of the qubit in Hz. Defaults to None which means that the anharmonicity is set to `transmon.anharmonicity`.
    digital_marker (str, optional): The digital marker for the pulses. Defaults to None. Can be set to "ON".
    """
    if transmon.xy is not None:
        if anharmonicity is None:
            anharmonicity = f"#/qubits/{transmon.name}/anharmonicity"
        transmon.xy.operations["x180_DragCosine"] = pulses.DragCosinePulse(
            length=length,
            amplitude=amplitude,
            alpha=alpha,
            anharmonicity=anharmonicity,
            detuning=detuning,
            digital_marker=digital_marker,
            axis_angle=0,
        )
        transmon.xy.operations["x90_DragCosine"] = pulses.DragCosinePulse(
            length="#../x180_DragCosine/length",
            amplitude=amplitude / 2,
            alpha="#../x180_DragCosine/alpha",
            anharmonicity="#../x180_DragCosine/anharmonicity",
            detuning="#../x180_DragCosine/detuning",
            digital_marker="#../x180_DragCosine/digital_marker",
            axis_angle=0,
        )
        transmon.xy.operations["-x90_DragCosine"] = pulses.DragCosinePulse(
            length="#../x180_DragCosine/length",
            amplitude="#../x90_DragCosine/amplitude",
            alpha="#../x180_DragCosine/alpha",
            anharmonicity="#../x180_DragCosine/anharmonicity",
            detuning="#../x180_DragCosine/detuning",
            digital_marker="#../x180_DragCosine/digital_marker",
            axis_angle=np.pi,
        )
        transmon.xy.operations["y180_DragCosine"] = pulses.DragCosinePulse(
            length="#../x180_DragCosine/length",
            amplitude="#../x180_DragCosine/amplitude",
            alpha="#../x180_DragCosine/alpha",
            anharmonicity="#../x180_DragCosine/anharmonicity",
            detuning="#../x180_DragCosine/detuning",
            digital_marker="#../x180_DragCosine/digital_marker",
            axis_angle=np.pi / 2,
        )
        transmon.xy.operations["y90_DragCosine"] = pulses.DragCosinePulse(
            length="#../x180_DragCosine/length",
            amplitude="#../x90_DragCosine/amplitude",
            alpha="#../x90_DragCosine/alpha",
            anharmonicity="#../x180_DragCosine/anharmonicity",
            detuning="#../x90_DragCosine/detuning",
            digital_marker="#../x180_DragCosine/digital_marker",
            axis_angle=np.pi / 2,
        )
        transmon.xy.operations["-y90_DragCosine"] = pulses.DragCosinePulse(
            length="#../x180_DragCosine/length",
            amplitude="#../x90_DragCosine/amplitude",
            alpha="#../x90_DragCosine/alpha",
            anharmonicity="#../x180_DragCosine/anharmonicity",
            detuning="#../x90_DragCosine/detuning",
            digital_marker="#../x180_DragCosine/digital_marker",
            axis_angle=-np.pi / 2,
        )
        transmon.set_gate_shape("DragCosine")


def add_Square_pulses(
    transmon: Union[FixedFrequencyTransmon, FluxTunableTransmon],
    amplitude: float,
    length: int,
    digital_marker: str = None,
):
    """
    Adds a set of Square pulses to a transmon qubit and sets the gate shape as 'Square'.
    The function will create the 6 operations corresponding to the set of single qubit gates:
    ["x180_Square", "x90_Square", "-x90_Square", "y180_Square", "y90_Square", "-y90_Square"].

    The specified parameters are the parameters for x180 and will be propagated to the other gates, except for the
    amplitude which is halved for the "x90", "-x90", "y90" and "-y90" gates.

    Parameters:
    transmon (Union[FixedFrequencyTransmon, FluxTunableTransmon]): The transmon qubit to which the pulses will be added.
    amplitude (float): The amplitude of the pulses in Volts.
    length (int): The length of the pulses in ns. Must be above 16ns and a multiple of 4ns.
    digital_marker (str, optional): The digital marker for the pulses. Defaults to None. Can be set to "ON".
    """
    if transmon.xy is not None:
        transmon.xy.operations["x180_Square"] = pulses.SquarePulse(
            length=length,
            amplitude=amplitude,
            digital_marker=digital_marker,
            axis_angle=0,
        )
        transmon.xy.operations["x90_Square"] = pulses.SquarePulse(
            length="#../x180_Square/length",
            amplitude=amplitude / 2,
            digital_marker="#../x180_Square/digital_marker",
            axis_angle=0,
        )
        transmon.xy.operations["-x90_Square"] = pulses.SquarePulse(
            length="#../x180_Square/length",
            amplitude="#../x90_Square/amplitude",
            digital_marker="#../x180_Square/digital_marker",
            axis_angle=np.pi,
        )
        transmon.xy.operations["y180_Square"] = pulses.SquarePulse(
            length="#../x180_Square/length",
            amplitude="#../x180_Square/amplitude",
            digital_marker="#../x180_Square/digital_marker",
            axis_angle=np.pi / 2,
        )
        transmon.xy.operations["y90_Square"] = pulses.SquarePulse(
            length="#../x180_Square/length",
            amplitude="#../x90_Square/amplitude",
            digital_marker="#../x180_Square/digital_marker",
            axis_angle=np.pi / 2,
        )
        transmon.xy.operations["-y90_Square"] = pulses.SquarePulse(
            length="#../x180_Square/length",
            amplitude="#../x90_Square/amplitude",
            digital_marker="#../x180_Square/digital_marker",
            axis_angle=-np.pi / 2,
        )
        transmon.set_gate_shape("Square")


def add_default_transmon_pulses(transmon: Union[FixedFrequencyTransmon, FluxTunableTransmon]):
    """
    Adds default pulses to a transmon qubit:
        * transmon.xy.operations["saturation"] = pulses.SquarePulse(amplitude=0.25, length=20 * u.us, axis_angle=0)
        * transmon.z.operations["const"] = pulses.SquarePulse(amplitude=0.1, length=100)
        * transmon.resonator.operations["readout"] = pulses.SquareReadoutPulse(length=2000, amplitude=0.01, threshold=0.0, digital_marker="ON")

    Parameters:
    transmon (Union[FixedFrequencyTransmon, FluxTunableTransmon]): The transmon qubit to which the pulses will be added.
    """
    if hasattr(transmon, "xy"):
        if transmon.xy is not None:
            transmon.xy.operations["saturation"] = pulses.SquarePulse(amplitude=0.25, length=20 * u.us, axis_angle=0)

    if hasattr(transmon, "z"):
        if transmon.z is not None:
            transmon.z.operations["const"] = pulses.SquarePulse(amplitude=0.1, length=100)

    if hasattr(transmon, "resonator"):
        if transmon.resonator is not None:
            transmon.resonator.operations["readout"] = pulses.SquareReadoutPulse(
                length=2000, amplitude=0.01, threshold=0.0, digital_marker="ON"
            )


def add_default_transmon_pair_pulses(transmon_pair: Union[FixedFrequencyTransmonPair, FluxTunableTransmonPair]):
    """
    Adds default pulses to a transmon qubit pair depending on its attributes:
        * transmon_pair.coupler.operations["const"] = pulses.SquarePulse(amplitude=0.1, length=100)
        * transmon_pair.cross_resonance.operations["square"] = pulses.SquarePulse(amplitude=0.1, length=100)
        * transmon_pair.zz_drive.operations["square"] = pulses.SquarePulse(amplitude=0.1, length=100)

    Parameters:
    transmon_pair (Union[FixedFrequencyTransmonPair, FluxTunableTransmonPair]): The transmon qubit pair to which the pulses will be added.
    """
    if hasattr(transmon_pair, "coupler"):
        if transmon_pair.coupler is not None:
            transmon_pair.coupler.operations["const"] = pulses.SquarePulse(amplitude=0.1, length=100)
    if hasattr(transmon_pair, "cross_resonance"):
        if transmon_pair.cross_resonance is not None:
            transmon_pair.cross_resonance.operations["square"] = pulses.SquarePulse(amplitude=0.1, length=100)
    if hasattr(transmon_pair, "zz_drive"):
        if transmon_pair.zz_drive is not None:
            transmon_pair.zz_drive.operations["square"] = pulses.SquarePulse(amplitude=0.1, length=100)
