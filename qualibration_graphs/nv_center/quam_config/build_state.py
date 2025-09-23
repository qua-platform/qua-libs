#%%
from quam_config import Quam
from quam_config.components.xy import XYDriveMW
from quam_config.components.spcm import SPCM_LF
from quam_config.components.laser import LaserAnalog
from quam.components.ports import MWFEMAnalogOutputPort, LFFEMAnalogInputPort, LFFEMAnalogOutputPort, FEMDigitalOutputPort

import numpy as np
import json
from qualang_tools.units import unit

from quam.components.pulses import GaussianPulse, SquarePulse, SquareReadoutPulse
from quam.components.channels import MWChannel, SingleChannel, InOutSingleChannel, TimeTaggingAddon
from quam.core import quam_dataclass
from typing import Dict, List, Union, ClassVar, Optional
from dataclasses import field
from quam_config.qubit.baseNV import BaseNV

#%% Building the machine configuration
# Create a new Quam machine configuration
# This is a simple configuration with one qubit and its associated components
# The qubit is a nv qubit with XY drive, SPCM, and laser
machine = Quam()
num_qubits = 1
for idx in range(num_qubits):
    nv = BaseNV(id=idx)
    machine.qubits[nv.name] = nv
#%%
# Define XY drive for qubit 0
machine.qubits['q0'].xy = XYDriveMW(
    operations={
        "cw": SquarePulse(amplitude=0.1, length=100),
        "x180": SquarePulse(amplitude=0.1, length=100),
        # "x180": GaussianPulse(amplitude=1, length=100, sigma=6),
        # "x90": GaussianPulse(amplitude=0.5, length=100, sigma=6),
    },
    opx_output=MWFEMAnalogOutputPort(
        controller_id="con1", fem_id=2, port_id=2, band=1, upconverter_frequency=int(3e9), full_scale_power_dbm=-11
    ),
    # intermediate_frequency=50e6,
    upconverter=1,
)

machine.qubits['q0'].spcm = SPCM_LF(
    operations={
        "readout": SquareReadoutPulse(amplitude=0.0, length=100),
    },
    opx_input=LFFEMAnalogInputPort(
        controller_id="con1", fem_id=1, port_id=1
    ),
    opx_input_offset=None,
    time_of_flight=28,
    smearing=0,
    time_tagging=TimeTaggingAddon(
        signal_threshold=0.195,  # in units of V
        signal_polarity="below",
        derivative_threshold=0.073,  # in units of V/ns
        derivative_polarity="below",
    ),
    opx_output=LFFEMAnalogOutputPort(
        controller_id="con1", fem_id=1, port_id=1
    )
)  
machine.qubits['q0'].laser = LaserAnalog(
    operations={
        "laser_on": SquarePulse(amplitude=0.4, length=1000),
        "laser_off": SquarePulse(amplitude=0, length=1000),
    },
    opx_output=LFFEMAnalogOutputPort(
        controller_id="con1", fem_id=1, port_id=2
    ),
    opx_output_offset=0,
    intermediate_frequency=None,
)  

#%% Network configuration
# Define the network configuration for the machine
# This includes the host IP, port, and cluster name
# This is necessary for the machine to communicate with the OPX and other components
machine.network = {
    "host": "192.0.0.1",
    "port": 8000,
    "cluster_name": "QB_Cluster",
}

machine.active_qubit_names = ["q0"]


# Give the qubit parameters###################################################################
u = unit(coerce_to_integer=True)


def get_band(freq):
    """Determine the MW fem DAC band corresponding to a given frequency.

    Args:
        freq (float): The frequency in Hz.

    Returns:
        int: The Nyquist band number.
            - 1 if 50 MHz <= freq < 5.5 GHz
            - 2 if 4.5 GHz <= freq < 7.5 GHz
            - 3 if 6.5 GHz <= freq <= 10.5 GHz

    Raises:
        ValueError: If the frequency is outside the MW fem bandwidth [50 MHz, 10.5 GHz].
    """
    if 50e6 <= freq < 5.5e9:
        return 1
    elif 4.5e9 <= freq < 7.5e9:
        return 2
    elif 6.5e9 <= freq <= 10.5e9:
        return 3
    else:
        raise ValueError(f"The specified frequency {freq} Hz is outside of the MW fem bandwidth [50 MHz, 10.5 GHz]")

#%%
def closest_number(lst, target):
    return min(lst, key=lambda x: abs(x - target))


def get_full_scale_power_dBm_and_amplitude(desired_power: float, max_amplitude: float = 0.5) -> tuple[int, float]:
    """Get the full_scale_power_dbm and waveform amplitude for the MW FEM to output the specified desired power.

    The keyword `full_scale_power_dbm` is the maximum power of normalized pulse waveforms in [-1,1].
    To convert to voltage:
        power_mw = 10**(full_scale_power_dbm / 10)
        max_voltage_amp = np.sqrt(2 * power_mw * 50 / 1000)
        amp_in_volts = waveform * max_voltage_amp
        ^ equivalent to OPX+ amp
    Its range is -11dBm to +16dBm with 3dBm steps.

    Args:
        desired_power (float): Desired output power in dBm.
        max_amplitude (float, optional): Maximum allowed waveform amplitude in V. Default is 0.5V.

    Returns:
        tuple[float, float]: The full_scale_power_dBm and waveform amplitude realizing the desired power.
    """
    allowed_powers = [-11, -8, -5, -2, 1, 4, 7, 10, 13, 16]
    resulting_power = desired_power - 20 * np.log10(max_amplitude)
    if resulting_power < 0:
        full_scale_power_dBm = closest_number(allowed_powers, max(resulting_power + 3, -11))
    else:
        full_scale_power_dBm = closest_number(allowed_powers, min(resulting_power + 3, 16))
    amplitude = 10 ** ((desired_power - full_scale_power_dBm) / 20)
    if -11 <= full_scale_power_dBm <= 16 and -1 <= amplitude <= 1:
        return full_scale_power_dBm, amplitude
    else:
        raise ValueError(
            f"The desired power is outside the specifications ([-11; +16]dBm, [-1; +1]), got ({full_scale_power_dBm}; {amplitude})"
        )
# %%                                    Qubit parameters
########################################################################################################################
# The keyword "band" refers to the following frequency bands:
#   1: (50 MHz - 5.5 GHz)
#   2: (4.5 GHz - 7.5 GHz)
#   3: (6.5 GHz - 10.5 GHz)
# Note that the "coupled" ports O1 & I1, O2 & O3, O4 & O5, O6 & O7, and O8 & I2 must be in the same band.

# Qubit drive frequencies
xy_freq = np.array([2.87, 2.81]) * u.GHz
xy_LO = np.array([3.2, 3.1]) * u.GHz
xy_if = xy_freq - xy_LO  # The intermediate frequency is inferred from the LO and qubit frequencies
assert np.all(np.abs(xy_if) < 400 * u.MHz), (
    "The xy intermediate frequency must be within [-400; 400] MHz. \n"
    f"Qubit drive frequencies: {xy_freq} \n"
    f"Qubit drive LO frequencies: {xy_LO} \n"
    f"Qubit drive IF frequencies: {xy_if} \n"
)

# Desired output power in dBm
drive_power = -10
# Get the full_scale_power_dBm and waveform amplitude corresponding to the desired powers
xy_full_scale, xy_amplitude = get_full_scale_power_dBm_and_amplitude(drive_power)

# Update qubit xy freq and power
for k, qubit in enumerate(machine.qubits.values()):
    qubit.f_01 = xy_freq.tolist()[k]  # Qubit 0 to 1 (|g> -> |e>) transition frequency
    qubit.xy.RF_frequency = None
    qubit.xy.RF_frequency = qubit.f_01  # Qubit drive frequency
    qubit.xy.opx_output.full_scale_power_dbm = xy_full_scale  # Max drive power in dBm
    qubit.xy.opx_output.upconverter_frequency = xy_LO.tolist()[k]  # Qubit drive up-converter frequency
    qubit.xy.opx_output.band = get_band(xy_LO.tolist()[k])  # Qubit drive band for the up-conversion
    qubit.grid_location = f"{k},0"  # Qubit grid location for plotting as "column,row"



########################################################################################################################
# %%                                    Laser parameters
########################################################################################################################
# The "output_mode" can be used to tailor the max voltage and frequency bandwidth, i.e.,
#   "direct":    1Vpp (-0.5V to 0.5V), 750MHz bandwidth
#   "amplified": 5Vpp (-2.5V to 2.5V), 350MHz bandwidth
# At 1 GS/s, use the "upsampling_mode" to optimize output for
#   modulated pulses (optimized for modulated pulses): "mw"
#   baseband pulses (optimized for clean step response): "pulse"

# Update flux channels
for k, qubit in enumerate(machine.qubits.values()):
    if hasattr(qubit, "z"):
        qubit.z.opx_output.output_mode = "direct"
        qubit.z.opx_output.upsampling_mode = "pulse"




#%%
with open("qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)
machine.save("/home/max/projects/gitlab/qua-libs-qb-fork/qualibration_graphs/nv_center/quam_config/state.json")

#%%



#%%


# for k, q in enumerate(machine.qubits):
#     # readout
#     machine.qubits[q].readout.operations["readout"].length = 2.5 * u.us



