from pathlib import Path
from scipy.signal.windows import gaussian
from qualang_tools.units import unit
from typing import List, Any, Dict
from qm.qua._dsl import _ResultSource, _Variable, _Expression
from qm.qua import declare, assign, play, fixed, Cast, amp, wait, ramp

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

######################
# Network parameters #
######################
qop_ip = "172.16.33.101"  # Write the QM router IP address
cluster_name = "Cluster_83"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

# Path to save data
octave_config = None


#############################################
#              OPX PARAMETERS               #
#############################################
class OPX_background_sequence:
    def __init__(self, configuration: dict, elements: list):
        self._elements = elements
        self._config = configuration
        self.current_level = [0.0 for _ in self._elements]
        self._realtime = False
        self._voltage_points = {}
        self.average_power = [0 for _ in self._elements]
        for el in self._elements:
            self._config["elements"][el]["operations"]["step"] = "step_pulse"
        self._config["pulses"]["step_pulse"] = {
            "operation": "control",
            "length": 16,
            "waveforms": {"single": "step_wf"},
        }
        self._config["waveforms"]["step_wf"] = {"type": "constant", "sample": 0.25}

    def _check_name(self, name, key):
        if name in key:
            return self._check_name(name + "%", key)
        else:
            return name

    def _add_op_to_config(self, el, name, amplitude, length):
        op_name = self._check_name(name, self._config["elements"][el]["operations"])
        pulse_name = self._check_name(f"{el}_{op_name}_pulse", self._config["pulses"])
        wf_name = self._check_name(f"{el}_{op_name}_wf", self._config["waveforms"])
        self._config["elements"][el]["operations"][op_name] = pulse_name
        self._config["pulses"][pulse_name] = {
            "operation": "control",
            "length": length,
            "waveforms": {"single": wf_name},
        }
        self._config["waveforms"][wf_name] = {"type": "constant", "sample": amplitude}
        return op_name

    def add_step(
        self,
        level: list = None,
        duration: int = None,
        voltage_point_name: str = None,
        ramp_duration: int = None,
        current_offset: list = None,
    ):
        """
        If duration is QUA, then >= 32
        """
        if current_offset is None:
            current_offset = [0.0 for _ in self._elements]
        if voltage_point_name is not None:
            if duration is None:
                _duration = self._voltage_points[voltage_point_name]["duration"]
            else:
                _duration = duration

            for i, gate in enumerate(self._elements):
                if level is None:
                    voltage_level = self._voltage_points[voltage_point_name]["coordinates"][i]
                else:
                    voltage_level = level[i]

                if ramp_duration is None:
                    # If real-time amplitude and duration, then split into play and wait otherwise gap, but then duration > 32ns
                    # if (isinstance(voltage_level, (_Variable, _Expression)) and isinstance(_duration, (_Variable, _Expression))) or isinstance(self.current_level[i], (_Variable, _Expression)):
                    if isinstance(voltage_level, (_Variable, _Expression)) or isinstance(
                        self.current_level[i], (_Variable, _Expression)
                    ):
                        #     play("step" * amp((voltage_level - self.current_level[i]) * 4), gate)
                        #     wait((_duration - 16) >> 2, gate)
                        # if isinstance(_duration, (_Variable, _Expression)):

                        if isinstance(voltage_level, (_Variable, _Expression)):
                            expression = declare(fixed)
                            assign(expression, voltage_level)
                            self.average_power[i] += Cast.mul_int_by_fixed(_duration, expression)
                        else:
                            self.average_power[i] += int(voltage_level * _duration)
                        play("step" * amp((voltage_level - self.current_level[i] - current_offset[i]) * 4), gate)
                        wait((_duration - 16) >> 2, gate)
                    # elif isinstance(_duration, (_Variable, _Expression)):
                    elif duration is not None:
                        self.average_power[i] += int(voltage_level * _duration)
                        operation = self._add_op_to_config(
                            gate,
                            voltage_point_name,
                            amplitude=self._voltage_points[voltage_point_name]["coordinates"][i]
                            - self.current_level[i],
                            length=self._voltage_points[voltage_point_name]["duration"],
                        )

                        play(operation, gate, duration=_duration >> 2)
                    else:
                        self.average_power[i] += int(voltage_level * _duration)
                        operation = self._add_op_to_config(
                            gate,
                            voltage_point_name,
                            amplitude=self._voltage_points[voltage_point_name]["coordinates"][i]
                            - self.current_level[i],
                            length=self._voltage_points[voltage_point_name]["duration"],
                        )
                        play(operation, gate)

                else:
                    play(
                        ramp((voltage_level - self.current_level[i]) / ramp_duration), gate, duration=ramp_duration >> 2
                    )
                    wait(_duration >> 2, gate)
                self.current_level[i] = voltage_level

    def add_compensation_pulse(self, duration: int):
        for i, gate in enumerate(self._elements):
            if not isinstance(self.average_power[i], (_Variable, _Expression)):
                compensation_amp = -self.average_power[i] / duration
                operation = self._add_op_to_config(
                    gate, "compensation", amplitude=compensation_amp - self.current_level[i], length=duration
                )
                play(operation, gate)
            else:
                operation = self._add_op_to_config(gate, "compensation", amplitude=0.25, length=duration)
                compensation_amp = declare(fixed)
                test = declare(int)
                assign(test, self.average_power[i])
                assign(compensation_amp, -Cast.mul_fixed_by_int(1 / duration, test))
                play(operation * amp((compensation_amp - self.current_level[i]) * 4), gate)
            self.current_level[i] = compensation_amp

    def wait(self, duration):
        for i, gate in enumerate(self._elements):
            wait(duration >> 2, gate)
            self.current_level[i] = 0

    def add_points(self, name: str, coordinates: list, duration: int):
        self._voltage_points[name] = {}
        self._voltage_points[name]["coordinates"] = coordinates
        self._voltage_points[name]["duration"] = duration


######################
#       READOUT      #
######################
# DC readout parameters
readout_len = 1 * u.us
readout_amp = 0.4
IV_scale_factor = 0.5e-9  # in A/V

# Reflectometry
resonator_IF = 151 * u.MHz
reflectometry_readout_length = 1 * u.us
reflect_amp = 30 * u.mV

# Time of flight
time_of_flight = 24

######################
#      DC GATES      #
######################

## Section defining the points from the charge stability map - can be done in the config
# Relevant points in the charge stability map as ["P1", "P2"] in V
level_init = [0.1, -0.1]
level_manip = [0.2, -0.2]
level_readout = [0.12, -0.12]

# Duration of each step
duration_init = 2500
duration_manip = 1000
duration_readout = readout_len + 100
duration_compensation_pulse = 4 * u.us

pi_length = 32
pi_half_length = 16

pi_amps = [0.27, -0.27]
pi_half_amps = [0.27, -0.27]


P1_amp = 0.27
P2_amp = 0.27
B_center_amp = 0.2
charge_sensor_amp = 0.2

block_length = 16
bias_length = 16

hold_offset_duration = 4

######################
#      RF GATES      #
######################
qubit_LO_left = 4 * u.GHz
qubit_IF_left = 100 * u.MHz
qubit_LO_right = 4 * u.GHz
qubit_IF_right = 100 * u.MHz

# Pi pulse


pi_amp_left = 0.1
pi_half_amp_left = 0.1
pi_length_left = 40
pi_amp_right = 0.1
pi_half_amp_right = 0.1
pi_length_right = 40
# Square pulse
cw_amp = 0.1
cw_len = 100
# Gaussian pulse
gaussian_length = 20
gaussian_amp = 0.1


#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # qubit_left I
                2: {"offset": 0.0},  # qubit_left Q
                3: {"offset": 0.0},  # qubit_right I
                4: {"offset": 0.0},  # qubit_right Q
                5: {"offset": 0.0},  # P1 qubit_left
                6: {"offset": 0.0},  # P2 qubit_right
                7: {"offset": 0.0},  # Barrier center
                8: {"offset": 0.0},  # charge sensor gate
                9: {"offset": 0.0},  # charge sensor DC
                10: {"offset": 0.0},  # charge sensor RF
            },
            "digital_outputs": {
                1: {},  # TTL for QDAC
                2: {},  # TTL for QDAC
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},  # DC input
                2: {"offset": 0.0, "gain_db": 0},  # RF input
            },
        },
    },
    "elements": {
        "B_center": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "operations": {
                "bias": "bias_B_center_pulse",
            },
        },
        "B_center_sticky": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "bias": "bias_B_center_pulse",
            },
        },
        "P1": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "operations": {
                "step": "bias_P1_pulse",
                "pi": "P1_pi_pulse",
                "pi_half": "P1_pi_half_pulse",
            },
        },
        "P1_sticky": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "bias": "bias_P1_pulse",
            },
        },
        "P2": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "operations": {
                "step": "bias_P2_pulse",
                "pi": "P2_pi_pulse",
                "pi_half": "P2_pi_half_pulse",
            },
        },
        "P2_sticky": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "bias": "bias_P2_pulse",
            },
        },
        "qdac_trigger1": {
            "digitalInputs": {
                "trigger": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                }
            },
            "operations": {
                "trigger": "trigger_pulse",
            },
        },
        "qdac_trigger2": {
            "digitalInputs": {
                "trigger": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                }
            },
            "operations": {
                "trigger": "trigger_pulse",
            },
        },
        "qubit_left": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO_left,
                "mixer": "mixer_qubit_left",  # a fixed name, do not change.
            },
            "intermediate_frequency": qubit_IF_left,
            "operations": {
                "cw": "cw_pulse",
                "pi": "pi_left_pulse",
                "gauss": "gaussian_pulse",
                "pi_half": "pi_half_left_pulse",
            },
        },
        "qubit_right": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO_right,
                "mixer": "mixer_qubit_right",  # a fixed name, do not change.
            },
            "intermediate_frequency": qubit_IF_right,
            "operations": {
                "cw": "cw_pulse",
                "pi": "pi_right_pulse",
                "gauss": "gaussian_pulse",
                "pi_half": "pi_half_right_pulse",
            },
        },
        "sensor_gate": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "operations": {
                "bias": "bias_charge_pulse",
            },
        },
        "sensor_gate_sticky": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "bias": "bias_charge_pulse",
            },
        },
        "tank_circuit": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "readout": "reflectometry_readout_pulse",
            },
            "outputs": {
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "TIA": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "pulses": {
        "P1_pi_pulse": {
            "operation": "control",
            "length": pi_length,
            "waveforms": {
                "single": "P1_pi_wf",
            },
        },
        "P1_pi_half_pulse": {
            "operation": "control",
            "length": pi_half_length,
            "waveforms": {
                "single": "P1_pi_half_wf",
            },
        },
        "P2_pi_pulse": {
            "operation": "control",
            "length": pi_length,
            "waveforms": {
                "single": "P2_pi_wf",
            },
        },
        "P2_pi_half_pulse": {
            "operation": "control",
            "length": pi_half_length,
            "waveforms": {
                "single": "P2_pi_half_wf",
            },
        },
        "bias_P1_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_P1_pulse_wf",
            },
        },
        "bias_P2_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_P2_pulse_wf",
            },
        },
        "bias_B_center_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_B_center_pulse_wf",
            },
        },
        "bias_charge_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_charge_pulse_wf",
            },
        },
        "cw_pulse": {
            "operation": "control",
            "length": cw_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gaussian_length,
            "waveforms": {
                "I": "gaussian_wf",
                "Q": "zero_wf",
            },
        },
        "pi_left_pulse": {
            "operation": "control",
            "length": pi_length_left,
            "waveforms": {
                "I": "pi_left_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_left_pulse": {
            "operation": "control",
            "length": pi_length_left,
            "waveforms": {
                "I": "pi_half_left_wf",
                "Q": "zero_wf",
            },
        },
        "pi_right_pulse": {
            "operation": "control",
            "length": pi_length_right,
            "waveforms": {
                "I": "pi_right_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_right_pulse": {
            "operation": "control",
            "length": pi_length_right,
            "waveforms": {
                "I": "pi_half_right_wf",
                "Q": "zero_wf",
            },
        },
        "trigger_pulse": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "ON",
        },
        "reflectometry_readout_pulse": {
            "operation": "measurement",
            "length": reflectometry_readout_length,
            "waveforms": {
                "single": "reflect_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
            "digital_marker": "ON",
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "readout_pulse_wf",
            },
            "integration_weights": {
                "constant": "constant_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "P1_pi_wf": {"type": "constant", "sample": pi_amps[0] - level_manip[0]},
        "P1_pi_half_wf": {"type": "constant", "sample": pi_half_amps[0] - level_manip[0]},
        "P2_pi_wf": {"type": "constant", "sample": pi_amps[1] - level_manip[1]},
        "P2_pi_half_wf": {"type": "constant", "sample": pi_half_amps[1] - level_manip[1]},
        "bias_P1_pulse_wf": {"type": "constant", "sample": P1_amp},
        "bias_P2_pulse_wf": {"type": "constant", "sample": P2_amp},
        "bias_B_center_pulse_wf": {"type": "constant", "sample": B_center_amp},
        "bias_charge_pulse_wf": {"type": "constant", "sample": charge_sensor_amp},
        "readout_pulse_wf": {"type": "constant", "sample": readout_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": cw_amp},
        "reflect_wf": {"type": "constant", "sample": reflect_amp},
        "gaussian_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in gaussian_amp * gaussian(gaussian_length, gaussian_length / 5)],
        },
        "pi_left_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in pi_amp_left * gaussian(pi_length_left, pi_length_left / 5)],
        },
        "pi_half_left_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in pi_half_amp_left * gaussian(pi_length_left, pi_length_left / 5)],
        },
        "pi_right_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in pi_amp_right * gaussian(pi_length_right, pi_length_right / 5)],
        },
        "pi_half_right_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in pi_half_amp_right * gaussian(pi_length_right, pi_length_right / 5)],
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "cosine_weights": {
            "cosine": [(1.0, reflectometry_readout_length)],
            "sine": [(0.0, reflectometry_readout_length)],
        },
        "sine_weights": {
            "cosine": [(0.0, reflectometry_readout_length)],
            "sine": [(1.0, reflectometry_readout_length)],
        },
    },
    "mixers": {
        "mixer_qubit_left": [
            {
                "intermediate_frequency": qubit_IF_left,
                "lo_frequency": qubit_LO_left,
                "correction": (1, 0, 0, 1),
            },
        ],
        "mixer_qubit_right": [
            {
                "intermediate_frequency": qubit_IF_right,
                "lo_frequency": qubit_LO_right,
                "correction": (1, 0, 0, 1),
            },
        ],
    },
}
