import numpy as np
from qualang_tools.units import unit
from qm.qua._dsl import _Variable, _Expression
from qm.qua import declare, assign, play, fixed, Cast, amp, wait, ramp, ramp_to_zero
from qdac2_driver import QDACII, load_voltage_list

####################
# Helper functions #
####################
def update_readout_length(new_readout_length, config):

    config["pulses"]["lock_in_readout_pulse"]["length"] = new_readout_length
    config["integration_weights"]["cosine_weights"] = {
        "cosine": [(1.0, new_readout_length)],
        "sine": [(0.0, new_readout_length)],
    }
    config["integration_weights"]["sine_weights"] = {
        "cosine": [(0.0, new_readout_length)],
        "sine": [(1.0, new_readout_length)],
    }
    config["integration_weights"]["minus_sine_weights"] = {
        "cosine": [(0.0, new_readout_length)],
        "sine": [(-1.0, new_readout_length)],
    }
    
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

qdac_ip = "127.0.0.1"
qdac_port = 5025

# Path to save data
octave_config = None

#############################################
#              OPX PARAMETERS               #
#############################################
class OPX_virtual_gate_sequence:
    def __init__(self, configuration: dict, elements: list):
        """Framework allowing to design an arbitrary pulse sequence using virtual gates and pre-defined point from the
        charge stability map. TODO better docstring explaining how it works

        :param configuration: The OPX configuration.
        :param elements: List containing the elements taking part in the virtual gate.
        """
        # List of the elements involved in the virtual gates
        self._elements = elements
        # The OPX configuration
        self._config = configuration
        # Initialize the current voltage level for sticky elements
        self.current_level = [0.0 for _ in self._elements]
        # Relevant voltage points in the charge stability diagram
        self._voltage_points = {}
        # Keep track of the averaged voltage played for defining the compensation pulse at the end of the sequence
        self.average_power = [0 for _ in self._elements]
        self._expression = None
        self._expression2 = None
        # Add to the config the step operation (length=16ns & amp=0.25V)
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

    def _add_op_to_config(self, el: str, name: str, amplitude: float, length: int) -> str:
        """Add an operation to an element when the amplitude is fixed to release the number of real-time operations on
        the OPX.

        :param el: the element to which we want to add the operation.
        :param name: name of the operation.
        :param amplitude: Amplitude of the pulse in V.
        :param length: Duration of the pulse in ns.
        :return : The name of the created operation.
        """
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

    @staticmethod
    def _check_duration(duration: int):
        if duration is not None and not isinstance(duration, (_Variable, _Expression)):
            assert duration >= 4, "The duration must be a larger than 16 ns."

    def _update_averaged_power(self, level, duration, ramp_duration=None, current_level=None):
        if self.is_QUA(level):
            self._expression = declare(fixed)
            assign(self._expression, level)
            new_average = Cast.mul_int_by_fixed(duration, self._expression)
        elif self.is_QUA(duration):
            new_average = Cast.mul_int_by_fixed(duration, level)
        else:
            new_average = int(np.round(level * duration))

        if ramp_duration is not None:
            if not self.is_QUA(ramp_duration):
                if self.is_QUA(level):
                    self._expression2 = declare(fixed)
                    assign(self._expression2, (self._expression + current_level) >> 1)
                    new_average += Cast.mul_int_by_fixed(ramp_duration, self._expression2)
                elif self.is_QUA(current_level):
                    expression2 = declare(fixed)
                    assign(expression2, (level + current_level) >> 1)
                    new_average += Cast.mul_int_by_fixed(ramp_duration, expression2)
                elif self.is_QUA(duration):
                    new_average += Cast.mul_int_by_fixed(ramp_duration, (level + current_level) / 2)
                else:
                    new_average += int(np.round((level + current_level) * ramp_duration / 2))

            else:
                pass
        return new_average

    @staticmethod
    def is_QUA(var):
        return isinstance(var, (_Variable, _Expression))

    def add_step(
        self,
        level: list = None,
        duration: int = None,
        voltage_point_name: str = None,
        ramp_duration: int = None,
    ) -> None:
        """Add a voltage level to the pulse sequence.
        The voltage level is either identified by its voltage_point_name if added to the voltage_point dict beforehand, or by its level and duration.
        A ramp_duration can be used to ramp to the desired level instead of stepping to it.

        :param level: Desired voltage level of the different gates composing the virtual gate in Volt.
        :param duration: How long the voltage level should be maintained in ns. Must be a multiple of 4ns and larger than 16ns.
        :param voltage_point_name: Name of the voltage level if added to the list of relevant points in the charge stability map.
        :param ramp_duration: Duration in ns of the ramp if the voltage should be ramped to the desired level instead of stepped. Must be a multiple of 4ns and larger than 16ns.
        """
        self._check_duration(duration)
        self._check_duration(ramp_duration)

        if voltage_point_name is not None and duration is None:
            _duration = self._voltage_points[voltage_point_name]["duration"]
        elif duration is not None:
            _duration = duration
        else:
            raise RuntimeError(
                "Either the voltage_point_name or the duration and desired voltage level must be provided."
            )

        for i, gate in enumerate(self._elements):
            if voltage_point_name is not None and level is None:
                voltage_level = self._voltage_points[voltage_point_name]["coordinates"][i]
            elif level is not None:
                voltage_point_name = "unregistered_value"
                voltage_level = level[i]
            else:
                raise RuntimeError(
                    "Either the voltage_point_name or the duration and desired voltage level must be provided."
                )
            # Play a step
            if ramp_duration is None:
                self.average_power[i] += self._update_averaged_power(voltage_level, _duration)

                # Dynamic amplitude change...
                if self.is_QUA(voltage_level) or self.is_QUA(self.current_level[i]):
                    # if dynamic duration --> play step and wait
                    if self.is_QUA(_duration):
                        play("step" * amp((voltage_level - self.current_level[i]) * 4), gate)
                        wait((_duration - 16) >> 2, gate)
                    # if constant duration --> new operation and play(*amp(..))
                    else:
                        operation = self._add_op_to_config(
                            gate,
                            "step",
                            amplitude=0.25,
                            length=_duration,
                        )
                        play(operation * amp((voltage_level - self.current_level[i]) * 4), gate)

                # Fixed amplitude but dynamic duration --> new operation and play(duration=..)
                elif isinstance(_duration, (_Variable, _Expression)):
                    operation = self._add_op_to_config(
                        gate,
                        voltage_point_name,
                        amplitude=voltage_level - self.current_level[i],
                        length=16,
                    )
                    play(operation, gate, duration=_duration >> 2)

                # Fixed amplitude and duration --> new operation and play()
                else:
                    operation = self._add_op_to_config(
                        gate,
                        voltage_point_name,
                        amplitude=voltage_level - self.current_level[i],
                        length=_duration,
                    )
                    play(operation, gate)

            # Play a ramp
            else:
                self.average_power[i] += self._update_averaged_power(
                    voltage_level, _duration, ramp_duration, self.current_level[i]
                )

                if not self.is_QUA(ramp_duration):
                    ramp_rate = 1 / ramp_duration
                    play(ramp((voltage_level - self.current_level[i]) * ramp_rate), gate, duration=ramp_duration >> 2)
                    wait(_duration >> 2, gate)

            self.current_level[i] = voltage_level

    def add_compensation_pulse(self, duration: int) -> None:
        """Add a compensation pulse of the specified duration whose amplitude is derived from the previous operations.

        :param duration: Duration of the compensation pulse in clock cycles (4ns). Must be larger than 4 clock cycles.
        """
        self._check_duration(duration)
        for i, gate in enumerate(self._elements):
            if not self.is_QUA(self.average_power[i]):
                compensation_amp = -self.average_power[i] / duration
                operation = self._add_op_to_config(
                    gate, "compensation", amplitude=compensation_amp - self.current_level[i], length=duration
                )
                play(operation, gate)
            else:
                operation = self._add_op_to_config(gate, "compensation", amplitude=0.25, length=duration)
                compensation_amp = declare(fixed)
                eval_average_power = declare(int)
                assign(eval_average_power, self.average_power[i])
                assign(compensation_amp, -Cast.mul_fixed_by_int(1 / duration, eval_average_power))
                play(operation * amp((compensation_amp - self.current_level[i]) * 4), gate)
            self.current_level[i] = compensation_amp

    def ramp_to_zero(self, duration: int = None):
        """Ramp all the gate voltages down to zero Volt and reset the averaged voltage derived for defining the compensation pulse.

        :param duration: How long will it take for the voltage to ramp down to 0V in clock cycles (4ns). If not
            provided, the default pulse duration defined in the configuration will be used.
        """
        for i, gate in enumerate(self._elements):
            ramp_to_zero(gate, duration)
            self.current_level[i] = 0
            self.average_power[i] = 0
        if self._expression is not None:
            assign(self._expression, 0)
        if self._expression2 is not None:
            assign(self._expression2, 0)

    def add_points(self, name: str, coordinates: list, duration: int) -> None:
        """Register a relevant voltage point.

        :param name: Name of the voltage point.
        :param coordinates: Voltage value of each gate involved in the virtual gate in V.
        :param duration: How long should the voltages be maintained at this level in ns. Must be larger than 16ns and a multiple of 4ns.
        """
        self._voltage_points[name] = {}
        self._voltage_points[name]["coordinates"] = coordinates
        self._voltage_points[name]["duration"] = duration


######################
#       READOUT      #
######################
qds_IF = 1 * u.MHz
lock_in_readout_length = 1 * u.us
lock_in_readout_amp = 10 * u.mV
rotation_angle = (0.0 / 180) * np.pi

# Time of flight
time_of_flight = 24

######################
#      DC GATES      #
######################

## Section defining the points from the charge stability map - can be done in the config
level_readout = [0.12, -0.12]
level_dephasing = [-0.2, -0.1]

dephasing_ramp = 100
readout_ramp = 100
init_ramp = 100

# Duration of each step in ns
duration_readout = lock_in_readout_length
duration_compensation_pulse = 5 * u.us
duration_dephasing = 2000  # nanoseconds
duration_init = 10_000
duration_init_jumps = 16

# Step parameters
step_length = 16
P4_step_amp = 0.25
P5_step_amp = 0.25
P6_step_amp = 0.25
X4_step_amp = 0.25
X5_step_amp = 0.25
T6_step_amp = 0.25
charge_sensor_amp = 0.25

# Time to ramp down to zero for sticky elements in ns
ramp_down_duration = 4
bias_tee_cut_off_frequency = 400 * u.Hz

######################
#    QUBIT PULSES    #
######################
# Durations in ns
pi_length = 32
pi_half_length = 16
# Amplitudes in V
pi_amps = [0.27, -0.27]
pi_half_amps = [0.27, -0.27]


#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # 
                2: {"offset": 0.0},  # QDS 1 MHz drive
                3: {"offset": 0.0},  # 
                4: {"offset": 0.0},  # 
                5: {"offset": 0.0},  # P4
                6: {"offset": 0.0},  # X4
                7: {"offset": 0.0},  # P5
                8: {"offset": 0.0},  # X5
                9: {"offset": 0.0},  # P6
                10: {"offset": 0.0},  # T6
            },
            "digital_outputs": {
                1: {},  # TTL for QDAC
                2: {},  # TTL for QDAC
            },
            "analog_inputs": {
                2: {"offset": 0.0, "gain_db": 0},  # Lock-in channel
            },
        },
    },
    "elements": {
        "P4": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "operations": {
                "step": "P4_step_pulse",
            },
        },
        "P4_sticky": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "P4_step_pulse",
            },
        },
        "P5": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "operations": {
                "step": "P5_step_pulse",
            },
        },
        "P5_sticky": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "P5_step_pulse",
            },
        },
        "P6": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "operations": {
                "step": "P6_step_pulse",
            },
        },
        "P6_sticky": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "P6_step_pulse",
            },
        },
        "X4": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "operations": {
                "step": "X4_step_pulse",
            },
        },
        "X4_sticky": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "X4_step_pulse",
            },
        },
        "X5": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "operations": {
                "step": "X5_step_pulse",
            },
        },
        "X5_sticky": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "X5_step_pulse",
            },
        },
        "T6": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "operations": {
                "step": "T6_step_pulse",
            },
        },
        "T6_sticky": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "T6_step_pulse",
            },
        },
        "sensor_gate": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "operations": {
                "step": "bias_charge_pulse",
            },
        },
        "sensor_gate_sticky": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "bias_charge_pulse",
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
        "QDS": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "intermediate_frequency": qds_IF,
            "operations": {
                "readout": "lock_in_readout_pulse",
            },
            "outputs": {
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "QDS_twin": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "intermediate_frequency": qds_IF,
            "operations": {
                "readout": "lock_in_readout_pulse",
            },
            "outputs": {
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "pulses": {
        "P4_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "P4_step_wf",
            },
        },
        "P5_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "P5_step_wf",
            },
        },
        "P6_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "P6_step_wf",
            },
        },
        "X4_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "X4_step_wf",
            },
        },
        "X5_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "X5_step_wf",
            },
        },
        "T6_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "T6_step_wf",
            },
        },
        "bias_charge_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "charge_sensor_step_wf",
            },
        },
        "trigger_pulse": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "ON",
        },
        "lock_in_readout_pulse": {
            "operation": "measurement",
            "length": lock_in_readout_length,
            "waveforms": {
                "single": "lock_in_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "rotated_cos": "rotated_cosine_weights",
                "rotated_sin": "rotated_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "P4_step_wf": {"type": "constant", "sample": P4_step_amp},
        "P5_step_wf": {"type": "constant", "sample": P5_step_amp},
        "P6_step_wf": {"type": "constant", "sample": P6_step_amp},
        "X4_step_wf": {"type": "constant", "sample": X4_step_amp},
        "X5_step_wf": {"type": "constant", "sample": X5_step_amp},
        "T6_step_wf": {"type": "constant", "sample": T6_step_amp},
        "charge_sensor_step_wf": {"type": "constant", "sample": charge_sensor_amp},
        "lock_in_wf": {"type": "constant", "sample": lock_in_readout_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1, lock_in_readout_length)],
            "sine": [(0.0, lock_in_readout_length)],
        },
        "cosine_weights": {
            "cosine": [(1.0, lock_in_readout_length)],
            "sine": [(0.0, lock_in_readout_length)],
        },
        "sine_weights": {
            "cosine": [(0.0, lock_in_readout_length)],
            "sine": [(1.0, lock_in_readout_length)],
        },
        "rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), lock_in_readout_length)],
            "sine": [(np.sin(rotation_angle), lock_in_readout_length)],
        },
        "rotated_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), lock_in_readout_length)],
            "sine": [(np.cos(rotation_angle), lock_in_readout_length)],
        },
    },
}