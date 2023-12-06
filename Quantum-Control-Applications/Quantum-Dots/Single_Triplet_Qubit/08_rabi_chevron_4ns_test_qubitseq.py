"""
A simple sandbox to showcase different QUA functionalities during the installation.
"""
import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.loops import from_array
from configuration import *
from scipy.optimize import minimize
from macros import RF_reflectometry_macro, DC_current_sensing_macro
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery import baking
from typing import List, Any, Dict
from qm.qua._dsl import _ResultSource, _Variable, _Expression
from qm import generate_qua_script


class OPX_background_sequence:
    def __init__(self, configuration: dict, elements: list):
        self._elements = elements
        self._config = configuration
        self.current_level = [0.0 for _ in self._elements]
        self._realtime = False
        self._voltage_points = {}
        for el in self._elements:
            self._config["elements"][el]["operations"]["step"] = "step_pulse"
        self._config["pulses"]["step_pulse"] = {
            "operation": "control",
            "length": 16,
            "waveforms": {"single": "step_wf"}
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
            "waveforms": {"single": wf_name}
        }
        self._config["waveforms"][wf_name] = {"type": "constant", "sample": amplitude}
        return op_name

    def add_step(self, level: list=None, duration: int=None, voltage_point_name:str = None, ramp_duration:int = None, current_offset:list = None):
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
                    if isinstance(voltage_level, (_Variable, _Expression)) and isinstance(_duration, (_Variable, _Expression)):
                    #     play("step" * amp((voltage_level - self.current_level[i]) * 4), gate)
                    #     wait((_duration - 16) >> 2, gate)
                    # if isinstance(_duration, (_Variable, _Expression)):
                        play("step" * amp((voltage_level - self.current_level[i] - current_offset[i]) * 4), gate)
                        wait((_duration - 16) >> 2, gate)
                    elif isinstance(_duration, (_Variable, _Expression)):
                        operation = self._add_op_to_config(gate, voltage_point_name,
                                                           amplitude=self._voltage_points[voltage_point_name][
                                                                         "coordinates"][i] - self.current_level[i],
                                                           length=self._voltage_points[voltage_point_name]["duration"])

                        play(operation, gate, duration=_duration >> 2)
                    else:
                        operation = self._add_op_to_config(gate, voltage_point_name,
                                                           amplitude=self._voltage_points[voltage_point_name][
                                                                         "coordinates"][i] - self.current_level[i],
                                                           length=self._voltage_points[voltage_point_name]["duration"])
                        play(operation, gate)

                else:
                    play(ramp((voltage_level - self.current_level[i]) / ramp_duration), gate, duration=ramp_duration >> 2)
                    wait(_duration >> 2, gate)
                self.current_level[i] = voltage_level

    def wait(self, duration):
        for i, gate in enumerate(self._elements):
            wait(duration >> 2, gate)
            self.current_level[i] = 0

    def add_points(self, name: str, coordinates: list, duration: int):
        self._voltage_points[name] = {}
        self._voltage_points[name]["coordinates"] = coordinates
        self._voltage_points[name]["duration"] = duration


###################
# The QUA program #
###################

# Relevant points in the charge stability map as ["P1", "P2"] in V
level_init = [0.1, -0.1]
level_manip = [0.3, -0.3]
level_readout = [0.2, -0.2]
# Duration of each step
duration_init = 2500
duration_manip = 1000
duration_readout = 500
pi_len = 44

durations = np.arange(16, 40, 10)
pi_levels = np.arange(0.21, 0.3, 0.01)

seq = OPX_background_sequence(config, ["P1_sticky", "P2_sticky"])
seq.add_points("initialization", level_init, duration_init)
seq.add_points("manipulation", level_manip, duration_manip)
seq.add_points("readout", level_readout, duration_readout)

qubit_seq = OPX_background_sequence(config, ["P1", "P2"])
qubit_seq.add_points("pi", [0.25, -0.25], pi_len)
pi_level = [-0.2, 0.05]


with program() as hello_qua:
    t = declare(int)
    a = declare(fixed)
    seq.add_step(voltage_point_name="readout")
    with for_(*from_array(a, pi_levels)):
        with for_(*from_array(t, durations)):
            with strict_timing_():
                seq.add_step(voltage_point_name="initialization", ramp_duration=None)
                seq.add_step(voltage_point_name="manipulation")
                seq.add_step(voltage_point_name="readout")

            wait((duration_init + duration_manip) * u.ns - (t>>2) - 4, "P1", "P2") # Need -4 because of a gap
            play("step" * amp((a-level_manip[0]) * 4), "P1", duration=t>>2)
            play("step" * amp((-a-level_manip[1]) * 4), "P2", duration=t>>2)

            wait((duration_init + duration_manip) * u.ns, "tank_circuit", "TIA")
            I, Q, I_st, Q_st = RF_reflectometry_macro()
            dc_signal, dc_signal_st = DC_current_sensing_macro()

# Program where the manipulation point is swept together with the qubit pulse
# with program() as hello_qua:
#     t = declare(int)
#     a = declare(fixed)
#     seq.add_step(voltage_point_name="readout")
#     with for_(*from_array(a, pi_levels)):
#         with for_(*from_array(t, durations)):
#             with strict_timing_():
#                 seq.add_step(voltage_point_name="initialization", ramp_duration=None)
#                 seq.add_step(voltage_point_name="manipulation", duration=t)
#                 seq.add_step(voltage_point_name="readout")
#
#                 wait(duration_init * u.ns, "P1", "P2")
#                 play("step" * amp((a-level_manip[0]) * 4), "P1", duration=t>>2)
#                 play("step" * amp((-a-level_manip[1]) * 4), "P2", duration=t>>2)
#                 # qubit_seq.wait(duration_init)
#                 # qubit_seq.add_step(voltage_point_name="pi", level=[a, -a], duration=t, current_offset=level_manip)

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, hello_qua, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    plt.axhline(0.1, color="k", linestyle="--")
    plt.axhline(0.3, color="k", linestyle="--")
    plt.axhline(0.2, color="k", linestyle="--")
    plt.axhline(-0.1, color="k", linestyle="--")
    plt.axhline(-0.3, color="k", linestyle="--")
    plt.axhline(-0.2, color="k", linestyle="--")
    plt.yticks([-0.2, -0.3, -0.1, 0.0, 0.1, 0.3, 0.2], ["readout", "manip", "init", "0", "init", "manip", "readout"])
    plt.legend("")
    samples = job.get_simulated_samples()
    report = job.get_simulated_waveform_report()
    report.create_plot(samples, plot=True)
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(hello_qua)
