"""
manual_mixer_calibration.py: Calibration for mixer imperfections
Author: Yoav Romach - Quantum Machines
Created: 22/12/2020
Created on QUA version: 0.7.411
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *

QMm = QuantumMachinesManager()

simulate = False
with program() as play_pulse_cont:
    with infinite_loop_():
        play('test_pulse', 'qubit')

QM1 = QMm.open_qm(config)

if simulate:
    job = QM1.simulate(play_pulse_cont, SimulationConfig(1000))
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    job = QM1.execute(play_pulse_cont)

    # When done, the halt command can be called and the offsets can be written directly into the config file.
    # job.halt

# These are the 3 commands used to correct for mixer imperfections. The first two commands are used to set the DC of the
# I, Q channels to compensate for the LO leakage. The last command is used to correct for the phase and amplitude
# mismatches between the channels.
# The output if the IQ Mixer should be connected to a spectrum analyzer and values should be chosen as to minimize the
# unwanted peaks.
# If python can read the output of the spectrum analyzer, then this process can be automated and the correct values can
# found using an optimization method such as Nelder-Mead:
# https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html

# QM1.set_output_dc_offset_by_element('qubit', 'I', -0.05)
# QM1.set_output_dc_offset_by_element('qubit', 'Q', 0.03)
# QM1.set_mixer_correction('mixer_qubit', int(qubit_IF), int(qubit_LO), IQ_imbalance_correction(0.15, 0.3))
