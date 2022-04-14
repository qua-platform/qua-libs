from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from Configuration import config
from Configuration import qm
import matplotlib.pyplot as plt


#############
# Time trace:
#############
from qua_macros import measure_spin

meas_len = 1e3
rep_num = 1e7
threshold = -7

with program() as spin_qubit_spec:

    se = declare(bool)

    measure_spin("z", threshold, se)
    with while_(se):
        play("pi", "spin_qubit")
        measure_spin("z", 7, se)

    # Save
    save(se, "states")

job = qm.simulate(spin_qubit_spec, SimulationConfig(8000))
job.get_simulated_samples().con1.plot()

job = qm.simulate(spin_qubit_spec, SimulationConfig(8000))
# job.get_simulated_samples().con1.plot()

sim = job.get_simulated_samples()
pulses = job.simulated_analog_waveforms()
dig_pulses = job.simulated_digital_waveforms()


plt.figure(1)
ax1 = plt.subplot(311)
plt.plot(sim.con1.analog["1"])
plt.plot(sim.con1.analog["2"])
plt.ylabel("spin_qubit")

plt.subplot(312)
plt.plot(sim.con1.digital["1"])
plt.ylabel("readout")

plt.subplot(313)
plt.plot(sim.con1.digital["7"])
plt.plot(sim.con1.digital["8"])
plt.ylabel("counting")
