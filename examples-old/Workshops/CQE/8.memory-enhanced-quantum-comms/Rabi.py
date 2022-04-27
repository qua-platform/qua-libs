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
threshold = 7

with program() as Rabi:

    ##############################
    # Declare real-time variables:
    ##############################

    n = declare(int)
    t = declare(int)
    result1 = declare(int, size=int(meas_len / 500))
    resultLen1 = declare(int)
    result2 = declare(int, size=int(meas_len / 500))
    resultLen2 = declare(int)
    counts = declare(int)
    se = declare(bool)

    ###############
    # The sequence:
    ###############

    with for_(n, 0, n < rep_num, n + 1):
        with for_(t, 60, t < 400, t + 40):
            # Qubit preparation:
            play("pi", "spin_qubit", duration=t)

            # Readout:
            # align('spin_qubit', 'readout', 'readout1', 'readout2')
            # play('on', 'readout', duration=1000)
            # measure('readout', 'readout1', None, time_tagging.raw(result1, meas_len, targetLen=resultLen1))
            # measure('readout', 'readout2', None, time_tagging.raw(result2, meas_len, targetLen=resultLen2))
            # assign(counts, resultLen1+resultLen2)
            # assign(se, counts>threshold)
            measure_spin("z", threshold, se)

        # Save to client
        save(se, "states")

# job = qm.execute(Rabi)

job = qm.simulate(Rabi, SimulationConfig(8000))
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
