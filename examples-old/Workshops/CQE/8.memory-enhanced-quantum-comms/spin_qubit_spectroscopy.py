from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from Configuration import config
from Configuration import qm
import matplotlib.pyplot as plt

#############
# Time trace:
#############

meas_len = 1e3
rep_num = 1e7

with program() as spin_qubit_spec:

    ##############################
    # Declare real-time variables:
    ##############################

    n = declare(int)
    f = declare(int)
    result1 = declare(int, size=int(meas_len / 500))
    resultLen1 = declare(int)
    result2 = declare(int, size=int(meas_len / 500))
    resultLen2 = declare(int)
    counts = declare(int)

    ###############
    # The sequence:
    ###############

    with for_(n, 0, n < rep_num, n + 1):
        with for_(f, 4e6, f < 14e6, f + 4e6):
            update_frequency("spin_qubit", f)
            play("saturation", "spin_qubit", duration=1000)
            align("spin_qubit", "readout", "readout1", "readout2")
            play("on", "readout", duration=1000)
            measure(
                "readout",
                "readout1",
                "adc",
                time_tagging.raw(result1, meas_len, targetLen=resultLen1),
            )
            measure(
                "readout",
                "readout2",
                None,
                time_tagging.raw(result2, meas_len, targetLen=resultLen2),
            )
            assign(counts, resultLen1 + resultLen2)

        save(counts, "counts")

        # with for_(n, 0, n < resultLen1, n + 1):
        #     save(result1[n], "res1")
        # with for_(p, 0, p < resultLen2, p + 1):
        #     save(result2[p], "res2")


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
