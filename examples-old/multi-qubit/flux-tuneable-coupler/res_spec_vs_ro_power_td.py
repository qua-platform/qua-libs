from Configuration import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np

avgs = 1000

a_min = 0.2
a_max = 1.0
da = 0.1
a_vec = [float(arg) for arg in np.arange(a_min, a_max + da / 2, da)]

f_min = int(48e6)
f_max = int(52e6)
df = 1e6
f_vec = [int(arg) for arg in np.arange(f_min, f_max + df / 2, df)]

rr_relaxation_time = 300  # ns

with program() as res_spec_vs_ro_power_td:

    n = declare(int)
    f = declare(int)
    a = declare(fixed)

    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)

    I_stream = declare_stream()
    Q_stream = declare_stream()

    with for_(n, 0, n < avgs, n + 1):

        with for_(f, f_min, f < f_max + df / 2, f + df):

            update_frequency("RR_1", f)

            with for_(a, a_min, a < a_max + da / 2, a + da):

                wait(rr_relaxation_time // 4, "RR_1")

                measure(
                    "long_readout" * amp(a),
                    "RR_1",
                    None,
                    demod.full("long_integW_cos", I1, "out1"),
                    demod.full("long_integW_sin", Q1, "out1"),
                    demod.full("long_integW_cos", I2, "out2"),
                    demod.full("long_integW_sin", Q2, "out2"),
                )
                assign(I, I1 + Q2)
                assign(Q, Q1 - I2)
                save(I, I_stream)
                save(Q, Q_stream)

    with stream_processing():

        I_stream.buffer(len(f_vec), len(a_vec)).average().save("I")
        Q_stream.buffer(len(f_vec), len(a_vec)).average().save("Q")

######################################
# OPEN COMMUNICATION WITH THE SERVER #
######################################
qmm = QuantumMachinesManager()

##########################
# OPEN A QUANTUM MACHINE #
##########################
qm = qmm.open_qm(config)

########################
# SIMULATE QUA PROGRAM #
########################

job = qm.simulate(res_spec_vs_ro_power_td, SimulationConfig(10000))
plt.figure()
job.get_simulated_samples().con1.plot()
