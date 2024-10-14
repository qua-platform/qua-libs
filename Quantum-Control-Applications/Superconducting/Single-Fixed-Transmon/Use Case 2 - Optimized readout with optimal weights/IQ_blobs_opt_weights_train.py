"""
Measure the qubit in the ground and excited state to create the IQ blobs. If the separation and the fidelity are good
enough, gives the parameters needed for active reset
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *
from TwoStateDiscriminator import TwoStateDiscriminator
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

###################
# The QUA program #
###################

n_runs = 1000

lsb = False

rr_qe = "resonator"

cooldown_time = 5 * qubit_T1

with program() as training:
    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    adc_st = declare_stream(adc_trace=True)

    with for_(n, 0, n < n_runs, n + 1):
        measure(
            "readout",
            "resonator",
            adc_st,
            dual_demod.full("cos", "sin", I),
            dual_demod.full("minus_sin", "cos", Q),
        )
        save(I, I_st)
        save(Q, Q_st)
        wait(cooldown_time * u.ns, "resonator")

        align()  # global align

        play("x180", "qubit")
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            adc_st,
            dual_demod.full("cos", "sin", I),
            dual_demod.full("minus_sin", "cos", Q),
        )
        save(I, I_st)
        save(Q, Q_st)
        wait(cooldown_time * u.ns, "resonator")

    with stream_processing():
        I_st.save_all("I")
        Q_st.save_all("Q")
        adc_st.input1().with_timestamps().save_all("adc1")
        adc_st.input2().save_all("adc2")

discriminator = TwoStateDiscriminator(
    qmm=qmm,
    config=config,
    update_tof=False,
    rr_qe=rr_qe,
    path=f"ge_disc_params_{rr_qe}.npz",
    lsb=lsb,
    meas_len=readout_len,
    smearing=smearing,
)


discriminator.train(program=training, plot=True, dry_run=True, correction_method="robust")
