"""
resonator_spec_g_e.py: template for performing the 1D resonator spectroscopy for a ground and excited qubit (IO values)
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

##############################
# Program-specific variables #
##############################
n_avg = 1000  # Number of averaging loops

cooldown_time = 2 * u.us // 4  # Resonator cooldown time in clock cycles (4ns)
flux_settle_time = 10 * u.us // 4  # Flux settle time in clock cycles (4ns)

# Frequency sweep in Hz
f_min = 55 * u.MHz
f_max = 65 * u.MHz
df = 100 * u.kHz
freqs = np.arange(f_min, f_max + df / 2, df)  # +df/2 to add f_max to the scan

###################
# The QUA program #
###################
with program() as resonator_spec:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    flag = declare(bool)  # Flag conditioning a pi-pulse
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    # Get the value of flag from outside the QUA program (IO values)
    pause()
    assign(flag, IO1)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            # Adjust the flux line
            play("const" * amp(0), "flux_line")
            wait(flux_settle_time, "resonator")
            # Update the resonator frequency
            update_frequency("resonator", f)
            # Play a pi pulse on conditional flag (I/O values)
            with if_(flag):
                play("pi", "qubit")
                align("qubit", "resonator")
            # Measure the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            # Wait for the resonator to cooldown
            wait(cooldown_time, "resonator", "flux_line")
            # Save data to the stream processing
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(freqs)).average().save("Q")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = True
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    # Ground state spectroscopy
    job = qm.execute(resonator_spec)  # Execute QUA program
    qm.set_io1_value(False)  # Set the value of flag
    job.resume()  # Resume to the program
    Flag1_got = qm.get_io1_value()  # Check the value of flag
    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    Ig = res_handles.get("I").fetch_all()
    Qg = res_handles.get("Q").fetch_all()

    # Excited state spectroscopy
    job = qm.execute(resonator_spec)  # Execute QUA program
    qm.set_io1_value(True)  # Set the value of flag
    job.resume()  # Resume to the program
    Flag2_got = qm.get_io1_value()  # Check the value of flag
    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    Ie = res_handles.get("I").fetch_all()
    Qe = res_handles.get("Q").fetch_all()

    # Plots
    plt.figure(figsize=(8, 12))
    plt.subplot(221)
    plt.plot(freqs / u.MHz, np.sqrt(Ig**2 + Qg**2), "b.")
    plt.plot(freqs / u.MHz, np.sqrt(Ie**2 + Qe**2), "r.")
    plt.xlabel("freq [MHz]")
    plt.ylabel("resonator spectroscopy amplitude")
    plt.subplot(222)
    # detrend removes the linear increase of phase
    phase_g = signal.detrend(np.unwrap(np.angle(Ig + 1j * Qg)))
    phase_e = signal.detrend(np.unwrap(np.angle(Ie + 1j * Qe)))
    plt.plot(freqs / u.MHz, phase_g, "b.")
    plt.plot(freqs / u.MHz, phase_e, "r.")
    plt.xlabel("freq [MHz]")
    plt.ylabel("resonator spectroscopy phase")
    plt.subplot(223)
    plt.plot(freqs / u.MHz, np.sqrt(Ig**2 + Qg**2) - np.sqrt(Ie**2 + Qe**2))
    plt.xlabel("freq [MHz]")
    plt.ylabel("Difference between g and e (amplitude)")
    plt.subplot(224)
    plt.plot(freqs / u.MHz, phase_g - phase_e)
    plt.xlabel("freq [MHz]")
    plt.ylabel("Difference between g and e (phase)")
    plt.tight_layout()
