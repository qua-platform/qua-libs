"""
QUA script to optimize the readout frequency
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt

###################
# The QUA program #
###################

n_avg = 10000

cooldown_time = 5 * qubit_T1 // 4

f_min = 70e6
f_max = 80e6
df = 0.1e6

freqs = np.arange(f_min, f_max + df / 2, df)  # + df/2 to add f_max to freqs

with program() as IQ_blobs:
    n = declare(int)
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()
    n_st = declare_stream()

    f = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            update_frequency("qubit", f)
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
            )
            save(I_g, I_g_st)
            save(Q_g, Q_g_st)
            wait(cooldown_time, "resonator")

            align()  # global align

            play("x180", "qubit")
            align("qubit", "resonator")
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
            )
            save(I_e, I_e_st)
            save(Q_e, Q_e_st)
            wait(cooldown_time, "resonator")

    with stream_processing():
        n_st.save("iteration")
        # mean values
        I_g_st.buffer(len(freqs)).average().save("I_g_avg")
        Q_g_st.buffer(len(freqs)).average().save("Q_g_avg")
        I_e_st.buffer(len(freqs)).average().save("I_e_avg")
        Q_e_st.buffer(len(freqs)).average().save("Q_e_avg")
        # variances
        (
            ((I_g_st.buffer(len(freqs)) * I_g_st.buffer(len(freqs))).average())
            - (I_g_st.buffer(len(freqs)).average() * I_g_st.buffer(len(freqs)).average())
        ).save("I_g_var")
        (
            ((Q_g_st.buffer(len(freqs)) * Q_g_st.buffer(len(freqs))).average())
            - (Q_g_st.buffer(len(freqs)).average() * Q_g_st.buffer(len(freqs)).average())
        ).save("Q_g_var")
        (
            ((I_e_st.buffer(len(freqs)) * I_e_st.buffer(len(freqs))).average())
            - (I_e_st.buffer(len(freqs)).average() * I_e_st.buffer(len(freqs)).average())
        ).save("I_e_var")
        (
            ((Q_e_st.buffer(len(freqs)) * Q_e_st.buffer(len(freqs))).average())
            - (Q_e_st.buffer(len(freqs)).average() * Q_e_st.buffer(len(freqs)).average())
        ).save("Q_e_var")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(qop_ip)

qm = qmm.open_qm(config)

job = qm.execute(IQ_blobs)  # execute QUA program

# Get results from QUA program
results = fetching_tool(
    job,
    data_list=["I_g_avg", "Q_g_avg", "I_e_avg", "Q_e_avg", "I_g_var", "Q_g_var", "I_e_var", "Q_e_var", "iteration"],
    mode="live",
)

# Live plotting
fig = plt.figure()
interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

while results.is_processing():
    # Fetch results
    I_g_avg, Q_g_avg, I_e_avg, Q_e_avg, I_g_var, Q_g_var, I_e_var, Q_e_var, iteration = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg, start_time=results.get_start_time())
    # Plot results
    plt.cla()
    Z = (I_e_avg - I_g_avg) + 1j * (Q_e_avg - Q_g_avg)
    var = (I_g_var + Q_g_var + I_e_var + Q_e_var) / 4
    SNR = ((np.abs(Z)) ** 2) / (2 * var)
    plt.plot(freqs, SNR, ".-")
    plt.title("Readout optimization")
    plt.xlabel("Readout frequency [Hz]")
    plt.ylabel("SNR")
    plt.pause(0.1)
