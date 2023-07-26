from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from macros import multiplexed_readout

###################
# The QUA program #
###################
n_avg = 4000
dfs = np.arange(-0.5e6, 0.5e6, 0.02e6)
cooldown_time = 1 * u.us

with program() as iq_blobs:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(2)]
    Q_g = [declare(fixed) for _ in range(2)]
    I_e = [declare(fixed) for _ in range(2)]
    Q_e = [declare(fixed) for _ in range(2)]
    DI = declare(fixed)
    DQ = declare(fixed)
    D = declare(fixed)
    df = declare(int)
    D_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            update_frequency("rr1", df + resonator_IF_q1)
            update_frequency("rr2", df + resonator_IF_q2)

            # ground iq blobs for both qubits
            wait(cooldown_time * u.ns)
            align()
            multiplexed_readout(I_g, None, Q_g, None, resonators=[1, 2], weights="rotated_")

            # excited iq blobs for both qubits
            align()
            play("x180", "q1_xy")
            # play("x180", "q2_xy")
            align()
            multiplexed_readout(I_e, None, Q_e, None, resonators=[1, 2], weights="rotated_")

            assign(DI, (I_e[1] - I_g[1]) * 100)
            assign(DQ, (Q_e[1] - Q_g[1]) * 100)
            assign(D, DI * DI + DQ * DQ)
            save(D, D_st)

    with stream_processing():
        D_st.buffer(len(dfs)).average().save("D")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave=octave_config)

# open quantum machine
qm = qmm.open_qm(config)

# run job
job = qm.execute(iq_blobs)

# fetch data
results = fetching_tool(job, ["D"])
D = results.fetch_all()[0]
plt.plot(dfs, D)
plt.xlabel("Readout detuning (MHz)")
plt.ylabel("Distance between IQ blobs (a.u.)")
print(f"Shift readout frequency by {dfs[np.argmax(D)]} Hz")
plt.show()
