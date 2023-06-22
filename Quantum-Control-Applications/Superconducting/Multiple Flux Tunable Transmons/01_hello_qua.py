from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool

def measurement():
    measure("readout", "rr1", None,
            dual_demod.full("cos", "out1", "sin", "out2", I),
            dual_demod.full("minus_sin", "out1", "cos", "out2", Q))
    return I, Q

# QUA program
n_avg = 10000
amplitudes = np.arange(-2.2,0.1)
freqs = np.arange(100e6, 200e6, 1e6)
with program() as hello_qua:
    f = declare(int) # 32bits integer
    t = declare(int)
    n = declare(int)
    a = declare(fixed) # fixed: (1) 3.28 --> [-8, 8) resolution of 2**-28
    I = declare(fixed) # fixed: (1) 3.28 --> [-8, 8) resolution of 2**-28
    Q = declare(fixed) # fixed: (1) 3.28 --> [-8, 8) resolution of 2**-28
    I_st = declare_stream()
    Q_st = declare_stream()
    with for_(n, 0, n<n_avg, n+1):
        with for_(*from_array(f, freqs)):
            update_frequency("q1_xy", 0)
            # with for_(t, 10, t < 100, t+10):  # FPGA clock cycle is 4ns (250MHz) and minimum pulse is 16ns (4 cc)
            with for_(*from_array(a, amplitudes)):
                play("x180" * amp(a), "q1_xy")  # |prefactor| < 2
                wait(1000*u.ns, "q1_xy")
                I, Q = measurement()

                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(amplitudes)).buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(amplitudes)).buffer(len(freqs)).average().save("Q")


# open communication with opx
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

# simulate the test_config QUA program
job = qmm.simulate(config, hello_qua, SimulationConfig(40_000))
job.get_simulated_samples().con1.plot()
plt.show()

qm = qmm.open_qm(config)
job = qm.execute(hello_qua)

results = fetching_tool(job, ['I', 'Q'], mode='live')

while results.is_processing():
    I, Q = results.fetch_all()
    plt.plot(I)
    plt.pause(0.1)