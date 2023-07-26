from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import LoopbackInterface
from qm import SimulationConfig
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from scipy.signal import detrend
from qualang_tools.loops import from_array


###################
# The QUA program #
###################

## rr1
# freqs = np.arange(47e6, 51e6, 0.05e6)
# rr2
freqs = np.arange(-135e6, -128e6, 0.05e6)

depletion_time = 1000
n_avg = 1000

with program() as res_spec:
    n = declare(int)
    f = declare(int)  # Hz int 32 up to 2^32
    I = declare(fixed)  # signed 4.28 [-8, 8)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, freqs)):
            update_frequency("rr1", f)

            measure(
                "readout",
                "rr1",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )  # rr1
            # measure("readout", "rr2", None, dual_demod.full("cos", "out1", "sin", "out2", I),
            #         dual_demod.full("minus_sin", "out1", "cos", "out2", Q))
            wait(depletion_time * u.ns, "rr1")
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(freqs)).average().save("Q")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave=octave_config)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(
        config,
        res_spec,
        SimulationConfig(
            11000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=250)
        ),
    )
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(res_spec)
    # Create a result_handle to fetch data from the OPX
    res_handles = job.result_handles
    # Wait until the program is done
    res_handles.wait_for_all_values()
    # Fetch results
    I = res_handles.get("I").fetch_all()
    Q = res_handles.get("Q").fetch_all()
    # Data analysis
    s = I + 1j * Q
    idx = np.argmin(np.abs(s))
    # Plot
    fig, ax = plt.subplots(2, 1)
    ax[0].plot((resonator_LO + freqs) / u.MHz, np.abs(s))
    ax[0].set_ylabel("Amp (V)")
    ax[1].plot((resonator_LO + freqs) / u.MHz, detrend(np.unwrap(np.angle(s))))
    ax[1].set_ylabel("Phase (rad)")
    ax[1].set_xlabel("Freq (MHz)")
    ax[1].get_shared_x_axes().join(ax[0], ax[1])

    print(f"IF freq at resonance: {freqs[idx]*1e-6} MHz")
    plt.suptitle(f"resonator: {(resonator_LO + freqs[idx])/ u.MHz} MHz (IF={freqs[idx]*1e-6} MHz)")
    plt.tight_layout()
    plt.show()
