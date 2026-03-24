import numpy as np
import matplotlib.pyplot as plt

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from qualang_tools.units import unit

from configuration import *

u = unit(coerce_to_integer=True)

def update_readout_length(config, element, operation, new_readout_length):
    assert element in config["elements"], "The element must be in the config."
    assert operation in config["elements"][element]["operations"], "The operation must be one of {}".format(config["elements"]["operations"])
    assert new_readout_length % 4 == 0, "The new readout length must be a multiple of 4ns."

    pulse = config["elements"][element]["operations"][operation]
    config["pulses"][pulse]["length"] = new_readout_length
    weights = config["pulses"][pulse]["integration_weights"]
    for weight, value in weights.items():
        for quadrature in ["cosine", "sine"]:
            assert len(config["integration_weights"][value][
                           quadrature]) == 1, "This function only works for constant integration weights."
            iw = config["integration_weights"]["cosine_weights"][quadrature][0][0]
            config["integration_weights"][value][quadrature] = [(iw, new_readout_length)]

###################
# The QUA program #
###################
n_avg = 100
total_integration_time = 50 * u.ms
sampling_rate = 1e6
new_readout_length = int(1 / (sampling_rate * 1e-9) * 4) // 4
n_readout = int(total_integration_time / new_readout_length / 2)

update_readout_length(config, "resonator", "readout", new_readout_length)
update_readout_length(config, "resonator_twin", "readout", new_readout_length)

with program() as prog:
    I = [declare(fixed) for _ in range(2)]
    Q = [declare(fixed) for _ in range(2)]
    I_st = declare_stream()
    Q_st = declare_stream()

    n = declare(int)
    ind1 = declare(int)
    ind2 = declare(int)
    with for_(n, 0, n < n_avg, n + 1):
        # 1st readout
        with for_(ind1, 0, ind1 < n_readout, ind1 + 1):
            measure(
                "readout", "resonator",
                demod.full("cos", I[0], "out1"), demod.full("sin", Q[0], "out1"))
            wait(readout_len * u.ns, "resonator")
            save(I[0], I_st)
            save(Q[0], Q_st)

        # 2nd readout
        wait(readout_len * u.ns,  "resonator_twin")
        with for_(ind2, 0, ind2 < n_readout, ind2 + 1):
            measure(
                "readout", "resonator_twin",
                demod.full("cos", I[1], "out1"), demod.full("sin", Q[1], "out1"))
            wait(readout_len * u.ns, "resonator_twin")
            save(I[1], I_st)
            save(Q[1], Q_st)

    with stream_processing():
        I_st.buffer(n_readout*2).average().save("I")
        Q_st.buffer(n_readout*2).average().save("Q")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host="127.0.0.1", cluster_name="Cluster_1")

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, prog, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True)
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(prog)
    # Creates a result handle to fetch data from the OPX
    res_handles = job.result_handles
    # Waits (blocks the Python console) until all results have been acquired
    results = res_handles.fetch_results(wait_until_done=True, timeout=60, stream_names=["I", "Q"])
    # Convert data to V
    I = u.demod2volts(results.get("I"), readout_len, single_demod=True)
    Q = u.demod2volts(results.get("Q"), readout_len, single_demod=True)
    # Plots results
    time = np.linspace(0, total_integration_time*1e-9, 2*n_readout)
    plt.figure()
    plt.plot(time, np.sqrt(I**2 + Q**2))
    plt.xlabel("time (s)")
    plt.ylabel("amplitude (V)")
    plt.show()
