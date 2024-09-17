from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration_OPX1000 import *
from qualang_tools.results import fetching_tool
from qualang_tools.addons.variables import assign_variables_to_element
import matplotlib.pyplot as plt
import numpy as np
import plotly.io as pio
pio.renderers.default='browser'

# get the config
config = get_config(sampling_rate=1e9)

###################
# The QUA program #
###################
total_integration_time = 2  # in seconds
n_readout = int(total_integration_time * 1e9 / readout_len / 2)

with program() as continuous_demodulation:
    n = declare(int)
    I = [declare(fixed) for _ in range(2)]
    Q = [declare(fixed) for _ in range(2)]
    dc_signal = [declare(fixed) for _ in range(2)]
    I_st = [declare_stream() for _ in range(2)]
    Q_st = [declare_stream() for _ in range(2)]
    dc_signal_st = [declare_stream() for _ in range(2)]

    ind1 = declare(int)
    ind2 = declare(int)
    assign_variables_to_element("dc_readout_element", dc_signal[0])

    # Play a long pulse to record
    # wait(100 * u.ms, "lf_element_1")
    update_frequency("lf_element_1", 1_000)
    play("const", "lf_element_1", duration=1 * u.s)

    # 1st readout
    with for_(ind1, 0, ind1 < n_readout, ind1 + 1):  # The averaging loop
        measure(
            "readout"*amp(0), "dc_readout_element", None,
            integration.full("const", dc_signal[0], "out1"),
        )
        wait(readout_len * u.ns, "dc_readout_element")
        save(dc_signal[0], dc_signal_st[0])

    # 2nd readout
    wait(readout_len * u.ns+4 + 72//4,  "dc_readout_element_twin")  # Why such a gap on the wf report?
    with for_(ind2, 0, ind2 < n_readout, ind2 + 1):  # The averaging loop
        measure(
            "readout"*amp(0), "dc_readout_element_twin", None,
            integration.full("const", dc_signal[1], "out1"),
        )
        wait(readout_len * u.ns, "dc_readout_element_twin")
        save(dc_signal[1], dc_signal_st[1])

    with stream_processing():
        dc_signal_st[0].buffer(n_readout).save(f"dc_signal1")
        dc_signal_st[1].buffer(n_readout).save(f"dc_signal2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, continuous_demodulation, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    # Get the waveform report
    samples = job.get_simulated_samples()
    waveform_report = job.get_simulated_waveform_report()
    waveform_report.create_plot(samples, plot=True, save_path=None)
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it - Execute does not block python!
    job = qm.execute(continuous_demodulation)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["dc_signal1", "dc_signal2"], mode="live")
    # Fetch results
    dc1, dc2 = results.fetch_all()
    dc1, dc2 = u.demod2volts(dc1, readout_len), u.demod2volts(dc2, readout_len)
    # Concatenate the results
    data = [val for pair in zip(dc1, dc2) for val in pair]
    time = np.arange(0, 2 * n_readout * readout_len * 1e-9, readout_len * 1e-9)
    fft = np.abs(np.fft.fft(data))
    fftfreq = np.fft.fftfreq(len(data), d=readout_len*1e-9)
    # Plot the data
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(time, data)
    plt.ylabel("Signal")
    plt.xlabel("Time [s]")
    plt.subplot(212)
    plt.plot(fftfreq[:len(fft)//2],fft[:len(fft)//2])
    plt.ylabel("FFT Amplitude")
    plt.xlabel("Frequency [Hz]")
    plt.tight_layout()

    data = {
        "time": time,
        "data": data,
        "figure": fig,
        "config": config,
    }
    save_data(data, "continuous_measurement")

