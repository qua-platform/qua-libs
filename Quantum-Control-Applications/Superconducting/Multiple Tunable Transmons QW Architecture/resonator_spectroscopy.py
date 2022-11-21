"""
resonator_spec.py: performs the 1D resonator spectroscopy
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from datetime import datetime

##################
# State and QuAM #
##################
experiment = "res_spec"
debug = True
simulate = False
fit_data = True
qubit_list = [0, 1]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")

machine.readout_lines[0].lo_freq = 6.5e9
machine.readout_resonators[0].f_res = 6.6457e9
machine.readout_resonators[1].f_res = 6.7057e9

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 4e3
cooldown_time = 5 * u.us // 4

span = 50e6
df = 0.5e6
freq = [np.arange(machine.get_readout_IF(i) - span, machine.get_readout_IF(i) + span + df / 2, df) for i in qubit_list]

with program() as resonator_spec:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    f = declare(int)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]

    for i in range(len(qubit_list)):
        # bring other qubits than `i` to zero frequency
        machine.nullify_qubits(True, qubit_list, i)
        # set qubit frequency to working point
        set_dc_offset(machine.qubits[i].name + "_flux", "single", machine.get_flux_bias_point(i, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(f, freq[i])):
                update_frequency(machine.readout_resonators[i].name, f)
                measure(
                    "readout",
                    machine.readout_resonators[i].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[i]),
                )
                wait(cooldown_time, machine.readout_resonators[i].name)
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(len(qubit_list)):
            I_st[i].buffer(len(freq[i])).average().save(f"I{i}")
            Q_st[i].buffer(len(freq[i])).average().save(f"Q{i}")
            n_st[i].save(f"iteration{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    # Create the fitting object
    Fit = fitting.Fit()

    for q in range(len(qubit_list)):
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[q]["I"] = data[0]
            qubit_data[q]["Q"] = data[1]
            qubit_data[q]["iteration"] = data[2]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_avg, start_time=my_results.start_time)
            # Fitting
            if fit_data:
                fit = Fit.transmission_resonator_spectroscopy(
                    freq[q] / u.MHz, signal.detrend(np.unwrap(np.angle(qubit_data[q]["I"] + 1j * qubit_data[q]["Q"])))
                )
            # live plot
            if debug:
                plot_demodulated_data_1d(
                    freq[q] / u.MHz,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "frequency [MHz]",
                    f"resonator spectroscopy qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"marker": "."},
                )

        # Update state with new resonance frequency
        if fit_data:
            print(f"Previous resonance frequency: {machine.readout_resonators[q].f_res:.1f} Hz")
            machine.readout_resonators[q].f_res = (
                np.round(fit["f"][0] * 1e6)
                + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq
            )
            machine.readout_resonators[q].f_opt = machine.readout_resonators[q].f_res
            print(f"New resonance frequency: {machine.readout_resonators[q].f_res:.1f} Hz")

machine.save("./labnotebook/state_after_" + experiment + "_" + now + ".json")
machine.save("latest_quam.json")
