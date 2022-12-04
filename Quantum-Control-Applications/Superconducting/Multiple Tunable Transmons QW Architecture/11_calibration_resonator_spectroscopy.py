"""
Perform the 1D resonator spectroscopy
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array

##################
# State and QuAM #
##################
experiment = "1D_resonator_spectroscopy"
debug = True
simulate = False
fit_data = False
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

# machine.readout_lines[0].lo_freq = 6.0e9
# machine.readout_lines[0].lo_power = 13
# machine.readout_lines[0].length = 3e-6
# machine.readout_resonators[0].readout_amplitude =0.005
# machine.readout_resonators[0].f_opt = 6.131e9
# machine.readout_resonators[0].f_opt = machine.readout_resonators[0].f_res
config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 4e3

span = 2e6
df = 0.01e6
freq = [np.arange(machine.get_readout_IF(i) - span, machine.get_readout_IF(i) + span + df / 2, df) for i in qubit_list]

with program() as resonator_spec:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    f = declare(int)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]

    for q in range(len(qubit_list)):
        if not simulate:
            cooldown_time = machine.readout_resonators[q].relaxation_time // 4
        else:
            cooldown_time = 16
        # bring other qubits than `q` to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        # set qubit frequency to working point
        set_dc_offset(machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "working_point").value)

        with for_(n[q], 0, n[q] < n_avg, n[q] + 1):
            with for_(*from_array(f, freq[q])):
                update_frequency(machine.readout_resonators[q].name, f)
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[q]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[q]),
                    # demod.full("cos", I[q], "out1"),
                    # demod.full("sin", Q[q], "out1"),
                )
                wait(cooldown_time, machine.readout_resonators[q].name)
                save(I[q], I_st[q])
                save(Q[q], Q_st[q])
            save(n[q], n_st[q])

        align()

    with stream_processing():
        for q in range(len(qubit_list)):
            I_st[q].buffer(len(freq[q])).average().save(f"I{q}")
            Q_st[q].buffer(len(freq[q])).average().save(f"Q{q}")
            n_st[q].save(f"iteration{q}")

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
    figures = []
    for q in range(len(qubit_list)):
        # Live plotting
        if debug:
            fig = plt.figure()
            figures.append(fig)
            interrupt_on_close(fig, job)
        print(f"Qubit {q}")
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
            # live plot
            if debug:
                plot_demodulated_data_1d(
                    freq[q] + machine.readout_lines[0].lo_freq,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "frequency [Hz]",
                    f"resonator spectroscopy qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"marker": "."},
                )
            # Fitting
            if fit_data:
                try:
                    Fit.reflection_resonator_spectroscopy(
                        freq[q] + machine.readout_lines[0].lo_freq,
                        np.sqrt(qubit_data[q]["I"] ** 2 + qubit_data[q]["Q"] ** 2),
                        plot=False,
                    )
                    plt.subplot(211)
                    plt.cla()
                    fit = Fit.reflection_resonator_spectroscopy(
                        freq[q] + machine.readout_lines[0].lo_freq,
                        np.sqrt(qubit_data[q]["I"] ** 2 + qubit_data[q]["Q"] ** 2),
                        plot=True,
                    )
                    plt.pause(0.1)
                except (Exception,):
                    pass

        # Update state with new resonance frequency
        if fit_data:
            print(f"Previous resonance frequency: {machine.readout_resonators[q].f_res * 1e-9:.6f} GHz")
            machine.readout_resonators[q].f_res = np.round(fit["f"][0])
            machine.readout_resonators[q].f_opt = machine.readout_resonators[q].f_res
            print(f"New resonance frequency: {machine.readout_resonators[q].f_res * 1e-9:.6f} GHz")
            print(f"New resonance IF frequency: {machine.get_readout_IF(q) * 1e-6:.3f} MHz")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
