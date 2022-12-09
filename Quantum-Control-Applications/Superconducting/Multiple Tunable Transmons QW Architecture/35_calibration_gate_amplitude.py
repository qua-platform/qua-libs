"""
Calibrate the single qubit gates using error amplification
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from macros import *
from config import NUMBER_OF_QUBITS_W_CHARGE

##################
# State and QuAM #
##################
gate = "x90"
experiment = gate + "_amplitude_calibration"
debug = True
simulate = False
qubit_w_charge_list = [0, 1]
qubit_wo_charge_list = [2, 3, 4, 5]
qubit_list = [0, 5]  # you can shuffle the order at which you perform the experiment
injector_list = [0, 1]
digital = [1, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

# Set gate amplitude to max
base_amp = []
iteration_step = 1
if gate == "x90":
    iteration_step = 4
    for q in qubit_list:
        base_amp.append(machine.get_qubit_gate(q, gate_shape).angle2volt.deg90)
elif gate == "x180":
    iteration_step = 2
    for q in qubit_list:
        base_amp.append(machine.get_qubit_gate(q, gate_shape).angle2volt.deg180)

config = machine.build_config(digital, qubit_w_charge_list, qubit_wo_charge_list, injector_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 1e3

# Amplitude scan
a_min = 0.9
a_max = 1.1
da = 0.01
amps = np.arange(a_min, a_max + da / 2, da)

# Number of pulses scan
n_pulse_max = 100
pulse_vec = np.arange(0, n_pulse_max + 1, iteration_step)

with program() as gate_cal:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    a = declare(fixed)
    it = declare(int)
    pulses = declare(int)
    n_pulse = declare(int)
    state = [declare(bool) for _ in range(len(qubit_list))]
    state_st = [declare_stream() for _ in range(len(qubit_list))]

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        if q in qubit_w_charge_list:
            set_dc_offset(machine.qubits[q].name + "_charge", "single", machine.get_charge_bias_point(q, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(n_pulse, pulse_vec)):
                with for_(*from_array(a, amps)):
                    with for_(pulses, 0, pulses < n_pulse, pulses + 1):
                        if q in qubit_w_charge_list:
                            play(gate * amp(a), machine.qubits[q].name)
                        else:
                            play(gate * amp(a), machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("cos", I[i], "out1"),
                        demod.full("sin", Q[i], "out1"),
                    )
                    wait_cooldown_time_fivet1(q, machine, simulate, qubit_w_charge_list)
                    align()
                    assign(state[i], I[i] > machine.readout_resonators[q].ge_threshold)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    save(state[i], state_st[i])
                save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(len(amps)).buffer(len(pulse_vec)).average().save(f"I{q}")
            Q_st[i].buffer(len(amps)).buffer(len(pulse_vec)).average().save(f"Q{q}")
            state_st[i].boolean_to_int().buffer(len(amps)).buffer(len(pulse_vec)).average().save(f"state{q}")
            n_st[i].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=10000)
    job = qmm.simulate(config, gate_cal, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(gate_cal)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    # Create the fitting object
    Fit = fitting.Fit()

    for i, q in enumerate(qubit_list):
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        print("Qubit " + str(q))
        qubit_data[i]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"state{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[i]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()

            qubit_data[i]["I"] = data[0]
            qubit_data[i]["Q"] = data[1]
            qubit_data[i]["state"] = data[2]
            qubit_data[i]["iteration"] = data[3]
            # Progress bar
            progress_counter(qubit_data[i]["iteration"], n_avg, start_time=my_results.start_time)
            # live plot
            if debug:
                plot_demodulated_data_2d(
                    amps * base_amp[i],
                    pulse_vec,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    gate + " amplitude [V]",
                    "Number of pulses",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
        fig = plt.figure()
        figures.append(fig)
        colors = ["b", "r", "g", "m", "c"]
        j = 0
        for it in [1, int(0.25 * len(pulse_vec)), int(0.5 * len(pulse_vec)), int(0.75 * len(pulse_vec)), len(pulse_vec) - 1]:
            plt.subplot(211)
            plt.plot(amps * base_amp[i], qubit_data[i]["I"][it], colors[j] + "-", label=f"{pulse_vec[it]} iterations")
            plt.ylabel("I [a.u.]")
            plt.title(f"Amplitude calibration for qubit {q}")
            plt.subplot(212)
            plt.plot(amps * base_amp[i], qubit_data[i]["Q"][it], colors[j] + "-", label=f"{pulse_vec[it]} iterations")
            plt.xlabel(gate + " amplitude [V]")
            plt.ylabel("Q [a.u.]")
            plt.tight_layout()
            j += 1
        plt.subplot(211)
        plt.legend(ncol=j)
        plt.subplot(212)
        plt.legend(ncol=j)

        # Update state with new amplitude
        print(f"Previous {gate} amplitude: {base_amp[i]:.1f} V")
        # Chose I, Q or amp...
        # machine.qubits[q].driving.drag_cosine.angle2volt.deg180 =
        print(f"New {gate} amplitude: {machine.get_qubit_gate(q, gate_shape).angle2volt.deg180:.1f} V")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
