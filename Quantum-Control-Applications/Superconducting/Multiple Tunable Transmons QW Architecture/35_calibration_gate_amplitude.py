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


##################
# State and QuAM #
##################
gate = "x90"
experiment = gate + "_amplitude_calibration"
debug = True
simulate = False
qubit_list = [0]
digital = []
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

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 1e3

# Amplitude scan
a_min = 0.75
a_max = 1.25
da = 0.01
amps = np.arange(a_min, a_max + da / 2, da)

# Number of pulses scan
n_pulse_max = 100
pulse_vec = np.arange(0, n_pulse_max + 1, iteration_step)

with program() as gate_cal:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    a = declare(fixed)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]
    b = declare(fixed)
    it = declare(int)
    pulses = declare(int)
    n_pulse = declare(int)
    state = [declare(bool) for _ in range(len(qubit_list))]
    state_st = [declare_stream() for _ in range(len(qubit_list))]

    for q in range(len(qubit_list)):
        if not simulate:
            cooldown_time = 5 * int(machine.qubits[q].t1 * 1e9) // 4
        else:
            cooldown_time = 16
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(
            machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "near_anti_crossing").value
        )

        with for_(n[q], 0, n[q] < n_avg, n[q] + 1):
            with for_(*from_array(n_pulse, pulse_vec)):
                with for_(*from_array(a, amps)):
                    with for_(pulses, 0, pulses < n_pulse, pulses + 1):
                        play(gate * amp(a), machine.qubits[q].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I[q]),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q[q]),
                    )
                    wait(cooldown_time, machine.readout_resonators[q].name)
                    align()
                    assign(state[q], I[q] > machine.readout_resonators[q].ge_threshold)
                    save(I[q], I_st[q])
                    save(Q[q], Q_st[q])
                    save(state[q], state_st[q])
                save(n[q], n_st[q])

        align()

    with stream_processing():
        for q in range(len(qubit_list)):
            I_st[q].buffer(len(amps)).buffer(len(pulse_vec)).average().save(f"I{q}")
            Q_st[q].buffer(len(amps)).buffer(len(pulse_vec)).average().save(f"Q{q}")
            state_st[q].boolean_to_int().buffer(len(amps)).buffer(len(pulse_vec)).average().save(f"state{q}")
            n_st[q].save(f"iteration{q}")

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

    for q in range(len(qubit_list)):
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"state{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()

            qubit_data[q]["I"] = data[0]
            qubit_data[q]["Q"] = data[1]
            qubit_data[q]["state"] = data[2]
            qubit_data[q]["iteration"] = data[3]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_avg, start_time=my_results.start_time)
            # live plot
            if debug:
                plot_demodulated_data_2d(
                    amps * base_amp[q],
                    pulse_vec,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "Number of pulses",
                    gate + " amplitude [V]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
        fig = plt.figure()
        figures.append(fig)
        colors = ["b", "r", "g", "m", "c"]
        i = 0
        for it in [
            1,
            int(0.25 * len(pulse_vec)),
            int(0.5 * len(pulse_vec)),
            int(0.75 * len(pulse_vec)),
            len(pulse_vec) - 1,
        ]:
            plt.subplot(211)
            plt.plot(amps * base_amp[q], qubit_data[q]["I"][it], colors[i] + "-", label=f"{it} iterations")
            plt.ylabel("I [a.u.]")
            plt.title(f"DRAG calibration for qubit {q}")
            plt.subplot(212)
            plt.plot(amps * base_amp[q], qubit_data[q]["Q"][it], colors[i] + "-", label=f"{it} iterations")
            plt.xlabel(gate + " amplitude [V]")
            plt.ylabel("Q [a.u.]")
            plt.tight_layout()
            i += 1
        plt.subplot(211)
        plt.legend(ncol=i)
        plt.subplot(212)
        plt.legend(ncol=i)

        # Update state with new DRAG coefficient
        print(f"Previous {gate} amplitude: {base_amp[q]:.1f} V")
        # Chose I, Q or amp...
        # machine.qubits[q].driving.drag_cosine.angle2volt.deg180 =
        print(f"New {gate} amplitude: {machine.get_qubit_gate(q, gate_shape).angle2volt.deg180:.1f} V")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
