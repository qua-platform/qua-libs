"""
DRAG_cal.py: performs power drag calibration
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from datetime import datetime

##################
# State and QuAM #
##################
gate = "x90"
experiment = gate + "_cal"
debug = True
simulate = False
qubit_list = [0, 1]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")

# Set gate amplitude to max
base_amp = []
if gate == "x90":
    for i in qubit_list:
        machine.qubits[i].driving.drag_cosine.angle2volt.deg90 = 0.3
        base_amp.append(machine.qubits[i].driving.drag_cosine.angle2volt.deg90)
elif gate == "x180":
    for i in qubit_list:
        machine.qubits[i].driving.drag_cosine.angle2volt.deg180 = 0.49
        base_amp.append(machine.qubits[i].driving.drag_cosine.angle2volt.deg180)

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 4e3

cooldown_time = 5 * u.us // 4

a_min = 0.2
a_max = 1
da = 0.05
amps = np.arange(a_min, a_max + da / 2, da)

n_pulse_max = 13

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

    for i in range(len(qubit_list)):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, i)
        set_dc_offset(
            machine.qubits[i].name + "_flux", "single", machine.get_flux_bias_point(i, "near_anti_crossing").value
        )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(n_pulse, 1, n_pulse < n_pulse_max + 1, n_pulse + 1):
                with for_(*from_array(a, amps)):
                    with for_(pulses, 0, pulses < n_pulse, pulses + 1):
                        play(gate * amp(a), machine.qubits[i].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[i].name,
                        None,
                        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I[i]),
                        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q[i]),
                    )
                    wait(cooldown_time, machine.readout_resonators[i].name)
                    assign(state[i], I[i] > machine.readout_resonators[i].ge_threshold)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    save(state[i], state_st[i])
                save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(len(qubit_list)):
            I_st[i].buffer(n_pulse_max).buffer(len(amps)).average().save(f"I{i}")
            Q_st[i].buffer(n_pulse_max).buffer(len(amps)).average().save(f"Q{i}")
            state_st[i].boolean_to_int().buffer(n_pulse_max).buffer(len(amps)).average().save(f"state{i}")
            n_st[i].save(f"iteration{i}")

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
                    np.arange(1, n_pulse_max + 1),
                    amps * base_amp[q],
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "Number of pulses",
                    gate + " amplitude [V]",
                    f"{gate} amp calibration for qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
        fig = plt.figure()
        colors = ["b", "r", "g", "m", "c"]
        i = 0
        for it in [1, int(0.25 * n_pulse_max), int(0.5 * n_pulse_max), int(0.75 * n_pulse_max), n_pulse_max - 1]:
            plt.subplot(211)
            plt.plot(amps * base_amp[q], qubit_data[q]["I"][:, it], colors[i] + ".", label=f"{it} iterations")
            plt.ylabel("I [a.u.]")
            plt.title(f"DRAG calibration for qubit {q}")
            plt.subplot(212)
            plt.plot(amps * base_amp[q], qubit_data[q]["Q"][:, it], colors[i] + ".", label=f"{it} iterations")
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
        print(f"New {gate} amplitude: {machine.qubits[q].driving.drag_cosine.angle2volt.deg180:.1f} V")

machine.save("./lab_notebook/state_after_" + experiment + "_" + now + ".json")
machine.save("latest_quam.json")
