"""
power_rabi.py: performs power rabi
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
experiment = "power_rabi"
debug = True
simulate = False
fit_data = True
qubit_list = [0, 1]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")

machine.qubits[0].driving.drag_cosine.angle2volt.deg180 = 0.4
machine.qubits[1].driving.drag_cosine.angle2volt.deg180 = 0.4
config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 4e3

cooldown_time = 5 * u.us // 4

a_min = 0.2
a_max = 1
da = 0.01

amps = np.arange(a_min, a_max + da / 2, da)

with program() as power_rabi:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    a = declare(fixed)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]
    b = declare(fixed)

    for i in range(len(qubit_list)):
        # bring other qubits to zero frequency
        machine.nullify_qubits(True, qubit_list, i)
        set_dc_offset(
            machine.qubits[i].name + "_flux", "single", machine.get_flux_bias_point(i, "near_anti_crossing").value
        )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(a, amps)):
                play("x180" * amp(a), machine.qubits[i].name)
                align()
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
            I_st[i].buffer(len(amps)).average().save(f"I{i}")
            Q_st[i].buffer(len(amps)).average().save(f"Q{i}")
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
    job = qmm.simulate(config, power_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(power_rabi)

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
                plt.subplot(211)
                plt.cla()
                fit_I = Fit.rabi(
                    amps * machine.qubits[q].driving.drag_cosine.angle2volt.deg180, qubit_data[q]["I"], plot=debug
                )
                plt.subplot(212)
                plt.cla()
                fit_Q = Fit.rabi(
                    amps * machine.qubits[q].driving.drag_cosine.angle2volt.deg180, qubit_data[q]["I"], plot=debug
                )
            # live plot
            if debug and not fit_data:
                plot_demodulated_data_1d(
                    amps * machine.qubits[q].driving.drag_cosine.angle2volt.deg180,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "x180 amplitude [V]",
                    f"Power rabi {q}",
                    amp_and_phase=False,
                    fig=fig,
                    plot_options={"marker": "."},
                )

        # Update state with new resonance frequency
        if fit_data:
            print(f"Previous x180 amplitude: {machine.qubits[q].driving.drag_cosine.angle2volt.deg180:.1f} V")
            machine.qubits[q].driving.drag_cosine.angle2volt.deg180 = np.round(fit_I["amp"][0])
            print(f"New x180 amplitude: {machine.qubits[q].driving.drag_cosine.angle2volt.deg180:.1f} V")

machine.save("./labnotebook/state_after_" + experiment + "_" + now + ".json")
machine.save("latest_quam.json")
