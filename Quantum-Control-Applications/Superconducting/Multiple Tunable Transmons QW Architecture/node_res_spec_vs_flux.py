# ==================== DEFINE NODE ====================
import nodeio

nodeio.context(name="res_spec_flux", description="flux map of resonator spec")

inputs = nodeio.Inputs()
inputs.stream("state", units="JSON", description="boostrap state")
inputs.stream(
    "resources",
    units="list",
    description="contains a list of the digital outputs and qubits to be used for this experiment, e.g. [[2], [1, 5, 6]]",
)
inputs.stream("debug", units="boolean", description="triggers live plot visualization for debug purposes")
inputs.stream("gate_shape", units="str", description="gate shape to be used during experiment, e.g., drag_gaussian")


outputs = nodeio.Outputs()
outputs.define("state", units="JSON", description="state with updated resonance frequencies")

nodeio.register()

# ==================== DRY RUN DATA ====================

# set inputs data for dry-run of the node
inputs.set(state="quam_bootstrap_state.json")
inputs.set(resources=[[], [0, 1]])
inputs.set(debug=False)
inputs.set(gate_shape="drag_cosine")

# =============== RUN NODE STATE MACHINE ===============
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from utils import convert_to_bool

while nodeio.status.active:

    state = inputs.get("state")
    resources = inputs.get("resources")
    digital = resources[0]
    qubit_list = resources[1]
    debug = convert_to_bool(str(inputs.get("debug")))
    gate_shape = str(inputs.get("gate_shape"))

    ##################
    # State and QuAM #
    ##################
    simulate = False
    machine = QuAM("quam_bootstrap_state.json")
    config = machine.build_config(digital, qubit_list, gate_shape)

    ###################
    # The QUA program #
    ###################
    u = unit()

    n_avg = 4e2

    cooldown_time = 5 * u.us // 4

    f_min = [-70e6, -110e6, -170e6, -210e6]
    f_max = [-40e6, -80e6, -120e6, -180e6]
    df = 0.05e6

    bias_min = [-0.4, -0.4]
    bias_max = [0.4, 0.4]
    dbias = 0.05

    freqs = [np.arange(f_min[i], f_max[i] + 0.1, df) for i in range(len(qubit_list))]
    bias = [np.arange(bias_min[i], bias_max[i] + dbias / 2, dbias) for i in range(len(qubit_list))]

    with program() as resonator_spec:
        n = [declare(int) for _ in range(len(qubit_list))]
        n_st = [declare_stream() for _ in range(len(qubit_list))]
        f = declare(int)
        I = [declare(fixed) for _ in range(len(qubit_list))]
        Q = [declare(fixed) for _ in range(len(qubit_list))]
        I_st = [declare_stream() for _ in range(len(qubit_list))]
        Q_st = [declare_stream() for _ in range(len(qubit_list))]
        b = declare(fixed)

        for i in range(len(qubit_list)):
            with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
                with for_(b, bias_min[i], b < bias_max[i] + dbias / 2, b + dbias):
                    set_dc_offset(machine.qubits[i].name + "_flux", "single", b)
                    wait(250)  # wait for 1 us
                    with for_(
                        f, f_min[i], f <= f_max[i], f + df
                    ):  # Notice it's <= to include f_max (This is only for integers!)
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
                I_st[i].buffer(len(freqs[i])).buffer(len(bias[i])).average().save(f"I{i}")
                Q_st[i].buffer(len(freqs[i])).buffer(len(bias[i])).average().save(f"Q{i}")
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
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
        for q in range(len(qubit_list)):
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
                # live plot
                if debug:
                    plt.subplot(2, len(qubit_list), 1 + q)
                    plt.cla()
                    plt.title(f"resonator spectroscopy qubit {q}")
                    plt.pcolor(freqs[q] / u.MHz, bias[q], np.sqrt(qubit_data[q]["I"] ** 2 + qubit_data[q]["Q"] ** 2))
                    plt.xlabel("frequency [MHz]")
                    plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
                    plt.subplot(2, len(qubit_list), len(qubit_list) + 1 + q)
                    plt.cla()
                    phase = signal.detrend(np.unwrap(np.angle(qubit_data[q]["I"] + 1j * qubit_data[q]["Q"])))
                    plt.pcolor(freqs[q] / u.MHz, bias[q], phase)
                    plt.xlabel("frequency [MHz]")
                    plt.ylabel("Phase [rad]")
                    plt.pause(0.1)
                    plt.tight_layout()

        # do data analysis
        # choose flux nearby crossing to achieve large chi for better discrimination

        for i in range(len(machine.qubits)):
            machine.readout_resonators[i].f_res = machine.readout_resonators[i].f_res - 50e6

        outputs.set(state=machine._json)

        machine.save("quam_1108_calibration.json")

    nodeio.terminate_workflow()
