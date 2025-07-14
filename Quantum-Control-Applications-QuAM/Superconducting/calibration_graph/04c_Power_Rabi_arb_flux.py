"""
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and set the desired pi pulse duration (qubit spectroscopy).
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse amplitude in the state.
    - Save the current state
"""


# %% {Imports}
from datetime import datetime
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id
from quam_libs.lib.fit import fit_oscillation, oscillation
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 50
    operation_x180_or_any_90: str = "x180_Gaussian"
    min_amp_factor: float = 0.
    max_amp_factor: float = 2.0
    amp_factor_step: float = 0.1
    flux_pulse_amplitude: float = 0.05
    frequency_span_in_mhz: float = 200
    frequency_step_in_mhz: float = 5
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    state_discrimination: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True

node = QualibrationNode(name="04c_Power_Rabi_arb_flux", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"
state_discrimination = node.parameters.state_discrimination
operation = node.parameters.operation_x180_or_any_90  # The qubit operation to play
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32)
center_freqs = [q.freq_vs_flux_01_quad_term * node.parameters.flux_pulse_amplitude**2 for q in qubits]

with program() as power_rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    if state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_stream = [declare_stream() for _ in range(num_qubits)]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    df = declare(int)  # QUA variable for the qubit frequency

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubits[0])
    
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                
                with for_(*from_array(a, amps)):
                    # Initialize the qubits
                    if reset_type == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)
                    qubit.align()
                    qubit.z.play("const", amplitude_scale=node.parameters.flux_pulse_amplitude / qubit.z.operations["const"].amplitude, duration=500)
                    qubit.xy.wait(400)
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + center_freqs[i])
                    # Loop for error amplification (perform many qubit pulses)
                    qubit.xy.play(operation, amplitude_scale=a)
                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                    qubit.align()
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    if state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_stream[i])
                    else:
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_stream[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"state{i + 1}")



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, power_rabi, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(power_rabi)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": amps, "freqs": dfs})
        # Add the qubit pulse absolute amplitude to the dataset
        ds = ds.assign_coords(
        {
            "abs_amp": (
                ["qubit", "amp"],
                np.array([q.xy.operations[operation].amplitude * amps for q in qubits]),
            )
            }
        )
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}


    ds.state.plot()




    # %% {Data_analysis}
    
    # # %% {Update_state}
    # if node.parameters.load_data_id is None:
    #     with node.record_state_updates():
    #         for q in qubits:
    #             q.xy.operations[operation].amplitude = fit_results[q.name]["Pi_amplitude"]

        # # %% {Save_results}
        # node.outcomes = {q.name: "successful" for q in qubits}
        # node.results["initial_parameters"] = node.parameters.model_dump()
        # node.machine = machine
        # node.save()

# %%
