"""

"""


# %% {Imports}
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM

from quam_libs.macros import qua_declaration

from quam_libs.lib.plot_utils import plot_samples
from quam_libs.lib.save_utils import fetch_results_as_xarray, get_node_id

from qualang_tools.results import progress_counter, fetching_tool

from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Optional, List
import matplotlib.pyplot as plt


from quam.components.pulses import WaveformPulse


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    waveform_I: List[float] = [1.0] * 100
    waveform_Q: Optional[List[float]] = [0.5] * 100
    num_averages: int = 500
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True


node = QualibrationNode(name="wave_form_drive", parameters=Parameters())
node_id = get_node_id()


# %% {add wavefrom operation}

import os
import json

# Load state.json and wiring.json from the specified environment path
quam_state_folder_path = os.environ["QUAM_STATE_PATH"]

with open(os.path.join(quam_state_folder_path, "state.json"), "r") as f:
    state_data = json.load(f)

with open(os.path.join(quam_state_folder_path, "wiring.json"), "r") as f:
    wiring_data = json.load(f)

# Combine the state and wiring data into a single dictionary
quam_state = {**state_data, **wiring_data}

pulse = WaveformPulse(waveform_I=node.parameters.waveform_I, waveform_Q=node.parameters.waveform_Q)

active_qubits_names = quam_state["active_qubit_names"]

# If no qubits are specified in the node parameters, use the active qubits
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits_names = active_qubits_names

for q in qubits_names:
    quam_state["qubits"][q]["xy"]["operations"]["waveform"] = {"waveform_I": node.parameters.waveform_I, 
                                                               "waveform_Q": node.parameters.waveform_Q,
                                                               "__class__": "quam.components.pulses.WaveformPulse"}


# %% {Initialize_QuAM}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load(quam_state)

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# %% 
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
flux_point = "joint"

with program() as waveform_drive:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point        
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            qubit.xy.play("waveform")
            qubit.align()

            # readout the resonator
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
            # Wait for the qubit to decay to the ground state
            qubit.resonator.wait(machine.depletion_time * u.ns)
            # save data
            save(I[i], I_st[i])
            save(Q[i], Q_st[i])

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].average().save(f"I{i + 1}")
            Q_st[i].average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, waveform_drive, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(waveform_drive)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Plot and save is simulation}
if node.parameters.simulate:
    fig = plot_samples(samples, [qubit.name for qubit in qubits])
    node.results["figure"] = fig
    node.machine = machine
    node.save()

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, qubits)
        
    node.results = {"ds": ds}

    

        # # %% {Save_results}
        # node.outcomes = {q.name: "successful" for q in qubits}
        # node.results["initial_parameters"] = node.parameters.model_dump()
        # node.machine = machine
        # save_node(node)


# %%
