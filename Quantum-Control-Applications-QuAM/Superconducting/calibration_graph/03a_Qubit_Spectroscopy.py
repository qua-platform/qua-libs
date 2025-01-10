"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate frequencies dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly from the node parameters.

The data is post-processed to determine the qubit resonance frequency and the width of the peak.

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Set the flux bias to the desired working point, independent, joint or arbitrary, in the state.
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.

Before proceeding to the next node:
    - Update the qubit frequency in the state, as well as the expected x180 amplitude and IQ rotation angle.
    - Save the current state
"""

# %% {Imports}
from qualibrate import QualibrationNode

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.macros import qua_declaration
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.experiments.execution import print_progress_bar
from quam_libs.experiments.Qubit_Spectroscopy.parameters import Parameters
from quam_libs.experiments.Qubit_Spectroscopy.node import get_optional_parameters
from quam_libs.experiments.Qubit_Spectroscopy.analysis import fetch_dataset, fit_qubits
from quam_libs.experiments.Qubit_Spectroscopy.plotting import plot_qubit_response

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np
from qm.qua import *


# %% {Node_parameters}
node = QualibrationNode(
    name="03a_Qubit_Spectroscopy",
    parameters=Parameters(
        qubits=None,
        num_averages=500,
        operation="saturation",
        operation_amplitude_factor=0.1,
        operation_len_in_ns=None,
        frequency_span_in_mhz=100,
        frequency_step_in_mhz=0.25,
        flux_point_joint_or_independent="independent",
        target_peak_width=2e6,
        arbitrary_flux_bias=None,
        arbitrary_qubit_frequency_in_ghz=None,
        simulate=False,
        simulation_duration_ns=2500,
        timeout=100,
        load_data_id=None,
        multiplexed=False,
    ),
)


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Get the relevant QuAM components
qubits = machine.get_qubits_used_in_node(node)
resonators = machine.get_resonators_used_in_node(node)
num_qubits = len(qubits)
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()


# %% {QUA_program}
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
# Get the optional parameters
qubit_pulse_duration, arb_flux_bias_offset, detuning = get_optional_parameters(qubits, node.parameters)

with program() as qua_prog:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)  # QUA variable for the qubit frequency

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=node.parameters.flux_point_joint_or_independent, target=qubit)

        with for_(n, 0, n < node.parameters.num_averages, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detuning[qubit.name])
                qubit.align()
                # Bring the qubit to the desired point during the saturation pulse
                qubit.z.play(
                    "const",
                    amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude,
                    duration=qubit_pulse_duration[qubit.name] * u.ns,
                )
                # Play the saturation pulse
                qubit.xy.wait(qubit.z.settle_time * u.ns)
                qubit.xy.play(
                    node.parameters.operation,
                    amplitude_scale=node.parameters.operation_amplitude_factor,
                    duration=qubit_pulse_duration[qubit.name] * u.ns,
                )
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
            I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate and node.parameters.load_data_id is None:
    fig, samples = simulate_and_plot(qmm, config, qua_prog, node.parameters)
    node.results = {"figure": fig, "samples": samples}

else:
    if node.parameters.load_data_id is None:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            print_progress_bar(job, "n", node.parameters.num_averages)

    # %% {Data_fetching_and_dataset_creation}
    if node.parameters.load_data_id is None:
        ds = fetch_dataset(job, qubits, frequencies=dfs, detuning=detuning)
        node.results = {"ds": ds}
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]

    # %% {Data_analysis}
    ds, fit_results = fit_qubits(ds, qubits, node.parameters)
    node.results["fit_results"] = fit_results

    # %% {Plotting}
    fig = plot_qubit_response(ds, qubits, fit_results["fit_ds"])
    node.results["figure"] = fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for q in qubits:
                if fit_results[q.name]["fit_successful"]:
                    if node.parameters.flux_point_joint_or_independent == "arbitrary":
                        q.arbitrary_intermediate_frequency = float(
                            fit_results[q.name]["drive_freq"]
                            + detuning[q.name]
                            + q.xy.intermediate_frequency
                            - q.xy.RF_frequency
                        )
                        q.z.arbitrary_offset = arb_flux_bias_offset[q.name]
                    else:
                        q.xy.intermediate_frequency += fit_results[q.name]["drive_freq"] - q.xy.RF_frequency
                    if not node.parameters.flux_point_joint_or_independent == "arbitrary":
                        # Update the IW angle
                        q.resonator.operations["readout"].integration_weights_angle = fit_results[q.name]["angle"]
                        # Update the saturation amplitude
                        limits = instrument_limits(q.xy)
                        if fit_results[q.name]["saturation_amplitude"] < limits.max_wf_amplitude:
                            q.xy.operations["saturation"].amplitude = fit_results[q.name]["saturation_amplitude"]
                        else:
                            q.xy.operations["saturation"].amplitude = limits.max_wf_amplitude
                        # Update the expected x180 amplitude
                        if fit_results[q.name]["x180_amplitude"] < limits.max_x180_wf_amplitude:
                            q.xy.operations["x180"].amplitude = fit_results[q.name]["x180_amplitude"]
                        else:
                            q.xy.operations["x180"].amplitude = limits.max_x180_wf_amplitude
        node.results["ds"] = ds

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()
