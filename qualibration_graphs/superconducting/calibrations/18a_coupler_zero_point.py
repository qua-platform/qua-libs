# %% {Imports}
import math
from typing import Literal, Optional, List
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array

from qualibrate import QualibrationNode, NodeParameters
from quam_config import Quam
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.coupler_zero_point import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_amplitude_with_fit,
    plot_raw_phase,
)
from qualibration_libs.parameters import get_qubit_pairs, get_qubits

from qualibration_libs.legacy.lib.pulses import FluxPulse
from qualibration_libs.legacy.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from qualibration_libs.legacy.lib.save_utils import fetch_results_as_xarray, load_dataset

from qm import SimulationConfig

from qualibration_libs.legacy.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from qualibration_libs.legacy.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
#from quam_libs.components import CZMacro

# %% {Initialisation}
description = """
        COUPLER ZERO-INTERACTION CALIBRATION
This calibration program determines the flux bias point for tunable couplers that
results in zero effective coupling (g ≈ 0) between pairs of flux-tunable qubits.
This is a crucial step for architectures relying on dynamically tunable coupling
to implement high-fidelity two-qubit gates and isolate qubits during single-qubit operations.

The method performs a 2D sweep of:
    - The coupler flux bias (around its idle point).
    - The qubit control flux (to bring qubit frequencies closer to resonance).

Each point in this sweep involves initializing the control qubit in the excited state and applying
concurrent flux pulses to both the control qubit and the coupler. The resulting excitation in the
target qubit is measured either using state discrimination or IQ integration, depending on the
configuration. The aim is to identify the coupler bias point at which the residual interaction vanishes.

From the data, the optimal coupler flux (yielding minimal excitation transfer) and corresponding
control qubit flux (yielding maximal excitation retention) are extracted. These values are used
to update the coupler’s `decouple_offset` and the estimated qubit `detuning`.

This procedure ensures precise decoupling between qubits during idle or single-qubit operations, helping
mitigate unwanted crosstalk and residual ZZ interactions.

Prerequisites:
    - Coupler hardware model with known calibration structure.
    - Qubit frequencies, flux tuning models (quadratic term at least).
    - Active reset routines for fast initialization (optional).
    - Calibrated readout and XY pulses on the control and target qubits.
    - Initial coupler `decouple_offset` set near its expected g ≈ 0 point.

State update:
    - Coupler zero-point flux: `coupler.decouple_offset`
    - Control qubit detuning: `qubit_pair.detuning`

Additional notes:
    - Supports both simulation and hardware execution.
    - Results are visualized in a 2D map with overlays for idle and calibrated zero-g coupler flux points.
    - If enabled, detuning is also plotted on a secondary axis for interpretation.

This calibration is essential for optimizing gate scheduling, minimizing idling errors,
and preparing the system for entangling gate calibration.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="18a_coupler_zero_point",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.num_averages = 2
    node.parameters.coupler_flux_min = -0.05 #relative to the coupler set point
    node.parameters.coupler_flux_max = 0.05 #relative to the coupler set point
    node.parameters.coupler_flux_step = 0.0004
    node.parameters.qubit_flux_span = 0.026 # relative to the known/calculated detuning between the qubits
    node.parameters.qubit_flux_step = 0.0002
    node.parameters.pulse_duration_ns = 232
    node.parameters.cz_or_iswap = "iswap"
    node.parameters.use_saved_detuning = False
# Instantiate the QUAM class from the state file
node.machine = Quam.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    node.namespace["qubits"] = qubits = [qp.qubit_control for qp in qubit_pairs] + [
        qp.qubit_target for qp in qubit_pairs
    ]
    num_qubits = len(qubits)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_averages  # The number of averages

    flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
    # Loop parameters
    fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max+0.0001, node.parameters.coupler_flux_step)
    fluxes_qubit = np.arange(-node.parameters.qubit_flux_span / 2, node.parameters.qubit_flux_span / 2 + 0.0001, node.parameters.qubit_flux_step)
    fluxes_qp = {}

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
         "coupler_flux": xr.DataArray(fluxes_coupler, attrs={"long_name": "coupler flux", "units": "V"}),
        "qubit_flux": xr.DataArray(fluxes_qubit, attrs={"long_name": "qubit flux", "units": "V"}),
    }
    for qp in qubit_pairs:
        # estimate the flux shift to get the control qubit to the target qubit frequency
        if node.parameters.use_saved_detuning:
            est_flux_shift = qp.detuning
        elif node.parameters.cz_or_iswap == "iswap":
            est_flux_shift = np.sqrt(-(qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency) / qp.qubit_control.freq_vs_flux_01_quad_term) #TODO: figure out how to make this run properly after filters
        elif node.parameters.cz_or_iswap == "cz":
            est_flux_shift = np.sqrt(-(qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity) / qp.qubit_control.freq_vs_flux_01_quad_term) #TODO: figure out how to make this run properly after filters
        fluxes_qp[qp.name] = fluxes_qubit + est_flux_shift
    node.namespace["fluxes_qp"] = fluxes_qp

    assert node.parameters.pulse_duration_ns % 4 == 0, \
        f"Expected pulse duration to be divisible by 4, got {node.parameters.pulse_duration_ns} ns"
    pulse_duration_ns = node.parameters.pulse_duration_ns
    reset_coupler_bias = False

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        flux_coupler = declare(float)
        flux_qubit = declare(float)
        comp_flux_qubit = declare(float)
        n_st = declare_stream()
        qua_pulse_duration = declare(int, value = pulse_duration_ns // 4)
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for qubit in node.machine.active_qubits:
            node.machine.initialize_qpu(target=qubit)
            align()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for ii,qp in multiplexed_qubit_pairs.items():
                print("qubit control: %s, qubit target: %s" % (qp.qubit_control.name, qp.qubit_target.name))
                # Bring the active qubits to the minimum frequency point
                node.machine.set_all_fluxes(flux_point, qp)
                if reset_coupler_bias:
                    qp.coupler.set_dc_offset(0.0)
                wait(1000)
                
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(flux_coupler, fluxes_coupler)):
                        with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                            # reset
                            if node.parameters.reset_type == "active":
                                active_reset_simple(qp.qubit_control)
                                active_reset_simple(qp.qubit_target)
                                qp.align()
                            else:
                                wait(qp.qubit_control.thermalization_time * u.ns)
                                wait(qp.qubit_target.thermalization_time * u.ns)
                            align()
                            if "coupler_qubit_crosstalk" in qp.extras:
                                assign(comp_flux_qubit, flux_qubit  +  qp.extras["coupler_qubit_crosstalk"] * flux_coupler )
                            else:
                                print("No crosstalk compensated")
                                assign(comp_flux_qubit, flux_qubit)
                            # setting both qubits ot the initial state
                            qp.qubit_control.xy.play("x180")
                            if node.parameters.cz_or_iswap == "cz":
                                qp.qubit_target.xy.play("x180")
                            align()
                            # wait(8)
                            qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = qua_pulse_duration)
                            qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = qua_pulse_duration)
                            align()
                            wait(20)
                            # readout
                            if node.parameters.use_state_discrimination:
                                qp.qubit_control.readout_state_gef(state_c[ii])
                                qp.qubit_target.readout_state(state_t[ii])
                                save(state_c[ii], state_c_st[ii])
                                save(state_t[ii], state_t_st[ii])
                            else:
                                qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                save(I_c[ii], I_c_st[ii])
                                save(Q_c[ii], Q_c_st[ii])
                                save(I_t[ii], I_t_st[ii])
                                save(Q_t[ii], Q_t_st[ii])
            align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_control{i}")
                    state_t_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_target{i}")
                else:
                    I_c_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i}")



# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}

# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_averages,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset

# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    node.namespace["qubits"] = [qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]
    node.namespace["qubit_pairs"] = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]

# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result["success"] else "failed")
        for qubit_pair_name, fit_result in node.results["fit_results"].items()
    }
    
# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }

#%%
"""
 {Node_parameters}
qubit_pair_indexes = [1]  # [1, 2]
class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["q%s-%s"%(i,i+1) for i in qubit_pair_indexes] # ["coupler_q1_q2"]
    num_averages: int = 500
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None

    coupler_flux_min : float = -0.05 #relative to the coupler set point
    coupler_flux_max : float = 0.05 #relative to the coupler set point

    coupler_flux_step : float = 0.0004
    qubit_flux_span : float = 0.026 # relative to the known/calculated detuning between the qubits
    qubit_flux_step : float = 0.0002
    use_state_discrimination: bool = True
    pulse_duration_ns: int = 232
    cz_or_iswap: Literal["cz", "iswap"] = "cz"
    use_saved_detuning: bool = False


node = QualibrationNode(
    name="18a_coupler_zero_point_calibration", 
    description=description,
    parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

{Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = Quam.load()


# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
# if any([qp.q1.z is None or qp.q2.z is None for qp in qubit_pairs]):
#     warnings.warn("Found qubit pairs without a flux line. Skipping")

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %%

####################
# Helper functions #
####################


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max+0.0001, node.parameters.coupler_flux_step)
fluxes_qubit = np.arange(-node.parameters.qubit_flux_span / 2, node.parameters.qubit_flux_span / 2 + 0.0001, node.parameters.qubit_flux_step)
fluxes_qp = {}
for qp in qubit_pairs:
    # estimate the flux shift to get the control qubit to the target qubit frequency
    if node.parameters.use_saved_detuning:
        est_flux_shift = qp.detuning
    elif node.parameters.cz_or_iswap == "iswap":
        est_flux_shift = np.sqrt(-(qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency) / qp.qubit_control.freq_vs_flux_01_quad_term) #TODO: figure out how to make this run properly after filters
    elif node.parameters.cz_or_iswap == "cz":
        est_flux_shift = np.sqrt(-(qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity) / qp.qubit_control.freq_vs_flux_01_quad_term) #TODO: figure out how to make this run properly after filters
    fluxes_qp[qp.name] = fluxes_qubit + est_flux_shift

assert node.parameters.pulse_duration_ns % 4 == 0, \
    f"Expected pulse duration to be divisible by 4, got {node.parameters.pulse_duration_ns} ns"
pulse_duration_ns = node.parameters.pulse_duration_ns
reset_coupler_bias = False

with program() as CPhase_Oscillations:
    n = declare(int)
    flux_coupler = declare(float)
    flux_qubit = declare(float)
    comp_flux_qubit = declare(float)
    n_st = declare_stream()
    qua_pulse_duration = declare(int, value = pulse_duration_ns // 4)
    if node.parameters.use_state_discrimination:
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    else:
        I_control = [declare(float) for _ in range(num_qubit_pairs)]
        Q_control = [declare(float) for _ in range(num_qubit_pairs)]
        I_target = [declare(float) for _ in range(num_qubit_pairs)]
        Q_target = [declare(float) for _ in range(num_qubit_pairs)]
        I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]


    for i, qp in enumerate(qubit_pairs):
        print("qubit control: %s, qubit target: %s" %(qp.qubit_control.name, qp.qubit_target.name))
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp)
        if reset_coupler_bias:
            qp.coupler.set_dc_offset(0.0)
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux_coupler, fluxes_coupler)):
                with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                    # reset
                    if node.parameters.reset_type == "active":
                        active_reset_simple(qp.qubit_control)
                        active_reset_simple(qp.qubit_target)
                        qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                        wait(qp.qubit_target.thermalization_time * u.ns)
                    align()
                    if "coupler_qubit_crosstalk" in qp.extras:
                        assign(comp_flux_qubit, flux_qubit  +  qp.extras["coupler_qubit_crosstalk"] * flux_coupler )
                    else:
                        print("No crosstalk compensated")
                        assign(comp_flux_qubit, flux_qubit)
                    # setting both qubits ot the initial state
                    qp.qubit_control.xy.play("x180")
                    if node.parameters.cz_or_iswap == "cz":
                        qp.qubit_target.xy.play("x180")
                    align()
                    # wait(8)
                    qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = qua_pulse_duration)
                    qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = qua_pulse_duration)
                    align()
                    wait(20)
                    # readout
                    if node.parameters.use_state_discrimination:
                        if node.parameters.cz_or_iswap == "cz":
                            readout_state_gef(qp.qubit_control, state_control[i])
                        else:
                            readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        assign(state[i], state_control[i]*2 + state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                        save(state[i], state_st[i])
                    else:
                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                        save(I_control[i], I_st_control[i])
                        save(Q_control[i], Q_st_control[i])
                        save(I_target[i], I_st_target[i])
                        save(Q_target[i], Q_st_target[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
                state_st[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state{i + 1}")
            else:
                I_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        from qm import generate_qua_script
        with open("debug.py", "w+") as f:
            f.write(generate_qua_script(CPhase_Oscillations, config))
        job = qm.execute(CPhase_Oscillations)

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {  "flux_qubit": fluxes_qubit, "flux_coupler": fluxes_coupler})
        flux_qubit_full = np.array([fluxes_qp[qp.name] for qp in qubit_pairs])
        ds = ds.assign_coords({"flux_qubit_full": (["qubit", "flux_qubit"], flux_qubit_full)})
    else:
        ds, machine, _, qubit_pairs = load_dataset(node.parameters.load_data_id)

    node.results = {"ds": ds}
# %%
detuning_mode = "quadratic" # "cosine" or "quadratic"
if not node.parameters.simulate:
    flux_coupler_full = np.array([fluxes_coupler + qp.coupler.decouple_offset for qp in qubit_pairs])
    if detuning_mode == "quadratic":
        detuning = np.array([-fluxes_qp[qp.name] ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term  for qp in qubit_pairs])
    elif detuning_mode == "cosine":
        detuning = np.array([oscillation(fluxes_qubit, qp.qubit_control.extras['a'], qp.qubit_control.extras['f'], qp.qubit_control.extras['phi'], qp.qubit_control.extras['offset']) for qp in qubit_pairs])
    ds = ds.assign_coords({"flux_coupler_full": (["qubit", "flux_coupler"], flux_coupler_full)})
    ds = ds.assign_coords({"detuning": (["qubit", "flux_qubit"], detuning)})
    node.results = {"ds": ds}

# %%
node.results["results"] = {}

if not node.parameters.simulate:
    if node.parameters.use_state_discrimination:
        res_sum = -ds.state_control + ds.state_target
    else:
        res_sum = -ds.I_control + ds.I_target

    for i, qp in enumerate(qubit_pairs):
        coupler_min_arg = res_sum.sel(qubit = qp.name).mean(dim = 'flux_qubit').argmin()
        flux_coupler_min = ds.flux_coupler_full.sel(qubit = qp.name)[coupler_min_arg]
        qubit_max_arg = res_sum.sel(qubit = qp.name).mean(dim = "flux_coupler").argmax()
        flux_qubit_max = fluxes_qp[qp.name][qubit_max_arg]
        node.results["results"][qp.name] = {"flux_coupler_min": float(flux_coupler_min.values), "flux_qubit_max": float(flux_qubit_max)}

"""
# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_control.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.I_control.sel(qubit=qp['qubit'])

        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, idle: {qubit_pair.coupler.decouple_offset}V, g=0: {node.results['results'][qp['qubit']]['flux_coupler_min']:.4f}V", fontsize = 10)
        ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_min"], color = 'red', lw = 0.5, ls = '--')
        ax.axhline(1e3*machine.qubit_pairs[qp['qubit']].coupler.decouple_offset, color = 'blue', lw =0.5, ls = '--')
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_max"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        flux_qubit_data = ds.sel(qubit=qp['qubit']).flux_qubit_full.values*1e3
        detuning_data = ds.sel(qubit=qp['qubit']).detuning.values * 1e-6

        def flux_to_detuning(x):
            return np.interp(x, flux_qubit_data, detuning_data)

        def detuning_to_flux(y):
            return np.interp(y, detuning_data, flux_qubit_data)

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux pulse [mV]')
        ax.set_ylabel('Coupler flux pulse [mV]')
    grid.fig.suptitle('Control')
    plt.tight_layout()
    plt.show()
    node.results['figure_control'] = grid.fig

    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qp in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            values_to_plot = ds.state_target.sel(qubit=qp['qubit'])
        else:
            values_to_plot = ds.I_target.sel(qubit=qp['qubit'])

        values_to_plot.assign_coords({"flux_qubit_mV": 1e3*values_to_plot.flux_qubit_full, "flux_coupler_mV": 1e3*values_to_plot.flux_coupler_full}).plot(ax = ax, cmap = 'viridis', x = 'flux_qubit_mV', y = 'flux_coupler_mV')
        qubit_pair = machine.qubit_pairs[qp['qubit']]
        ax.set_title(f"{qp['qubit']}, idle: {qubit_pair.coupler.decouple_offset}V, g=0: {node.results['results'][qp['qubit']]['flux_coupler_min']:.4f}V", fontsize = 10)
        ax.axhline(1e3*node.results["results"][qp["qubit"]]["flux_coupler_min"], color = 'red', lw = 0.5, ls = '--')
        ax.axhline(1e3*machine.qubit_pairs[qp['qubit']].coupler.decouple_offset, color = 'blue', lw =0.5, ls = '--')
        ax.axvline(1e3*node.results["results"][qp["qubit"]]["flux_qubit_max"], color = 'red', lw =0.5, ls = '--')
        # Create a secondary x-axis for detuning
        flux_qubit_data = ds.sel(qubit=qp['qubit']).flux_qubit_full.values*1e3
        detuning_data = ds.sel(qubit=qp['qubit']).detuning.values * 1e-6

        def flux_to_detuning(x):
            return np.interp(x, flux_qubit_data, detuning_data)

        def detuning_to_flux(y):
            return np.interp(y, detuning_data, flux_qubit_data)

        sec_ax = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        sec_ax.set_xlabel('Detuning [MHz]')
        ax.set_xlabel('Qubit flux shift [mV]')
        ax.set_ylabel('Coupler flux [mV]')
    grid.fig.suptitle('Target')
    plt.tight_layout()
    plt.show()
    node.results['figure_target'] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.cz_or_iswap == "cz":
        for qp in qubit_pairs:
            macro_name = f"Cz_square_pulse_{node.parameters.pulse_duration_ns}"

            if macro_name not in qp.macros:
                qp.qubit_control.z.operations[macro_name] = FluxPulse(
                    id=macro_name,
                    length=pulse_duration_ns,
                    amplitude=0.1,
                )
                qp.coupler.operations[macro_name] = FluxPulse(
                    id=macro_name,
                    length=pulse_duration_ns,
                    amplitude=0.1,
                )
                qp.macros[macro_name] = CZMacro(
                    flux_pulse_control=f"#/qubits/{qp.qubit_control.name}/z/operations/{macro_name}",
                    coupler_flux_pulse=f"#/qubit_pairs/{qp.name}/coupler/operations/{macro_name}",
                )
            qp.macros["Cz"] = f"#./{macro_name}"

    with node.record_state_updates():
        for qp in qubit_pairs:
            qp.coupler.decouple_offset = node.results["results"][qp.name]["flux_coupler_min"]
            if node.parameters.cz_or_iswap == "cz":
                qp.macros["Cz"].coupler_flux_pulse.amplitude = node.results["results"][qp.name]["flux_qubit_max"]

            qp.detuning = node.results["results"][qp.name]["flux_qubit_max"]

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()