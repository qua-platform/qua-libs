# %%
"""
                                 CR_calib_cancel_drive_phase

The CR_calib scripts are designed for calibrating cross-resonance (CR) gates involving a system
with a control qubit and a target qubit. These scripts help estimate the parameters of a Hamiltonian,
which is represented as:
    H = I ⊗ (a_X X + a_Y Y + a_Z Z) + Z ⊗ (b_I I + b_X X + b_Y Y + b_Z Z)


For the calibration sequences, we employ echoed CR drive.
                                   ____      ____ 
            Control(fC): _________| pi |____| pi |________________
                             ____                     
                 CR(fT): ___|    |_____      _____________________
                                       |____|     _____
             Target(fT): ________________________| QST |__________
                                                         ______
            Readout(fR): _____________________ _________|  RR  |__

This script is to calibrate the phase of CR cancellation drive.
CR cancellation pulse is applied to the target qubit at the target qubit frequency.
Each sequence, which varies in the duration of the CR drive and the phase of CR cancel drive,
ends with state tomography of the target state (across X, Y, and Z bases).
This process is repeated with the control state in both |0> and |1> states.
We fit the two sets of CR duration versus tomography data to a theoretical model,
yielding two sets of three parameters: delta, omega_x, and omega_y.
Using these parameters, we estimate the interaction coefficients of the Hamiltonian.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Find the phase to shift for the CR cancel drive via phi = arctan(IY/IX).
      Alternatively, find the phase where a_Y (coeff of I_Y) is zero. We call it phi1.
      phi = phi0 - phi1.
      Note that the phase is in units of 2 * pi as it is used with `frame_rotation_2pi`.
      Update relevant parameters at the end of the script.

Reference: Sarah Sheldon, Easwar Magesan, Jerry M. Chow, and Jay M. Gambetta Phys. Rev. A 93, 060302(R) (2016)
"""

# %%
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import peaks_dips
from quam_libs.trackable_object import tracked_updates
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from quam_libs.macros import (
    qua_declaration,
    multiplexed_readout,
    node_save,
    active_reset,
    readout_state,
)
from cr_hamiltonian_tomography import (
    CRHamiltonianTomographyAnalysis,
    plot_cr_duration_vs_scan_param,
    plot_interaction_coeffs,
    PAULI_2Q,
)


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["q1-2"]
    num_averages: int = 20
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 1000
    wait_time_step_in_ns: int = 16
    min_cr_cancel_phase: float = 0.05
    max_cr_cancel_phase: float = 1.95
    step_cr_cancel_phase: float = 0.05
    cr_type: Literal["direct+cancel", "direct+cancel+echo"] = "direct+cancel+echo"
    cr_drive_amps: List[float] = [0.1]
    cr_cancel_amps: List[float] = [0.1]
    cr_drive_amp_scalings: List[float] = [0.5]
    cr_cancel_amp_scalings: List[float] = [0.5]
    cr_drive_phases: List[float] = [0.5]
    cr_cancel_phases: List[float] = [0.5]
    use_state_discrimination: bool = False
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="18d_CR_calib_cr_cancel_phase", parameters=Parameters())


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)


# Update the readout power to match the desired range, this change will be reverted at the end of the node.
tracked_qubits = []
for i, qp in enumerate(qubit_pairs):
    cr = qp.cross_resonance
    cr_name = cr.name
    qt_xy = qp.qubit_target.xy
    with tracked_updates(cr, auto_revert=False, dont_assign_to_none=True) as cr:
        cr.operations["square"].amplitude = node.parameters.cr_drive_amps[i]
        cr.operations["square"].axis_angle = node.parameters.cr_drive_phases[i] * 360
        tracked_qubits.append(cr)
    with tracked_updates(qt_xy, auto_revert=False, dont_assign_to_none=True) as qt_xy:
        qt_xy.operations[f"{cr_name}_Square"].amplitude = node.parameters.cr_cancel_amps[i]
        qt_xy.operations[f"{cr_name}_Square"].axis_angle = node.parameters.cr_cancel_phases[i] * 360
        tracked_qubits.append(qt_xy)


# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_time_ns = np.arange(
    node.parameters.min_wait_time_in_ns,
    node.parameters.max_wait_time_in_ns,
    node.parameters.wait_time_step_in_ns,
) // 4 * 4
idle_time_cycles = idle_time_ns // 4
cr_cancel_phases = np.arange(
    node.parameters.min_cr_cancel_phase,
    node.parameters.max_cr_cancel_phase,
    node.parameters.step_cr_cancel_phase,
)


###################
#   QUA Program   #
###################

with program() as cr_calib_unit_ham_tomo:
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    ph = declare(fixed)
    t = declare(int)
    s = declare(int)  # QUA variable for the control state
    c = declare(int)  # QUA variable for the projection index in QST

    for i, qp in enumerate(qubit_pairs):
        qc = qp.qubit_control
        qt = qp.qubit_target
        cr = qp.cross_resonance

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(ph, cr_cancel_phases)):
                with for_(*from_array(t, idle_time_cycles)):
                    with for_(c, 0, c < 3, c + 1):  # bases
                        with for_(s, 0, s < 2, s + 1):  # states
                            with if_(s == 1):
                                qc.xy.play("x180")
                                align(qc.xy.name, qt.xy.name, cr.name)

                            if node.parameters.cr_type == "direct+cancel":
                                # phase shift for cancel drive
                                cr.frame_rotation_2pi(node.parameters.cr_drive_phases[i])
                                qt.xy.frame_rotation_2pi(ph)
                                # direct + cancel
                                align(qc.xy.name, qt.xy.name, cr.name)
                                cr.play("square", duration=t, amplitude_scale=node.parameters.cr_drive_amp_scalings[i])
                                qt.xy.play(f"{cr.name}_Square", duration=t, amplitude_scale=node.parameters.cr_cancel_amp_scalings[i])
                                # align for the next step and clear the phase shift
                                align(qt.xy.name, cr.name)
                                reset_frame(cr.name)
                                reset_frame(qt.xy.name)

                            elif node.parameters.cr_type == "direct+cancel+echo":
                                # phase shift for cancel drive
                                cr.frame_rotation_2pi(node.parameters.cr_drive_phases[i])
                                qt.xy.frame_rotation_2pi(ph)
                                # direct + cancel
                                align(qc.xy.name, qt.xy.name, cr.name)
                                cr.play("square", duration=t, amplitude_scale=node.parameters.cr_drive_amp_scalings[i])
                                qt.xy.play(f"{cr.name}_Square", duration=t, amplitude_scale=node.parameters.cr_cancel_amp_scalings[i])
                                # pi pulse on control
                                align(qc.xy.name, qt.xy.name, cr.name)
                                qc.xy.play("x180")
                                # echoed direct + cancel
                                align(qc.xy.name, qt.xy.name, cr.name)
                                cr.play("square", duration=t, amplitude_scale=-node.parameters.cr_drive_amp_scalings[i])
                                qt.xy.play(f"{cr.name}_Square", duration=t, amplitude_scale=-node.parameters.cr_cancel_amp_scalings[i])
                                # pi pulse on control
                                align(qc.xy.name, qt.xy.name, cr.name)
                                qc.xy.play("x180")
                                # align for the next step and clear the phase shift
                                align(qc.xy.name, qt.xy.name)
                                reset_frame(cr.name)
                                reset_frame(qt.xy.name)

                            # QST on Target
                            align(qt.xy.name, cr.name)
                            with switch_(c):
                                with case_(0):  # projection along X
                                    qt.xy.play("-y90")
                                with case_(1):  # projection along Y
                                    qt.xy.play("x90")
                                with case_(2):  # projection along Z
                                    qt.xy.wait(qt.xy.operations["x180"].length * u.ns)

                            align(qt.xy.name, qc.resonator.name, qt.resonator.name)

                            # Measure the state of the resonators
                            readout_state(qc, state_control[i])
                            readout_state(qt, state_target[i])
                            save(state_control[i], state_st_control[i])
                            save(state_target[i], state_st_target[i])

                            # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                            wait(1 * u.us)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(2).buffer(3).buffer(len(idle_time_cycles)).buffer(len(cr_cancel_phases)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(3).buffer(len(idle_time_cycles)).buffer(len(cr_cancel_phases)).average().save(f"state_target{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cr_calib_unit_ham_tomo, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    qm = qmm.open_qm(config, close_other_machines=True)
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(cr_calib_unit_ham_tomo)

    results = fetching_tool(job, ["n"], mode="live")
    while results.is_processing():
        # Fetch results
        n = results.fetch_all()[0]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        
# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(
        job.result_handles,
        qubit_pairs,
        {"qc_state": ["0", "1"], "qt_component": ["X", "Y", "Z"], "times": idle_time_ns, "cr_cancel_phases": cr_cancel_phases},
    )
    # Then, add new data variables based on existing ones
    ds = ds.assign(
        bloch_control=-2 * ds["state_control"] + 1,
        bloch_target=-2 * ds["state_target"] + 1,
    )

    for qp in qubit_pairs:
        ds_sliced = ds.sel(qubit=qp.name)
    
        # Prepare the figure for live plotting
        fig, axss = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
        # plotting data
        plot_cr_duration_vs_scan_param(
            ds_sliced.bloch_control.data,
            ds_sliced.bloch_target.data,
            ds_sliced.times.data,
            cr_cancel_phases,
            "cr cancel phase",
            axss,
        )
        node.results[f"figure_{qp.name}"] = fig

        # Perform CR Hamiltonian tomography
        coeffs = []
        for idx, ph in enumerate(cr_cancel_phases):
            print("-" * 40)
            print(f"fitting for phase = {ph}")
            try:
                crht = CRHamiltonianTomographyAnalysis(
                    ts=ds_sliced.times.data,
                    data=ds_sliced.isel(cr_cancel_phases=idx).bloch_target.data,  # target data: len(cr_cancel_phases) x len(t_vec_cycle) x 3 x 2
                )
                crht.fit_params()
                coeffs.append(crht.interaction_coeffs_MHz)
                fig_analysis = crht.plot_fit_result(do_show=False)
                node.results[f"figure_analysis_{qp.name}_cr_cancel_phases={ph:5.4f}".replace(".", "-")] = fig_analysis
            except:
                print(f"-> failed")
                crht.interaction_coeffs_MHz = {k: None for k, v in crht.interaction_coeffs_MHz.items()}
                coeffs.append({p: None for p in PAULI_2Q})

        # Plot the estimated interaction coefficients
        fig_summary = plot_interaction_coeffs(coeffs, cr_cancel_phases, xlabel="cr cancel phase")
        node.results[f"figure_summary_{qp.name}"] = fig_summary

    qm.close()
    print("Experiment QM is now closed")
    plt.show(block=True)


# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        cr_cancel_phases = [0.5]
        for i, qp in enumerate(qubit_pairs):
            qt = qp.qubit_target
            qt.xy.operations[f"{cr.name}_Square"].axis_angle = cr_cancel_phases[i] * 360

    # Revert the change done at the beginning of the node
    for tracked_qubit in tracked_qubits:
        tracked_qubit.revert_changes()


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()


# %%
