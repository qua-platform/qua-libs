# %%
"""
        STARK INDUCED ZZ VS PHASE AND AMPLITUDE
The Stark-induced ZZ scripts are designed to measure the ZZ interaction between a control qubit
and a target qubit under various parameters. The ZZ interaction is measured as the difference 
in detuning caused by off-resonant drives at frequencies detuned from the target frequency.

The pulse sequences are as follow:
                                  ____ 
                Control(fC): ____| pi |________________________________________
                                        ______          ______
                Target(fT):  __________| pi/2 |________| pi/2 |________________
                                                ______                     
   ZZ_control (fT-detuning): __________________|  ZZ  |________________________
                                                ______
    ZZ_target (fT-detuning): __________________|  ZZ  |________________________
                                                                ______
                Readout(fR): __________________________________|  RR  |________

This script calibrates the relative drive phase and amplitude of the ZZ control or ZZ target, reproducing the Fig. 2(b) of the referenced paper.
The pulse sequence follows a driven Ramsey type and is repeated with the control qubit in both the |0⟩ and |1⟩ states.
The results are fitted to extract the qubit detuning, and the difference in detuning
between the |0⟩ and |1⟩ states reveals the strength of the ZZ interaction.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Pick an relative drive phase and amplitude that maximixe the ZZ interaction and update the config for
        - ZZ_TARGET_CONSTANTS["zz_target_c{qc}t{qt}"]["square_relative_phase"]
        - ZZ_CONTROL_CONSTANTS["zz_control_c{qc}t{qt}"]["square_amp"]
        - ZZ_TARGET_CONSTANTS["zz_target_c{qc}t{qt}"]["square_amp"]

Reference: Bradley K. Mitchell, et al, Phys. Rev. Lett. 127, 200502 (2021)
"""

# %%
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import peaks_dips
from quam_libs.trackable_object import tracked_updates
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
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


# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["q1-2"]
    qubit_to_sweep_amp: List[Literal["control", "target", "both"]] = ["control"]
    num_averages: int = 20
    operation: str = "x180"
    ramsey_freq_detuning_in_mhz: float = -4.0
    min_amp_scaling: float = 0.25
    max_amp_scaling: float = 1.50
    step_amp_scaling: float = 0.25
    min_zz_target_phase: float = 0.1
    max_zz_target_phase: float = 2.0
    step_zz_target_phase: float = 0.1
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 1000
    wait_time_step_in_ns: int = 16
    zz_control_amps: List[float] = [0.1]
    zz_target_amps: List[float] = [0.1]
    zz_control_amp_scalings: List[float] = [0.5]
    zz_target_amp_scalings: List[float] = [0.5]
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="20b_Stark_induced_ZZ_vs_phase_and_amplitude", parameters=Parameters())


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
    zz = qp.zz_drive
    zz_name = zz.name
    qt_xyd = qp.qubit_target.xy_detuned
    with tracked_updates(zz, auto_revert=False, dont_assign_to_none=True) as zz:
        zz.operations["square"].amplitude = node.parameters.zz_control_amps[i]
        tracked_qubits.append(zz)
    with tracked_updates(qt_xyd, auto_revert=False, dont_assign_to_none=True) as qt_xyd:
        qt_xyd.operations[f"{zz_name}_Square"].amplitude = node.parameters.zz_target_amps[i]
        tracked_qubits.append(qt_xyd)


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
zz_target_phases = np.arange(
    node.parameters.min_zz_target_phase,
    node.parameters.max_zz_target_phase,
    node.parameters.step_zz_target_phase,
)
amp_scalings = np.arange(
    node.parameters.min_amp_scaling,
    node.parameters.max_amp_scaling,
    node.parameters.step_amp_scaling,
)
ramsey_delta_phase = 4e-9 * 1e6 * node.parameters.ramsey_freq_detuning_in_mhz * node.parameters.wait_time_step_in_ns
ramsey_detuning = int(1e6 * node.parameters.ramsey_freq_detuning_in_mhz)


###################
#   QUA Program   #
###################

with program() as stark_induced_zz_vs_frequency:
    I_control, I_control_st, Q_control, Q_control_st, n, n_st = qua_declaration(num_qubit_pairs)
    I_target, I_target_st, Q_target, Q_target_st, _, _ = qua_declaration(num_qubit_pairs)
    t = declare(int)  # QUA variable for the idle time
    df = declare(int)
    s = declare(int)
    a = declare(fixed)
    ph = declare(fixed)
    phase = declare(fixed)

    for i, qp in enumerate(qubit_pairs):
        zz = qp.zz_drive
        qc = qp.qubit_control
        qt = qp.qubit_target
        qt.xy_detuned.update_frequency(zz.intermediate_frequency)

        with for_(n, 0, n < node.parameters.num_averages, n + 1):
            # Save the averaging iteration to get the progress bar
            save(n, n_st)
            
            with for_(*from_array(a, amp_scalings)):

                with for_(*from_array(ph, zz_target_phases)):
                    assign(phase, 0)
                    qt.xy_detuned.frame_rotation_2pi(ph)

                    with for_(*from_array(t, idle_time_cycles)):
                        assign(phase, phase + ramsey_delta_phase)

                        with for_(s, 0, s < 2, s + 1): # states 0:g or 1:e

                            with if_(s == 1):
                                qc.xy.play("x180")
                                align(qc.xy.name, qt.xy_detuned.name)

                            qt.xy.play('x90')
                            align(zz.name, qt.xy.name, qt.xy_detuned.name)

                            if node.parameters.qubit_to_sweep_amp == "both":
                                zz.play("square", amplitude_scale=a)  # drive pulse on q1 at f=ft-d with spec. amp & phase
                                qt.xy_detuned.play(f"{zz.name}_Square", amplitude_scale=a)   # drive pulse on q2 at f=ft-d with spec. amp & phase
                            elif node.parameters.qubit_to_sweep_amp == "control":
                                zz.play("square", amplitude_scale=a)  # drive pulse on q1 at f=ft-d with spec. amp & phase
                                qt.xy_detuned.play(f"{zz.name}_Square")   # drive pulse on q2 at f=ft-d with spec. amp & phase
                            elif node.parameters.qubit_to_sweep_amp == "target":
                                zz.play("square")  # drive pulse on q1 at f=ft-d with spec. amp & phase
                                qt.xy_detuned.play(f"{zz.name}_Square", amplitude_scale=a)   # drive pulse on q2 at f=ft-d with spec. amp & phase
  
                            qt.xy.frame_rotation_2pi(phase)
                            align(zz.name, qt.xy.name, qt.xy_detuned.name)
                            qt.xy.play('x90')

                            # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                            align()

                            # Measure the state of the resonators
                            qc.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                            qt.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                            save(I_control[i], I_control_st[i])
                            save(Q_control[i], Q_control_st[i])
                            save(I_target[i], I_target_st[i])
                            save(Q_target[i], Q_target_st[i])

                            # Wait for the qubits to decay to the ground state
                            qt.resonator.wait(qt.thermalization_time * u.ns)
                            zz.wait(qt.thermalization_time * u.ns)

                            # Reset the frame of the qubits in order not to accumulate rotations
                            reset_frame(qt.xy.name)
                            reset_frame(qt.xy_detuned.name)

    with stream_processing():
        n_st.save("n")
        for i, qb in enumerate(qubit_pairs):
            I_control_st[i].buffer(2).buffer(len(idle_time_cycles)).buffer(len(zz_target_phases)).buffer(len(amp_scalings)).average().save(f"I_control{i + 1}")
            Q_control_st[i].buffer(2).buffer(len(idle_time_cycles)).buffer(len(zz_target_phases)).buffer(len(amp_scalings)).average().save(f"Q_control{i + 1}")
            I_target_st[i].buffer(2).buffer(len(idle_time_cycles)).buffer(len(zz_target_phases)).buffer(len(amp_scalings)).average().save(f"I_target{i + 1}")
            Q_target_st[i].buffer(2).buffer(len(idle_time_cycles)).buffer(len(zz_target_phases)).buffer(len(amp_scalings)).average().save(f"Q_target{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, stark_induced_zz_vs_frequency, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    qm = qmm.open_qm(config, close_other_machines=True)
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(stark_induced_zz_vs_frequency)

    # %% {Live_plot}
    results = fetching_tool(job, ["n"], mode="live")
    while results.is_processing():
        # Fetch results
        n = results.fetch_all()[0]
        # Progress bar
        progress_counter(n, node.parameters.num_averages, start_time=results.start_time)

    # %% {Data_fetching_and_dataset_creation}
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(
        job.result_handles,
        qubit_pairs,
        {"qc_state": [0, 1], "time": idle_time_ns, "zz_target_phases": zz_target_phases, "amp_scalings": amp_scalings},
    )
    ds.time.attrs["long_name"] = "idle_time"
    ds.time.attrs["units"] = "nSec"
    # Add the dataset to the node
    node.results = {"ds": ds}


    # Prepare the figure for live plotting
    # %% {Plot raw data}
    Vnames = ["Ic_g", "Qc_g", "Ic_e", "Qc_e", "It_g", "Qt_g", "It_e", "Qt_e"]
    for qp in qubit_pairs:
        for i, amp in enumerate(amp_scalings):
            Vs = [
                ds.sel(qubit=qp.name, qc_state=0).isel(amp_scalings=i).I_control,
                ds.sel(qubit=qp.name, qc_state=0).isel(amp_scalings=i).Q_control,
                ds.sel(qubit=qp.name, qc_state=1).isel(amp_scalings=i).I_control,
                ds.sel(qubit=qp.name, qc_state=1).isel(amp_scalings=i).Q_control,
                ds.sel(qubit=qp.name, qc_state=0).isel(amp_scalings=i).I_target,
                ds.sel(qubit=qp.name, qc_state=0).isel(amp_scalings=i).Q_target,
                ds.sel(qubit=qp.name, qc_state=1).isel(amp_scalings=i).I_target,
                ds.sel(qubit=qp.name, qc_state=1).isel(amp_scalings=i).Q_target,
            ]
            fig, axss = plt.subplots(4, 2, figsize=(6, 10), sharex=True, sharey=True)
            # Live plot data
            plt.suptitle("Off-resonant Stark shift - I & Q")
            for j, (ax, V, Vname) in enumerate(zip(axss.ravel(), Vs, Vnames)):
                ax.pcolor(ds.time, ds.zz_target_phases, V)
                ax.set_xlabel("Idle time [ns]") if j % 2 == 1 else None
                ax.set_ylabel("Drive Frequency") if j // 2 == 0 else None
                ax.set_title(Vname)
            plt.tight_layout()
            plt.show(block=False)
            node.results[f"figure_raw_{qp.name}_amp_scaling={amp:5.4f}".replace(".", "-")] = fig
            

    # %% {Fit the data}
    from qualang_tools.plot.fitting import Fit
    for qp in qubit_pairs:
        ds_sliced_g = ds.sel(qubit=qp.name, qc_state=0)
        ds_sliced_e = ds.sel(qubit=qp.name, qc_state=1)
        St_g = ds_sliced_g.I_target + 1j * ds_sliced_g.Q_target
        St_e = ds_sliced_e.I_target + 1j * ds_sliced_e.Q_target
        detuning_qt = np.zeros((2, len(amp_scalings), len(zz_target_phases)))
        # fit & plot
        for i, (s, St) in enumerate(zip(["g", "e"], [St_g, St_e])):
            for j, amp in enumerate(amp_scalings):
                for k, phase in enumerate(zz_target_phases):
                    try:
                        fig_analysis = plt.figure()
                        fit = Fit()
                        ramsey_fit = fit.ramsey(ds.time, np.abs(St[j, k, :]), plot=True)
                        qb_T2 = np.abs(ramsey_fit["T2"][0])
                        detuning_qt[i, j, k] = ramsey_fit["f"][0] * u.GHz - node.parameters.ramsey_freq_detuning_in_mhz * u.MHz
                        plt.xlabel("Idle time [ns]")
                        plt.ylabel("abs(I + iQ) [V]")
                        plt.legend((f"qubit detuning = {-detuning_qt[i, j, k] / u.kHz:.3f} kHz", f"T2* = {qb_T2:.0f} ns"))
                        plt.title(f"Ramsey with off-resonant drive and Qc = {s} at amp = {amp:5.4f} phase = {phase:5.4f}")
                        plt.tight_layout()
                        plt.show(block=False)
                        node.results[f"figure_{qp.name}_ramsey_fit_Qc={s}_amp_scaling={amp:5.4f}_phase={phase:5.4f}".replace(".", "-")] = fig_analysis
                        plt.close()
                    except (Exception,):
                        print(f"-> failed: amp={amp:5.4f} phase={phase:5.4f}".replace(".", "-"))


        
    # %% {Plot the summary}
    for qp in qubit_pairs:
        for i, amp in enumerate(amp_scalings):
            fig_summary, axs = plt.subplots(2, 1, figsize=(5, 6), sharex=True)
            # conditional qubit detuning
            axs[0].plot(zz_target_phases, detuning_qt[0, i, :])
            axs[0].plot(zz_target_phases, detuning_qt[1, i, :])
            axs[0].set_xlabel("Drive Freq detuning [Hz]")
            axs[0].set_ylabel("Qubit Freq detuning [Hz]")
            axs[0].set_title("Off-resonant Stark shift")
            axs[0].legend(["Qc=g", "Qc=e"])
            # zz interaction
            axs[1].plot(zz_target_phases, detuning_qt[1, i, :] - detuning_qt[0, i, :], color='m')
            axs[1].set_xlabel("Drive Freq detuning [Hz]")
            axs[1].set_ylabel("ZZ interaction [Hz]")
            axs[1].set_title("Stark-induce ZZ interaction")
            plt.tight_layout()
            plt.show(block=False)
            node.results[f"figure_summary_{qp.name}_amp_scaling={amp:5.4f}".replace(".", "-")] = fig_summary


# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        _zz_control_amp = [0.1]
        _zz_target_amp = [0.1]
        _zz_target_phases = [0.25]
        for i, qp in enumerate(qubit_pairs):
            zz = qp.zz_drive
            qc = qp.qubit_control
            qt = qp.qubit_target
            if node.parameters.qubit_to_sweep_amp == "both":
                zz.operations["square"].amplitude = _zz_control_amp[i]
                qt.xy_detuned.operations[f"{zz_name}_Square"].amplitude = _zz_target_amp[i]
                qt.xy_detuned.operations[f"{zz_name}_Square"].axis_angle = _zz_target_phases[i] * 360
            
            elif node.parameters.qubit_to_sweep_amp == "control":
                zz.operations["square"].amplitude = _zz_control_amp[i]
                qt.xy_detuned.operations[f"{zz_name}_Square"].axis_angle = _zz_target_phases[i] * 360
            
            elif node.parameters.qubit_to_sweep_amp == "target":
                qt.xy_detuned.operations[f"{zz_name}_Square"].amplitude = _zz_target_amp[i]
                qt.xy_detuned.operations[f"{zz_name}_Square"].axis_angle = _zz_target_phases[i] * 360


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
