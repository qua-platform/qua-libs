# %%
"""
        CZ CALIB CZ PULSE VS FREQUENCY AND AMPLITUDE
The CZ calibration scripts are designed to calibrate the CZ gate by adjusting the frequency and amplitude.
Entanglement is measured as R = 0.5 * |r1 - r0|^2 , where r1 and r0 are the expectation values of the Bloch vectors.

The pulse sequences are as follow:
                                  ______ 
                Control(fC): ____|  pi  |__________________________
                                          ______                     
   ZZ_control (fT-detuning): ____________|  ZZ  |__________________
                                  ______  ______  _____
    ZZ_target (fT-detuning): ____| pi/2 ||  ZZ  || QST |___________
                                                         ______
                Readout(fR): ___________________________|  RR  |___

This script measures entanglement as a function of frequency and amplitude, replicating Fig. 3(b) of the referenced paper.
The pulse sequence is repeated with the control qubit in both the |0⟩ and |1⟩ states.
The optimal frequency and amplitude pair is selected where the entanglement measure R is maximized (close to 1).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Pick a pair of frequency and amplitude that maximize R and update the config for
        - ZZ_CONTROL_CONSTANTS["zz_control_c{qc}t{qt}"]["detuning"]
        - ZZ_TARGET_CONSTANTS["zz_target_c{qc}t{qt}"]["detuning"]
        - ZZ_CONTROL_CONSTANTS["zz_control_c{qc}t{qt}"]["square_amp"]
        - ZZ_TARGET_CONSTANTS["zz_target_c{qc}t{qt}"]["square_amp"]
      In the end, we want to make the CZ gate as short short as possible with highest fidelity.
      Thus, we want to pick a large enough amplitude for the ve however without causing too much of leakage.

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
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.1
    ramsey_freq_detuning_in_mhz: float = -4.0
    min_amp_scaling: float = 0.25
    max_amp_scaling: float = 1.50
    step_amp_scaling: float = 0.25
    zz_control_amps: List[float] = [0.1]
    zz_target_amps: List[float] = [0.1]
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="21a_cz_calib_cz_pulse_vs_frequency_and_amplitude", parameters=Parameters())


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


# Parameters Definition
# The frequency sweep around the resonator resonance frequency
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)
amp_scalings = np.arange(
    node.parameters.min_amp_scaling,
    node.parameters.max_amp_scaling,
    node.parameters.step_amp_scaling,
)


###################
#   QUA Program   #
###################

with program() as cz_calib_cz_pulse_vs_frequency_and_amplitude:
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    df = declare(int)
    a = declare(fixed)
    c = declare(int)
    s = declare(int)
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
            
                with for_(*from_array(df, dfs)):
                    assign(phase, 0)
                    zz.update_frequency(df + zz.intermediate_frequency)
                    qt.xy_detuned.update_frequency(df + zz.intermediate_frequency)

                    with for_(c, 0, c < 3, c + 1): # bases 

                        with for_(s, 0, s < 2, s + 1): # states 0:g or 1:e

                            with if_(s == 1):
                                qc.xy.play("x180")
                                align(qc.xy.name, qt.xy_detuned.name)

                            qt.xy.play('x90')
                            align(zz.name, qt.xy.name, qt.xy_detuned.name)

                            zz.play("square", amplitude_scale=a)  # drive pulse on q1 at f=ft-d with spec. amp & phase
                            qt.xy_detuned.play(f"{zz.name}_Square", amplitude_scale=a)   # drive pulse on q2 at f=ft-d with spec. amp & phase

                            align(zz.name, qt.xy.name, qt.xy_detuned.name)

                            # QST on Target
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
            state_st_control[i].buffer(2).buffer(3).buffer(len(dfs)).buffer(len(amp_scalings)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(3).buffer(len(dfs)).buffer(len(amp_scalings)).average().save(f"state_target{i + 1}")



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cz_calib_cz_pulse_vs_frequency_and_amplitude, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    qm = qmm.open_qm(config, close_other_machines=True)
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(cz_calib_cz_pulse_vs_frequency_and_amplitude)

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
        {"qc_state": [0, 1], "qt_component": ["X", "Y", "Z"], "freq": dfs, "amp_scalings": amp_scalings},
    )
    ds.freq.attrs["long_name"] = "frequency"
    ds.freq.attrs["units"] = "Hz"
    # Add the dataset to the node
    node.results = {"ds": ds}


    # %% {Plot raw data}
    Vnames = [
        "state_control_X_g", "state_control_X_e",
        "state_control_Y_g", "state_control_Y_e",
        "state_control_Z_g", "state_control_Z_e",
        "state_target_X_g", "state_target_X_e",
        "state_target_Y_g", "state_target_Y_e",
        "state_target_Z_g", "state_target_Z_e",
    ]
    for qp in qubit_pairs:
        fig, axss = plt.subplots(4, 3, figsize=(8, 10), sharex=True, sharey=True)
        Vs = [
            ds.sel(qubit=qp.name, qc_state=0, qt_component="X").state_control,
            ds.sel(qubit=qp.name, qc_state=1, qt_component="X").state_control,
            ds.sel(qubit=qp.name, qc_state=0, qt_component="Y").state_control,
            ds.sel(qubit=qp.name, qc_state=1, qt_component="Y").state_control,
            ds.sel(qubit=qp.name, qc_state=0, qt_component="Z").state_control,
            ds.sel(qubit=qp.name, qc_state=1, qt_component="Z").state_control,
            ds.sel(qubit=qp.name, qc_state=0, qt_component="X").state_target,
            ds.sel(qubit=qp.name, qc_state=1, qt_component="X").state_target,
            ds.sel(qubit=qp.name, qc_state=0, qt_component="Y").state_target,
            ds.sel(qubit=qp.name, qc_state=1, qt_component="Y").state_target,
            ds.sel(qubit=qp.name, qc_state=0, qt_component="Z").state_target,
            ds.sel(qubit=qp.name, qc_state=1, qt_component="Z").state_target,
        ]
        # Live plot data
        plt.suptitle("Off-resonant Stark shift - I & Q")
        for j, (ax, V, Vname) in enumerate(zip(axss.ravel(), Vs, Vnames)):
            ax.pcolor(ds.freq, ds.amp_scalings, V)
            ax.set_xlabel("Freq detuning [Hz]") if j // 3 == 3 else None
            ax.set_ylabel("Amplitude scale") if j % 3 == 0 else None
            ax.set_title(Vname)
        plt.tight_layout()
        plt.show(block=False)
        node.results[f"figure_raw_{qp.name}"] = fig


    # %% {Plot entanglement measure}
    # Compute the entanglement measure
    data_e = ds.sel(qc_state=1).state_target
    data_g = ds.sel(qc_state=0).state_target
    R = 0.5 * ((data_e - data_g) ** 2).sum(dim="qt_component")
    for qp in qubit_pairs:
        # Summary
        fig_summary = plt.figure()
        plt.pcolor(ds.freq, ds.amp_scalings, R.sel(qubit=qp.name))
        plt.xlabel("Freq detuning [Hz]")
        plt.ylabel("Amplitude scale")
        plt.title("CZ Gate Calibration")
        plt.colorbar()
        plt.tight_layout()
        node.results[f"figure_summary_{qp.name}"] = fig_summary


# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        _zz_amp_scaling = [0.9]
        _zz_target_detuning_in_mhz = [0.25]
        for i, qp in enumerate(qubit_pairs):
            qp.zz_drive.operations["square"].amplitude *= _zz_amp_scaling[i]
            qp.zz_drive.detuning += _zz_target_detuning_in_mhz[i] * u.MHz
            qp.qubit_target.xy_detuned.operations[f"{zz_name}_Square"].amplitude *= _zz_amp_scaling[i]

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
