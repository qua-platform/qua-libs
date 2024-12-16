# %%
"""
        CZ CALIB CZ PULSE VS LOCAL PHASE
The CZ calibration scripts are designed to calibrate the CZ gate by compenstating the phases for ZI and IZ.
CZ = exp(- i/2 * pi/2 * (- ZI - IZ + ZZ)) and CZ pulse = exp(- i/2 * (a * ZI + b * IZ + pi/2 * ZZ)) without phase compensations.
By adding phases phi_ZI and phi_IZ to ZI and IZ, CZ <- exp(- i/2 * ((a + phi_ZI) * ZI + (b + phi_IZ) * IZ + pi/2 * ZZ)) 
Namely, we want to compensate the phases such that it forms CZ.

    a + phi_ZI = -pi/2
    b + phi_IZ = -pi/2

The pulse sequences are as follow:
                                   _____                    ______
                Control(fC): _____| y90 |__________________| -y90 |___________
                                  ______ 
                Target(fT):  ____| x180 |____________________________________
                                          ______  ________                    
   ZZ_control (fT-detuning): ____________|  ZZ  || phi_ZI |___________________
                                          ______  ________ 
    ZZ_target (fT-detuning): ____________|  ZZ  || phi_IZ |___________________
                                                                     ______
                Readout(fR): _______________________________________|  RR  |___

This script measures entanglement as a function of phi_ZI (by flipping the role of qc and qz: phi_IZ), replicating Fig. S2(b) of the referenced paper.
The pulse sequence is repeated with the control qubit in both the |0⟩ and |1⟩ states.
The optimal phi_ZI and phi_IZ are selected where a + phi_ZI = -pi/2 and b + phi_IZ = -pi/2.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Having found the frequency, amplitudes and relative phase shift of zz_control and zz_target.

Next steps before going to the next node:
    - Pick phi_ZI and phi_IZ such that (a + phi_ZI) = -pi/2 and (b + phi_IZ) = -pi/2 and update the config for
        - ZZ_CONTROL_CONSTANTS["zz_control_c{qc}t{qt}"]["square_phi_ZI"]
        - ZZ_TARGET_CONSTANTS["zz_target_c{qc}t{qt}"]["square_phi_IZ"]

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
    qubit_to_sweep_local_phase: List[Literal["control", "target"]] = ["control"]
    num_averages: int = 20
    min_local_phase: float = 0.05
    max_local_phase: float = 1.95
    step_local_phase: float = 0.05
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="21b_cz_calib_cz_pulse_vs_phase", parameters=Parameters())


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

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()


# Parameters Definition
local_phases = np.arange(
    node.parameters.min_local_phase,
    node.parameters.max_local_phase,
    node.parameters.step_local_phase,
)


###################
#   QUA Program   #
###################

with program() as cz_calib_cz_pulse_vs_local_phase:
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
    ph = declare(fixed)

    for i, qp in enumerate(qubit_pairs):
        zz = qp.zz_drive
        qc = qp.qubit_control
        qt = qp.qubit_target
        qt.xy_detuned.update_frequency(zz.intermediate_frequency)

        with for_(n, 0, n < node.parameters.num_averages, n + 1):
            # Save the averaging iteration to get the progress bar
            save(n, n_st)

            for ramsey_target in ["qc", "qt"]:

                if ramsey_target == "qc":
                    qb_ramsey_target = qc
                    qb_ramsey_control = qt
                elif ramsey_target == "qt":
                    qb_ramsey_target = qt
                    qb_ramsey_control = qc

                align()
                
                with for_(*from_array(ph, local_phases)):

                    with for_(s, 0, s < 2, s + 1): # states

                        # Prepare Qt to |1>
                        with if_(s == 0):
                            qb_ramsey_target.xy.play("y90")
                        with if_(s == 1):
                            qb_ramsey_target.xy.play("y90")
                            qb_ramsey_control.xy.play("x180")

                        align()
                        zz.play("square")
                        qt.xy_detuned.play(f"{zz.name}_Square")
                        align()

                        qb_ramsey_target.xy.frame_rotation(ph)

                        # Bring Qc back to z axis                
                        qb_ramsey_target.xy.play("-y90")

                        # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                        align()

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
            state_st_control[i].buffer(2).buffer(len(local_phases)).buffer(2).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(len(local_phases)).buffer(2).average().save(f"state_target{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cz_calib_cz_pulse_vs_local_phase, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    qm = qmm.open_qm(config, close_other_machines=True)
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(cz_calib_cz_pulse_vs_local_phase)

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
        {"ramsey_control_state": [0, 1], "local_phases": local_phases, "ramsey_target": ["qc", "qt"]},
    )
    ds.local_phases.attrs["long_name"] = "local_phases"
    ds.local_phases.attrs["units"] = "2pi rad."
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Plot raw data}
    Vnames = [
        "ramsey_target=qc",
        "ramsey_target=qt",
        "ramsey_target=qc",
        "ramsey_target=qt",
    ]
    for qp in qubit_pairs:
        fig, axss = plt.subplots(2, 2, figsize=(8, 6))
        Vs = [
            ds.sel(qubit=qp.name, ramsey_target="qc").state_control,
            ds.sel(qubit=qp.name, ramsey_target="qt").state_control,
            ds.sel(qubit=qp.name, ramsey_target="qc").state_target,
            ds.sel(qubit=qp.name, ramsey_target="qt").state_target,
        ]
        plt.suptitle(f"Local phase calibration for CZ")
        for i, (ax, V, Vname) in enumerate(zip(axss.ravel(), Vs, Vnames)):
            ax.plot(ds.local_phases, V.sel(ramsey_control_state=0), color="b", label=[f"{qb_ramsey_control} = |0>"])
            ax.plot(ds.local_phases, V.sel(ramsey_control_state=1), color="r", label=[f"{qb_ramsey_control} = |1>"])
            ax.set_xlabel("Phase [2pi rad.]" if i // 2 == 1 else None)
            ax.set_ylabel(f"State control {qp.qubit_control.name}" if i % 2 == 0 else f"State target {qp.qubit_target.name}")
            ax.set_title(Vname)
        plt.tight_layout()
        node.results[f"figure_summary_{qp.name}"] = fig


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
