# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.experiments.simulation import simulate_and_plot
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V, subtract_slope, apply_angle
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.power_tools import set_output_power_mw_channel
from quam_libs.trackable_object import tracked_updates
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
from typing import Literal, Optional, List
import numpy as np


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 100
    frequency_span_in_mhz: float = 15
    frequency_step_in_mhz: float = 0.1
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    max_power_dbm: int = -30
    min_power_dbm: int = -50
    num_power_points: int = 100
    max_amp: float = 0.1
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    multiplexed: bool = False
    load_data_id: Optional[int] = None

node = QualibrationNode(name="02c_Resonator_Spectroscopy_vs_Amplitude", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)

machine = QuAM.load()

qubits = machine.get_qubits_used_in_node(node.parameters)
resonators = [qubit.resonator for qubit in qubits]
prev_amps = [rr.operations["readout"].amplitude for rr in resonators]

# Update the readout power to match the desired range, this change will be reverted at the end of the node.
with tracked_updates(machine.twpa, auto_revert=False, dont_assign_to_none=True) as tracked_twpa:
    set_output_power_mw_channel(
        channel=machine.twpa,
        power_in_dbm=node.parameters.max_power_dbm,
        operation="const",
        max_amplitude=node.parameters.max_amp
    )

config = machine.generate_config()

if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The twpa pulse amplitude sweep
amp_min = resonators[0].calculate_voltage_scaling_factor(
    fixed_power_dBm=node.parameters.max_power_dbm,
    target_power_dBm=node.parameters.min_power_dbm
)
amp_max = 1

amps = np.geomspace(amp_min, amp_max, node.parameters.num_power_points)

span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

qubit = qubits[0]
num_qubits = 1

with program() as twpa_calibration:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)

    a = declare(fixed)
    df = declare(int)

    machine.set_all_fluxes(flux_point=flux_point, target=qubits[0])

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            update_frequency(machine.twpa.name, df + machine.twpa.intermediate_frequency)
            qubit.resonator.wait(machine.depletion_time * u.ns)

            with for_(*from_array(a, amps)):
                machine.twpa.play("const", amplitude_scale=a)

                qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                qubit.resonator.wait(machine.depletion_time * u.ns)
                save(I[0], I_st[0])
                save(Q[0], Q_st[0])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    fig, samples = simulate_and_plot(qmm, config, twpa_calibration, node.parameters)
    node.results = {"figure": fig, "samples": samples}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(twpa_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        power_dbm = np.linspace(
            node.parameters.min_power_dbm,
            node.parameters.max_power_dbm,
            node.parameters.num_power_points
        )
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"power_dbm": power_dbm, "freq": dfs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        ds = ds.assign({"phase": subtract_slope(apply_angle(ds.I + 1j * ds.Q, dim="freq"), dim="freq")})
        # Add the tpwa RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + machine.twpa.RF_frequency for q in qubits])
        ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq)})
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        ds.power_dbm.attrs["long_name"] = "Power"
        ds.power_dbm.attrs["units"] = "dBm"

        # Normalize the IQ_abs with respect to the amplitude axis
        ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["freq"])})

    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}

    # %% {Plotting}
    node.results["figure"] = fig

    # %% {Update_state}
    # Revert the change done at the beginning of the node
    tracked_twpa.revert_changes()

    # Save fitting results
    if not node.parameters.load_data_id:
        with node.record_state_updates():
            pass

    # %% {Save_results}
    if node.parameters.load_data_id is not None:
        if node.storage_manager is not None:
            node.storage_manager.active_machine_path = None

    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
