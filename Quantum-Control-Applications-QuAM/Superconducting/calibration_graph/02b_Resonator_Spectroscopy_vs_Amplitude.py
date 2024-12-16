# %%
"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for all resonators simultaneously.
This is done across various readout intermediate dfs and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "f_res" and "", in the state.
    - Adjust the readout amplitude, labeled as "readout_pulse_amp", in the state.
    - Save the current state by calling machine.save("quam")
"""


# %% {Imports}
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


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 100
    frequency_span_in_mhz: float = 30
    frequency_step_in_mhz: float = 0.1
    simulate: bool = False
    timeout: int = 100
    max_power_dbm: int = 1
    min_power_dbm: int = -50
    max_amp: float = 0.4
    ro_line_attenuation_dB: float = 0
    multiplexed: bool = True


node = QualibrationNode(
    name="02b_Resonator_Spectroscopy_vs_Amplitude", parameters=Parameters()
)


u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
resonators = [qubit.resonator for qubit in qubits]
prev_amps = [rr.operations["readout"].amplitude for rr in resonators]
num_qubits = len(qubits)
num_resonators = len(resonators)

# Update the readout power to match the desired range, this change will be reverted at the end of the node.
tracked_qubits = []
for i, qubit in enumerate(qubits):
    with tracked_updates(qubit, auto_revert=False, dont_assign_to_none=True) as qubit:
        qubit.resonator.operations["readout"].amplitude = node.parameters.max_amp
        qubit.resonator.opx_output.full_scale_power_dbm = node.parameters.max_power_dbm
        tracked_qubits.append(qubit)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
amp_max = 10 ** (-(node.parameters.max_power_dbm - node.parameters.max_power_dbm) / 20)
amp_min = 10 ** (-(node.parameters.max_power_dbm - node.parameters.min_power_dbm) / 20)
amps = np.geomspace(
    amp_min, amp_max, 100
)  # 100 points from 0.01 to 1.0, logarithmically spaced

# The frequency sweep around the resonator resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)

with program() as multi_res_spec_vs_amp:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    df = declare(int)  # QUA variable for the readout frequency

    for i, qubit in enumerate(qubits):
        # resonator of this qubit
        rr = qubit.resonator

        with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
            save(n, n_st)

            with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
                # Update the resonator frequencies for all resonators
                update_frequency(rr.name, df + rr.intermediate_frequency)
                rr.wait(machine.depletion_time * u.ns)

                with for_(
                    *from_array(a, amps)
                ):  # QUA for_ loop for sweeping the readout amplitude
                    # readout the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                    # wait for the resonator to relax
                    rr.wait(machine.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
        if not node.parameters.multiplexed:
            align(*[rr.name for rr in resonators])

    with stream_processing():
        n_st.save("n")
        for i in range(num_resonators):
            I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_amp, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    qm = qmm.open_qm(config, close_other_machines=True)
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(multi_res_spec_vs_amp)

    # %% {Live_plot}
    results = fetching_tool(job, ["n"], mode="live")
    while results.is_processing():
        # Fetch results
        n = results.fetch_all()[0]
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": amps, "freq": dfs})
    # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
    ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
    # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
    RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
    ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq)})
    ds.freq_full.attrs["long_name"] = "Frequency"
    ds.freq_full.attrs["units"] = "GHz"

    # Add the absolute readout pulse amplitude to the dataset
    def abs_amp(q):
        def foo(amp):
            return amp * node.parameters.max_amp

        return foo

    def abs_pow(q):
        def foo(amp):
            return amp + q.resonator.get_output_power("readout")

        return foo

    ds = ds.assign_coords(
        {"abs_amp": (["qubit", "amp"], np.array([abs_amp(q)(amps) for q in qubits]))}
    )
    ds.abs_amp.attrs["long_name"] = "Amplitude"
    ds.abs_amp.attrs["units"] = "V"
    # Add the absolute readout power to the dataset
    # ds = ds.assign_coords({'power_dbm': (['qubit', 'amp'], np.array([u.volts2dBm(a) - node.parameters.ro_line_attenuation_dB for a in ds.abs_amp.values]))})
    ds = ds.assign_coords(
        {
            "power_dbm": (
                ["qubit", "amp"],
                np.array(
                    [
                        abs_pow(q)(20 * np.log10(amps))
                        - node.parameters.ro_line_attenuation_dB
                        for q in qubits
                    ]
                ),
            )
        }
    )
    ds.power_dbm.attrs["long_name"] = "Power"
    ds.power_dbm.attrs["units"] = "dBm"
    # Normalize the IQ_abs with respect to the amplitude axis
    ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["freq"])})
    # Add the dataset to the node
    node.results = {"ds": ds}

# %% {Data_analysis}
if not node.parameters.simulate:
    # Follow the resonator line for each amplitude - This gives a ds with all qubits for each amplitude
    res_min_vs_amp = [
        peaks_dips(
            ds.IQ_abs_norm.sel(amp=amp), dim="freq", prominence_factor=5
        ).position
        for amp in ds.amp
    ]
    # This concatenates all the amplitudes within the same ds
    res_min_vs_amp = xr.concat(res_min_vs_amp, "amp")
    # Get the full resonance frequencies for all amplitudes
    res_freq_full = ds.freq_full.sel(freq=0, method="nearest") + res_min_vs_amp
    # Get the resonance frequency at high and low readout powers
    res_low_power = res_min_vs_amp.sel(amp=slice(0.001, 0.03)).mean(dim="amp")
    res_hi_power = res_min_vs_amp.isel(amp=-1)
    # Find the maximum readout amplitude for which the resonance frequency is close to the low power resonance
    rr_pwr = xr.where(
        abs(res_min_vs_amp - res_low_power) < 0.15 * abs(res_hi_power - res_low_power),
        res_min_vs_amp.amp,
        0,
    ).max(dim="amp")
    # Take 30% of it for being sure to be far from the punch out (?)
    RO_power_ratio = 0.3
    rr_pwr = RO_power_ratio * rr_pwr

# %% {Plotting}
if not node.parameters.simulate:
    grid_names = [f"{q.name}_0" for q in qubits]
    grid = QubitGrid(ds, grid_names)

    for ax, qubit in grid_iter(grid):
        # Create a secondary y-axis for power in dBm
        ax2 = ax.twinx()
        # Plot the data using the secondary y-axis
        ds.loc[qubit].IQ_abs_norm.plot(
            ax=ax2,
            add_colorbar=False,
            x="freq_full",
            y="abs_amp",
            robust=True,
            yscale="log",
        )
        ds.loc[qubit].IQ_abs_norm.plot(
            ax=ax, add_colorbar=False, x="freq_full", y="power_dbm", robust=True
        )
        ax.set_ylabel("Power (dBm)")
        # Plot the resonance frequency  for each amplitude
        ax2.plot(
            res_freq_full.loc[qubit],
            ds.abs_amp.loc[qubit],
            color="orange",
            linewidth=0.5,
        )
        # Plot where the optimum readout power was found
        ax2.axhline(
            y=abs_amp(machine.qubits[qubit["qubit"]])(rr_pwr.loc[qubit]).values,
            color="r",
            linestyle="--",
        )

    grid.fig.suptitle("Resonator spectroscopy VS. power at base")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        if float(rr_pwr.sel(qubit=q.name)) > 0:
            with node.record_state_updates():
                q.resonator.operations["readout"].amplitude = float(
                    rr_pwr.sel(qubit=q.name)
                )
                q.resonator.intermediate_frequency += int(
                    res_low_power.sel(qubit=q.name).values
                )
        fit_results[q.name]["RO_amplitude"] = float(rr_pwr.sel(qubit=q.name))
        fit_results[q.name]["RO_frequency"] = q.resonator.RF_frequency
    node.results["resonator_frequency"] = fit_results

    # Revert the change done at the beginning of the node
    for tracked_qubit in tracked_qubits:
        tracked_qubit.revert_changes()

# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
node.save()
