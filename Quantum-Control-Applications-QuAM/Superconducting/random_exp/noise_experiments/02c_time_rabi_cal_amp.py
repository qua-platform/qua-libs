# %%
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["q1"]
    num_averages: int = 500
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 500
    num_time_steps: int = 500
    flux_point_joint_or_independent_or_arbitrary: Literal['joint', 'independent', 'arbitrary'] = "joint"    
    simulate: bool = False
    timeout: int = 100
    use_state_discrimination: bool = True
    reset_type: Literal['active', 'thermal'] = "thermal"
    drive_pulse_name: str = "x180_Square"
    drive_amp_scale: float = 0.2
    target_freq_in_Mhz: float = 10

node = QualibrationNode(
    name="02c_time_rabi_cal_amp",
    parameters=Parameters()
)

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, active_reset, readout_state

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation, oscillation


# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.unique(
    np.geomspace(
        node.parameters.min_wait_time_in_ns, node.parameters.max_wait_time_in_ns, node.parameters.num_time_steps
    )
    // 4
).astype(int)

flux_point = node.parameters.flux_point_joint_or_independent_or_arbitrary  # 'independent' or 'joint'
if flux_point == "arbitrary":
    detunings = {q.name : q.arbitrary_intermediate_frequency for q in qubits}
    arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    detunings = {q.name: 0.0 for q in qubits}

with program() as t1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint" or "arbitrary":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        # Wait for the flux bias to settle
        for qb in qubits:
            wait(1000, qb.z.name)

        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(t, idle_times):
                if node.parameters.reset_type == "active":
                    active_reset(qubit)
                else:
                    qubit.resonator.wait(qubit.thermalization_time * u.ns)
                    qubit.align()
                qubit.xy.play(node.parameters.drive_pulse_name, amplitude_scale=node.parameters.drive_amp_scale, duration = t)
                qubit.align()

                
                # Measure the state of the resonators
                if node.parameters.use_state_discrimination:
                    readout_state(qubit, state[i])
                    save(state[i], state_st[i])
                else:
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, t1, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(t1)
        # Get results from QUA program
        for i in range(num_qubits):
            print(f"Fetching results for qubit {qubits[i].name}")
            data_list = ["n"]
            results = fetching_tool(job, data_list, mode="live")
            while results.is_processing():
            # Fetch results
                fetched_data = results.fetch_all()
                n = fetched_data[0]

                progress_counter(n, n_avg, start_time=results.start_time)


# %%
# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})

    ds = ds.assign_coords(idle_time=4*ds.idle_time/1e3)  # convert to usec
    ds.idle_time.attrs = {'long_name': 'idle time', 'units': 'usec'}

# %% {Data_analysis}
if not node.parameters.simulate:
    fit_results = {}
    if node.parameters.use_state_discrimination:
        fit_data = fit_oscillation(ds.state, 'idle_time')
    else:
        fit_data = fit_oscillation(ds.I, 'idle_time')    
    # Fit the power Rabi oscillations
    fit_evals = oscillation(
        ds.idle_time,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="f"),
        fit_data.sel(fit_vals="phi"),
        fit_data.sel(fit_vals="offset"),
    )

    def fit_cosine_with_fft_guess(ds, qubits, use_state_discrimination=True):
        """
        Fit dataset to cosine function using FFT for initial frequency guess
        
        Parameters:
        -----------
        ds : xarray.Dataset
            Dataset containing the measurements
        qubits : list
            List of qubit objects to analyze
        use_state_discrimination : bool
            Whether to fit state or I quadrature data
            
        Returns:
        --------
        dict : Dictionary containing fit results for each qubit with keys:
            'A': amplitude
            'f': frequency in MHz
            'phi': phase in radians 
            'offset': vertical offset
            'fit_func': fitted cosine function
        """
        def cosine(x, A, f, phi, offset):
            return A * np.cos(2*np.pi*f*x + phi) + offset
            
        from scipy.optimize import curve_fit
        from scipy.fft import fft, fftfreq
        
        fit_results = {}
        
        for q in qubits:
            # Get data for this qubit
            x_data = ds.idle_time.values
            if use_state_discrimination:
                y_data = ds.state.sel(qubit=q.name).values
            else:
                y_data = ds.I.sel(qubit=q.name).values
                
            # Use FFT to guess frequency
            N = len(x_data)
            T = (x_data[1] - x_data[0])  # sampling interval
            yf = fft(y_data)
            xf = fftfreq(N, T)
            
            # Get positive frequencies only
            xf = xf[:N//2]
            yf = np.abs(yf[:N//2])
            
            # Find peak frequency
            f_guess = abs(xf[np.argmax(yf[1:])+1])  # Skip DC component
            
            # Initial parameter guesses
            A_guess = (np.max(y_data) - np.min(y_data))/2
            offset_guess = np.mean(y_data)
            p0 = [A_guess, f_guess, 0, offset_guess]
            
            # Perform the fit
            popt, pcov = curve_fit(cosine, x_data, y_data, p0=p0)
            
            # Store fit results
            fit_results[q.name] = {
                'A': popt[0],
                'f': popt[1],
                'phi': popt[2],
                'offset': popt[3],
                'fit_func': lambda x, p=popt: cosine(x, *p)
            }
            
            print(f"\nFit results for {q.name}:")
            print(f"Initial frequency guess from FFT: {f_guess:.3f} MHz")
            print(f"Final fit results:")
            print(f"Amplitude: {popt[0]:.3f}")
            print(f"Frequency: {popt[1]:.3f} MHz")
            print(f"Phase: {popt[2]:.3f} rad")
            print(f"Offset: {popt[3]:.3f}")
        
        return fit_results
        
    # Perform the fit
    fit_data = fit_cosine_with_fft_guess(ds, qubits, node.parameters.use_state_discrimination)

    fit_evals = {}
    for q in qubits:
        fit_evals[q.name] = fit_data[q.name]['fit_func'](ds.idle_time)
    
# Save fitting results
    for q in qubits:
        fit_results[q.name] = {}
        fit_results[q.name]["f_fit"] = fit_data[q.name]["f"]
        fit_results[q.name]["phi_fit"] = fit_data[q.name]["phi"]
        fit_results[q.name]["phi_fit"] = fit_results[q.name]["phi_fit"] - np.pi * (fit_results[q.name]["phi_fit"] > np.pi / 2)
        fit_results[q.name]["amp_fit"] = node.parameters.drive_amp_scale * node.parameters.target_freq_in_Mhz / fit_results[q.name]['f_fit']
    node.results["fit_results"] = fit_results
    print(f"found frequency of {fit_results[q.name]['f_fit']:.2f} MHz for qubit {q.name} with scale {node.parameters.drive_amp_scale}, to reach {node.parameters.target_freq_in_Mhz:.2f} MHz, multiply amplitude by {fit_results[q.name]['amp_fit']:.2f}")

# %% {Plotting}
if not node.parameters.simulate:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.sel(qubit = qubit['qubit']).state.plot(ax = ax)
            
            ax.set_ylabel('State')
        else:
            ds.sel(qubit = qubit['qubit']).I.plot(ax = ax)
            ax.set_ylabel('I (V)')
        ax.plot(ds.idle_time, fit_evals[qubit['qubit']])
        ax.set_xlabel("Idle time [usec]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Rabi : I vs. amplitude")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

# %%
if not node.parameters.simulate:
    with node.record_state_updates():
        for q in qubits:
            q.xy.operations[node.parameters.drive_pulse_name].amplitude *= fit_results[q.name]["amp_fit"]

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
