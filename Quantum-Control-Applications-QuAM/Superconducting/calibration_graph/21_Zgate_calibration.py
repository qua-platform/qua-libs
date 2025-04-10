# %%
"""
RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing resonant gates.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency (f_01) in the state.
    - Save the current state by calling machine.save("quam")
"""
# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam.components.pulses import SquarePulse
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.multi_user import qm_session
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from scipy.optimize import curve_fit
import xarray as xr
from quam_libs.lib import guess

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 1000
    ref_frequnecy_MHz: float = 0.0
    num_points: int = 100
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    reset_type: Literal['active', 'thermal'] = "active"
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True

node = QualibrationNode(
    name="21_Zgate_calibration",
    parameters=Parameters()
)


# %% {Utility_functions}

def cosine_func(x, A, f, offset):
    return A * np.cos(2 * np.pi * f * x) + offset

def find_maximum_cosine(A, f, xmin, xmax):
    """
    Finds the x value where the cosine function A * cos(2 * pi * f * x + phi) reaches its maximum
    within the range [xmin, xmax].
    
    Parameters:
        A: Amplitude of the cosine function
        f: Frequency of the cosine function
        phi: Phase shift of the cosine function
        xmin: Minimum value of x in the range
        xmax: Maximum value of x in the range
        
    Returns:
        x_max: The x value where the cosine function reaches its maximum within the given range
        max_value: The maximum value of the cosine function within the range
    """
    # The period of the cosine function
    period = 1 / f
    
    # Find the general solution for the maximum (cosine = 1)
    def x_for_max(n):
        return (2 * n * np.pi ) / (2 * np.pi * f)
    
    # Calculate n values within the range [xmin, xmax]
    n_min = np.ceil((2 * np.pi * f * xmin ) / (2 * np.pi))
    n_max = np.floor((2 * np.pi * f * xmax ) / (2 * np.pi))
    
    # Generate all possible n values within the range
    n_values = np.arange(n_min, n_max + 1, 1)
    
    # Find corresponding x values
    x_values = x_for_max(n_values)
    
    # Filter x values to be within [xmin, xmax]
    x_values_in_range = x_values[(x_values >= xmin) & (x_values <= xmax)]
    
    # The maximum occurs at the first valid x value, since cosine repeats periodically
    if len(x_values_in_range) > 0:
        x_max = x_values_in_range[0]
        max_value = A * np.cos(2 * np.pi * f * x_max )
        return x_max, max_value
    else:
        return None, None
# %%

# %% {Initialize_QuAM_and_QOP}

# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

# for qubit in qubits[3:]:
#     qubit.xy.operations['x90'].amplitude = 0
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()

num_qubits = len(qubits)

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# assert operation_time % 4 == 0, "Operation time must be a multiple of 4"

quad_terms = {qubit.name: qubit.freq_vs_flux_01_quad_term for qubit in qubits}

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

frequencies = {qubit.name: 1e9*np.linspace(0, 1.5 / qubit.xy.operations['x180'].length, node.parameters.num_points) + node.parameters.ref_frequnecy_MHz*1e6 for qubit in qubits}

fluxes = {qubit.name: np.sqrt(-frequencies[qubit.name]/quad_terms[qubit.name]) for qubit in qubits}
reset_type = node.parameters.reset_type

# %%
with program() as ramsey:
    I, I_st, Q, Q_st, _, n_st = qua_declaration(num_qubits=num_qubits)

    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    shots = [declare(int) for _ in range(num_qubits)]

    t = [declare(int) for _ in range(num_qubits)]  # QUA variable for the idle time
    flux = [declare(fixed) for _ in range(num_qubits)]  # QUA variable for the flux dc level

    if node.parameters.multiplexed:
        for i , qubit in enumerate(qubits):
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)

    for i, qubit in enumerate(qubits):
        if not node.parameters.multiplexed:
            # Bring the active qubits to the desired frequency point
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(shots[i], 0, shots[i] < n_avg, shots[i] + 1):
            save(shots[i], n_st)
            with for_each_(flux[i], fluxes[qubit.name]):
                if reset_type == "active":
                    active_reset(qubit)
                else:
                    qubit.resonator.wait(machine.thermalization_time * u.ns)
                    qubit.align()
                qubit.align()

                qubit.xy.play("x90")
                
                qubit.align()
                wait(10, qubit.z.name)
                qubit.z.play("const", amplitude_scale = flux[i] / qubit.z.operations['const'].amplitude, duration=qubit.xy.operations['x180'].length // 4)
                wait(10, qubit.z.name)
                qubit.align()
                
                qubit.xy.play("x90")
                qubit.align()
                # Measure the state of the resonators
                readout_state(qubit, state[i])
                save(state[i], state_st[i])
                
                qubit.align()
        
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].buffer(len(fluxes[qubits[0].name])).average().save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
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
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ramsey)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # fetch results
            n = results.fetch_all()[0]
            # progress bar
            progress_counter(n, n_avg, start_time=results.start_time)



# %%
if not node.parameters.simulate:
    handles = job.result_handles
    ds = fetch_results_as_xarray(handles, qubits, {"flux_int": np.arange(len(fluxes[qubits[0].name]))})
    node.results = {}
    
    def flux_num_to_flux(q):
        return fluxes[q.name]
    
    def detuning(q, flux):
        return -1e-6 * q.freq_vs_flux_01_quad_term * flux**2

    ds = ds.assign_coords(
        {"flux": (["qubit", "flux"], np.array([flux_num_to_flux(q) for q in qubits]))}
    )

    ds = ds.assign_coords(
        {"detuning": (["qubit", "flux"], np.array([detuning(q, ds.sel(qubit=q.name).flux) for q in qubits]))}
    )
    ds.detuning.attrs["long_name"] = "Detuning"
    ds.detuning.attrs["units"] = "MHz"

    node.results['ds'] = ds
    # %%

    fit_data = {}
    fitted = []

    for qubit in qubits:
        x = ds.sel(qubit = qubit.name).detuning
        y = ds.sel(qubit = qubit.name).state
        # Initial guess for parameters
        A_guess = (y.max() - y.min()) / 2
        f_guess = guess.frequency(x,y)  
        f_guess = 1.5 / (x.max() - x.min())
        offset_guess = y.mean()
        
        try:
            popt, _ = curve_fit(cosine_func, x, y, p0=[A_guess, f_guess,  offset_guess])
            A_fit, f_fit, offset_fit = popt
            
            # Verify that A_fit is positive
            # if A_fit < 0:
                # A_fit = -A_fit
                # phi_fit += np.pi  # Adjust phase by pi to maintain the same curve shape
            
            Z_id, _ = find_maximum_cosine(A_fit, f_fit, x.min().values, x.max().values)
            Z_90 = Z_id + 1/(4*f_fit)
            Z_180 = Z_id + 1/(2*f_fit)
            Z_270 = Z_id + 3/(4*f_fit)           
            fit_data[qubit.name] = {'A': A_fit, 'f': f_fit, 'offset': offset_fit,
                                    'Z_id': Z_id, 'Z_90': Z_90, 'Z_180': Z_180, 'Z_270': Z_270}
            fitted.append(cosine_func(x, *popt))
            
        except RuntimeError:
            print(f"Curve fit failed for {qubit.name}")
    # %%

    # Concatenate the fitted DataArrays along the 'qubit' dimension
    fitted_da = xr.concat(fitted, dim='qubit')

    # Assign qubit names as coordinates
    fitted_da = fitted_da.assign_coords(qubit=[q.name for q in qubits])

    # Add the fitted DataArray to the dataset
    ds['fitted'] = fitted_da
    node.results['ds'] = ds
    # %%


    # Plot the original data and fitted curves for each qubit

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        flux_q = ds.sel(qubit=qubit['qubit']).flux
        det_q = ds.sel(qubit=qubit['qubit']).detuning
        ax.plot(det_q, ds.sel(qubit=qubit['qubit']).state, marker='o', linestyle='', label='Data')
        ax.plot(det_q, ds.sel(qubit=qubit['qubit']).fitted, label='Fitted')
        # ds.sel(qubit=qubit['qubit']).state.plot(ax=ax, x = 'detuning', marker='o', linestyle='', label='Data')
        # ds.sel(qubit=qubit['qubit']).fitted.plot(ax=ax, x = 'detuning', label='Fitted')
        ax.axvline(x = fit_data[qubit['qubit']]['Z_id'], color = 'k', linestyle = '--')
        ax.axvline(x = fit_data[qubit['qubit']]['Z_90'], color = 'k', linestyle = '--')
        ax.axvline(x = fit_data[qubit['qubit']]['Z_180'], color = 'k', linestyle = '--')
        ax.axvline(x = fit_data[qubit['qubit']]['Z_270'], color = 'k', linestyle = '--')
        ax.set_xlim(0, det_q.max().values)
        # Create a second x-axis for flux with proper non-linear mapping
        def detuning_to_flux(det):
            return 1e3 * np.sqrt(-1e6 * det / machine.qubits[qubit['qubit']].freq_vs_flux_01_quad_term)

        def flux_to_detuning(flux):
            return -1e-6 * (flux/1e3)**2 * machine.qubits[qubit['qubit']].freq_vs_flux_01_quad_term

        # Create secondary axis with the transformation functions
        ax2 = ax.secondary_xaxis('top', functions=(flux_to_detuning, detuning_to_flux))
        ax2.set_xlabel('Flux (mV)')

        ax.set_title(qubit['qubit'])
        ax.set_xlabel('Detuning (MHz)')
        ax.set_ylabel('State')
        # ax.legend()
    grid.fig.suptitle('Ramsey: Data and Fitted Curves')
    plt.tight_layout()
    plt.show()
    # node.results['figure_fitted'] = grid.fig
    node.results['figure_fitted'] = []


    # %%
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qubit in qubits:
                qubit.z.operations['z0'] = SquarePulse(
                    length=qubit.xy.operations['x180'].length,
                    amplitude=np.sqrt(-1e6*fit_data[qubit.name]['Z_id']/qubit.freq_vs_flux_01_quad_term)
                )
                qubit.z.operations['z90'] = SquarePulse(length=qubit.xy.operations['x180'].length, amplitude=np.sqrt(-1e6*fit_data[qubit.name]['Z_90']/qubit.freq_vs_flux_01_quad_term))
                qubit.z.operations['z180'] = SquarePulse(length=qubit.xy.operations['x180'].length, amplitude=np.sqrt(-1e6*fit_data[qubit.name]['Z_180']/qubit.freq_vs_flux_01_quad_term))
                qubit.z.operations['-z90'] = SquarePulse(length=qubit.xy.operations['x180'].length, amplitude=np.sqrt(-1e6*fit_data[qubit.name]['Z_270']/qubit.freq_vs_flux_01_quad_term))
                print(f"{qubit.name}  Z180: {np.sqrt(-1e6*fit_data[qubit.name]['Z_180']/qubit.freq_vs_flux_01_quad_term)}")
    # %%

    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%

# %%
