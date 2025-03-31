# %%
"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e threshold (ge_threshold) in the state.
    - Save the current state by calling machine.save("quam")
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.mixture import GaussianMixture
from scipy.optimize import curve_fit

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_runs: int = 2000
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    timeout: int = 100


node = QualibrationNode(name="11e_IQ_Blobs_G_E_F_test", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

for q in qubits:
    # Check if an optimized GEF frequency exists
    if not hasattr(q, "GEF_frequency_shift"):
        q.resonator.GEF_frequency_shift = 0
    # check if an EF_x180 operation exists
    if "EF_x180" in q.xy.operations:
        GEF_operation = "EF_x180"
    else:
        GEF_operation = "x180"


### Helper functions
def find_biggest_gaussian(da):
    # Define Gaussian function
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))

    # Get histogram data
    hist, bin_edges = np.histogram(da, bins=100)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit multiple Gaussians
    initial_guess = [(hist.max(), bin_centers[hist.argmax()], (bin_centers[-1] - bin_centers[0]) / 4)]
    popt, _ = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)


    # Find the biggest Gaussian
    biggest_gaussian = {'amp': popt[0], 'mu': popt[1], 'sigma': popt[2]}
    
    return biggest_gaussian['mu']

# %% {QUA_program}
n_runs = node.parameters.num_runs  # Number of runs
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(num_qubits=num_qubits)
    I_f, I_f_st, Q_f, Q_f_st, _, _ = qua_declaration(num_qubits=num_qubits)
    n = declare(int)
    n_st = declare_stream()
    state_g, state_e, state_f = [declare(int) for _ in range(num_qubits)], [declare(int) for _ in range(num_qubits)], [declare(int) for _ in range(num_qubits)]
    state_g_st, state_e_st, state_f_st = [declare_stream() for _ in range(num_qubits)], [declare_stream() for _ in range(num_qubits)], [declare_stream() for _ in range(num_qubits)]
    
    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qubit)


        with for_(n, 0, n < n_runs, n + 1):
            # ground iq blobs for all qubits
            save(n, n_st)
            if reset_type == "active":
                active_reset_gef(qubit)
            elif reset_type == "thermal":
                wait(qubit.thermalization_time * u.ns)
            else:
                raise ValueError(f"Unrecognized reset type {reset_type}.")

            qubit.align()
            readout_state_gef(qubit, state_g[i], "readout")
            qubit.align()
            # save data
            save(state_g[i], state_g_st[i])

            if reset_type == "active":
                active_reset_gef(qubit)
            elif reset_type == "thermal":
                wait(qubit.thermalization_time * u.ns)
            else:
                raise ValueError(f"Unrecognized reset type {reset_type}.")
            qubit.align()
            qubit.xy.play("x180")
            qubit.align()
            readout_state_gef(qubit, state_e[i], "readout")
            qubit.align()
            # save data
            save(state_e[i], state_e_st[i])
            
            if reset_type == "active":
                active_reset_gef(qubit)
            elif reset_type == "thermal":
                wait(qubit.thermalization_time * u.ns)
            else:
                raise ValueError(f"Unrecognized reset type {reset_type}.")
            qubit.align()
            qubit.xy.play("x180")
            update_frequency(
                qubit.xy.name, qubit.xy.intermediate_frequency - qubit.anharmonicity
            )
            qubit.xy.play(GEF_operation)
            update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
            qubit.align()
            readout_state_gef(qubit, state_f[i], "readout")
            qubit.align()
            # save data
            save(state_f[i], state_f_st[i])

        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_g_st[i].save_all(f"state_g{i + 1}")
            state_e_st[i].save_all(f"state_e{i + 1}")
            state_f_st[i].save_all(f"state_f{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout, keep_dc_offsets_when_closing=True) as qm:
        job = qm.execute(iq_blobs)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_runs, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    ds = fetch_results_as_xarray(
        job.result_handles, qubits, {"N": np.linspace(1, n_runs, n_runs)}
    )

    # Fix the structure of ds to avoid tuples
    def extract_value(element):
        if isinstance(element, tuple):
            return element[0]
        return element

    ds = xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,  # This ensures the function is applied element-wise
        dask="parallelized",  # This allows for parallel processing
        output_dtypes=[float],  # Specify the output data type
    )

    node.results = {"ds": ds, "results": {}}
    
# %% {Data analysis}
# Create confusion matrix xarray with dimensions state_prep, state_meas, qubit
confusion = xr.DataArray(
    data=np.zeros((3, 3, len(qubits))),
    coords={
        'state_prep': ['g', 'e', 'f'],
        'state_meas': ['g', 'e', 'f'],
        'qubit': ds.qubit  # Use the qubit coordinate from the existing dataset
    },
    dims=['state_prep', 'state_meas', 'qubit']
)

# Fill in values for each state combination
for q in qubits:
    # Ground state preparation
    confusion.loc[dict(state_prep='g', state_meas='g', qubit=q.name)] = (ds.state_g==0).sel(qubit=q.name).sum(dim="N")
    confusion.loc[dict(state_prep='g', state_meas='e', qubit=q.name)] = (ds.state_g==1).sel(qubit=q.name).sum(dim="N")
    confusion.loc[dict(state_prep='g', state_meas='f', qubit=q.name)] = (ds.state_g==2).sel(qubit=q.name).sum(dim="N")
    
    # Excited state preparation
    confusion.loc[dict(state_prep='e', state_meas='g', qubit=q.name)] = (ds.state_e==0).sel(qubit=q.name).sum(dim="N")
    confusion.loc[dict(state_prep='e', state_meas='e', qubit=q.name)] = (ds.state_e==1).sel(qubit=q.name).sum(dim="N")
    confusion.loc[dict(state_prep='e', state_meas='f', qubit=q.name)] = (ds.state_e==2).sel(qubit=q.name).sum(dim="N")
    
    # Second excited state preparation
    confusion.loc[dict(state_prep='f', state_meas='g', qubit=q.name)] = (ds.state_f==0).sel(qubit=q.name).sum(dim="N")
    confusion.loc[dict(state_prep='f', state_meas='e', qubit=q.name)] = (ds.state_f==1).sel(qubit=q.name).sum(dim="N")
    confusion.loc[dict(state_prep='f', state_meas='f', qubit=q.name)] = (ds.state_f==2).sel(qubit=q.name).sum(dim="N")



confusion = confusion / n_runs

# %%


# %% {Plotting}
if not node.parameters.simulate:
    
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        ax.imshow(confusion.sel(qubit = qubit["qubit"]))
        ax.set_xticks([0, 1, 2], labels=["|g>", "|e>", "|f>"])
        ax.set_yticks([0, 1, 2], labels=["|g>", "|e>", "|f>"])
        ax.set_ylabel("Prepared")
        ax.set_xlabel("Measured")
        for p, prep in enumerate(["g", "e", "f"]):
            for m, meas in enumerate(["g", "e", "f"]):
                color = "k" if p == m else "w"
                value = confusion.sel(state_prep=prep, state_meas=meas, qubit=qubit["qubit"]).values
                ax.text(
                    m,
                    p,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                )
        ax.set_title(qubit["qubit"])

    grid.fig.suptitle("g.s. and e.s. fidelity")
    plt.tight_layout()
    plt.show()
    node.results["figure_fidelity"] = grid.fig
    plt.show()

# %% {Update_state}


# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
    
# %%
