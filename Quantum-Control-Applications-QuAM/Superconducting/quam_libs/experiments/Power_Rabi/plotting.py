from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from quam_libs.lib.plot_utils import QubitGrid, grid_iter


def plot(ds, qubits, fit_results, N_pi, state_discrimination) -> Figure:
    
    """
    Plots the Rabi oscillation results and fit for a given dataset and qubits.
    Parameters:
    ds (xarray.Dataset): The dataset containing the experimental data.
    qubits (list): A list of qubit objects to be plotted.
    fit_results (dict): A dictionary containing the fit evaluation results and data max indices.
    N_pi (int): The number of pi pulses used in the experiment.
    state_discrimination (bool): A flag indicating whether state discrimination is used.
    Returns:
    Figure: The matplotlib figure object containing the plots.
    """
    

    fit_evals = fit_results["fit_evals"]
    data_max_idx = fit_results["data_max_idx"]
    
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        if N_pi == 1:
            if state_discrimination:
                ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(ax=ax, x="amp_mV")
                ax.plot(ds.abs_amp.loc[qubit] * 1e3, fit_evals.loc[qubit][0])
                ax.set_ylabel("Qubit state")
            else:
                (ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].I * 1e3).plot(ax=ax, x="amp_mV")
                ax.plot(ds.abs_amp.loc[qubit] * 1e3, 1e3 * fit_evals.loc[qubit][0])
                ax.set_ylabel("Trans. amp. I [mV]")

        elif N_pi > 1:
            if state_discrimination:
                ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].state.plot(ax=ax, x="amp_mV", y="N")
            else:
                (ds.assign_coords(amp_mV=ds.abs_amp * 1e3).loc[qubit].I * 1e3).plot(ax=ax, x="amp_mV", y="N")
            ax.set_ylabel("num. of pulses")
            ax.axvline(1e3 * ds.abs_amp.loc[qubit][data_max_idx.loc[qubit]], color="r")
        ax.set_xlabel("Amplitude [mV]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Rabi : I vs. amplitude")
    plt.tight_layout()
    plt.show()
    
    return grid.fig