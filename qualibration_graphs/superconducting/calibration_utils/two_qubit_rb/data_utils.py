import dataclasses
import re
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

@dataclasses.dataclass
class RBResult:
    """
    Class for analyzing and visualizing the results of a Randomized Benchmarking (RB) experiment.

    Attributes:
        circuit_depths (list[int]): List of circuit depths used in the RB experiment.
        num_repeats (int): Number of repeated sequences at each circuit depth.
        num_averages (int): Number of averages for each sequence.
        state (np.ndarray): Measured states from the RB experiment.
    """

    circuit_depths: list[int]
    num_repeats: int
    num_averages: int
    state: np.ndarray

    def __post_init__(self):
        """
        Initializes the xarray Dataset to store the RB experiment data.
        """
        self.data = xr.Dataset(
            data_vars={"state": (["repeat", "circuit_depth", "average"], self.state)},
            coords={
                "repeat": range(self.num_repeats),
                "circuit_depth": self.circuit_depths,
                "average": range(self.num_averages),
            },
        )

    def plot_hist(self, n_cols=3):
        """
        Plots histograms of the N-qubit state distribution at each circuit depth.

        Args:
            n_cols (int): Number of columns in the plot grid. Adjusted if fewer circuit depths are provided.
        """
        if len(self.circuit_depths) < n_cols:
            n_cols = len(self.circuit_depths)
        n_rows = max(int(np.ceil(len(self.circuit_depths) / n_cols)), 1)
        plt.figure()
        for i, circuit_depth in enumerate(self.circuit_depths, start=1):
            ax = plt.subplot(n_rows, n_cols, i)
            self.data.state.sel(circuit_depth=circuit_depth).plot.hist(ax=ax, xticks=range(4))
        plt.tight_layout()

    def plot(self):
        """
        Plots the raw recovery probability decay curve as a function of circuit depth.
        The curve is plotted using the averaged probability and without any fitting.
        """
        recovery_probability = (self.data.state == 0).sum(("repeat", "average")) / (
            self.num_repeats * self.num_averages
        )
        recovery_probability.rename("Recovery Probability").plot.line()

    def plot_with_fidelity(self):
        """
        Plots the RB fidelity as a function of circuit depth, including a fit to an exponential decay model.
        The fitted curve is overlaid with the raw data points, and error bars are included.
        
        Note: If fit() has been called successfully, the fitted curve and fidelity metrics will be displayed.
        If fit() failed or hasn't been called, only the raw data will be plotted.
        
        Returns:
            matplotlib.figure.Figure: The figure object containing the plot.
        """
        # std of average
        error_bars = (self.data == 0).stack(combined=("average", "repeat")).std(dim="combined").state.data / np.sqrt(self.num_repeats * self.num_averages)

        fig = plt.figure()
        plt.errorbar(
            self.circuit_depths,
            self.get_decay_curve(),
            yerr=error_bars,
            fmt=".",
            capsize=2,
            elinewidth=0.5,
            color="blue",
            label="Experimental Data",
        )

        # Only plot fit curve if fit parameters exist
        try:
            # Check if fit parameters exist by trying to access them
            A = self.A
            alpha = self.alpha
            B = self.B
            circuit_depths_smooth_axis = np.linspace(self.circuit_depths[0], self.circuit_depths[-1], 100)
            plt.plot(
                circuit_depths_smooth_axis,
                rb_decay_curve(np.array(circuit_depths_smooth_axis), A, alpha, B),
                color="red",
                linestyle="--",
                label="Exponential Fit",
            )
        except AttributeError:
            # Fit parameters don't exist, skip plotting fit curve
            pass

        # Only show fidelity title if fit was successful
        try:
            fidelity = self.fidelity
            if isinstance(self, InterleavedRBResult):
                title = f"target gate fidelity = {fidelity * 100:.2f}%"
            else:
                title = f"2Q average Clifford fidelity = {fidelity * 100:.2f}%"
                
            plt.text(
                0.5,
                0.95,
                title,
                horizontalalignment="center",
                verticalalignment="top",
                fontdict={"fontsize": "large", "fontweight": "bold"},
                transform=plt.gca().transAxes,
            )
        except AttributeError:
            # Show warning if fit failed
            plt.text(
                0.5,
                0.95,
                "Fit failed - insufficient data points",
                horizontalalignment="center",
                verticalalignment="top",
                fontdict={"fontsize": "large", "fontweight": "bold", "color": "red"},
                transform=plt.gca().transAxes,
            )
        
        # Add average gate fidelity if it was calculated
        try:
            avg_gate_fidelity = self.average_gate_fidelity
            plt.text(
                0.5,
                0.88,
                f"Average Gate Fidelity = {avg_gate_fidelity * 100:.2f}%",
                horizontalalignment="center",
                verticalalignment="top",
                fontdict={"fontsize": "large", "fontweight": "bold"},
                transform=plt.gca().transAxes,
            )
        except AttributeError:
            # Average gate fidelity not available, skip
            pass

        plt.xlabel("Circuit Depth")
        plt.ylabel(r"Probability to recover to $|00\rangle$")
        plt.legend(framealpha=0)
        
        return fig
        
        

    def plot_two_qubit_state_distribution(self):
        """
        Plot how the two-qubit state is distributed as a function of circuit-depth on average.
        """
        plt.plot(
            self.circuit_depths,
            (self.data.state == 0).mean(dim="average").mean(dim="repeat").data,
            label=r"$|00\rangle$",
            marker=".",
            color="c",
            linewidth=3,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 1).mean(dim="average").mean(dim="repeat").data,
            label=r"$|01\rangle$",
            marker=".",
            color="b",
            linewidth=1,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 2).mean(dim="average").mean(dim="repeat").data,
            label=r"$|10\rangle$",
            marker=".",
            color="y",
            linewidth=1,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 3).mean(dim="average").mean(dim="repeat").data,
            label=r"$|11\rangle$",
            marker=".",
            color="r",
            linewidth=1,
        )
        plt.axhline(0.25, color="grey", linestyle="--", linewidth=2, label="2Q mixed-state")

        plt.xlabel("Circuit Depth")
        plt.ylabel(r"Probability to recover to a given state")
        plt.title("2Q State Distribution vs. Circuit Depth")
        plt.legend(framealpha=0, title=r"2Q State $\mathbf{|q_cq_t\rangle}$", title_fontproperties={"weight": "bold"})
        plt.show()

    def fit_exponential(self):
        """
        Fits the decay curve of the RB data to an exponential model.

        Returns:
            tuple: Fitted parameters (A, alpha, B) where:
                - A is the amplitude.
                - alpha is the decay constant.
                - B is the offset.
        """
        decay_curve = self.get_decay_curve()

        popt, _ = curve_fit(rb_decay_curve, self.circuit_depths, decay_curve, p0=[0.75, 0.9, 0.25], maxfev=10000)
        A, alpha, B = popt
        
        self.alpha = alpha

        return A, alpha, B

    def fit(self, average_layers_per_clifford=None, average_gates_per_2q_layer=None):
        """
        Fits the RB data and calculates all error and fidelity metrics.
        
        Args:
            average_layers_per_clifford (float, optional): Average number of 2q layers per Clifford.
                If provided, will calculate error_per_2q_layer, error_per_gate, and average_gate_fidelity.
            average_gates_per_2q_layer (float, optional): Average number of gates per 2q layer.
                If provided, will calculate error_per_2q_layer, error_per_gate, and average_gate_fidelity.
        
        This method calculates and stores the following attributes:
            - A, alpha, B: Fitted exponential parameters
            - fidelity: 2Q Clifford fidelity
            - epc: Error per Clifford
            - error_per_2q_layer: Error per 2q layer (if constants provided)
            - error_per_gate: Error per gate (if constants provided)
            - average_gate_fidelity: Average gate fidelity (if constants provided)
        """
        # Fit exponential decay
        A, alpha, B = self.fit_exponential()
        self.A = A
        self.alpha = alpha
        self.B = B
        
        # Calculate fidelity and error per Clifford
        fidelity = self.get_fidelity(alpha)
        self.fidelity = fidelity
        self.epc = 1 - self.fidelity
        
        # Calculate additional metrics if constants are provided
        if average_layers_per_clifford is not None and average_gates_per_2q_layer is not None:
            self.error_per_2q_layer = (1 - fidelity) / average_layers_per_clifford
            self.error_per_gate = self.error_per_2q_layer / average_gates_per_2q_layer
            self.average_gate_fidelity = 1 - self.error_per_gate

    def get_fidelity(self, alpha):
        """
        Calculates the average fidelity per Clifford based on the decay constant.

        Args:
            alpha (float): Decay constant from the exponential fit.

        Returns:
            float: Estimated average fidelity per Clifford.
        """
        n_qubits = 2  # Assuming 2 qubits as per the context
        d = 2**n_qubits
        r = 1 - alpha - (1 - alpha) / d # error per clifford
        fidelity = 1 - r

        return fidelity

    def get_decay_curve(self):
        """
        Calculates the decay curve from the RB data.

        Returns:
            np.ndarray: Decay curve representing the fidelity as a function of circuit depth.
        """
        return (self.data.state == 0).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)
    
    def get_decay_curve_1q(self, qubit_index: int):
        """
        Calculates the decay curve for a single qubit.
        
        Args:
            qubit_index (int): Index of the qubit to calculate the decay curve for.

        Returns:
            np.ndarray: Decay curve representing the fidelity as a function of circuit depth.
        """
        
        if qubit_index == 0:
            return ((self.data.state == 0) | (self.data.state == 1)).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)
        elif qubit_index == 1:
            return ((self.data.state == 0) | (self.data.state == 2)).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)
        else:
            raise ValueError(f"Qubit index {qubit_index} not supported")
        


def rb_decay_curve(x, A, alpha, B):
    """
    Exponential decay model for RB fidelity.

    Args:
        x (array-like): Circuit depths.
        A (float): Amplitude of the decay.
        alpha (float): Decay constant.
        B (float): Offset of the curve.

    Returns:
        np.ndarray: Calculated decay curve.
    """
    return A * alpha**x + B


class InterleavedRBResult(RBResult):
    """
    Class for analyzing and visualizing the results of a Interleaved Randomized Benchmarking (IRB) experiment.
    """
    standard_rb_alpha: float = 1
    
    def __init__(self, standard_rb_alpha: float, circuit_depths: list[int], num_repeats: int, num_averages: int, state: np.ndarray):
        super().__init__(circuit_depths, num_repeats, num_averages, state)
        self.standard_rb_alpha = standard_rb_alpha

    def get_fidelity(self, alpha: float):
        """
        Calculates the interleaved gate fidelity using the formula from https://arxiv.org/pdf/1203.4550.
        """
        return 1 - ((2**2 - 1) * (1 - alpha / self.standard_rb_alpha) / 2**2)


def extract_string(handle: Any) -> Optional[str]:
    """Return the stream name prefix before a trailing integer, e.g. ``state1`` -> ``state``."""
    if handle is None:
        return None
    s = str(handle)
    m = re.match(r"^(.*?)(\d+)$", s)
    return m.group(1) if m else None


def fetch_results_as_xarray(
    handles: Any,
    qubits: List[Any],
    measurement_axis: Dict[str, Any],
    keys: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Fetch QUA stream results into an xarray dataset (same layout as qualibration_libs ``fetch_results_as_xarray``).

    Parameters:
    - handles: Stream handles, e.g. ``job.result_handles`` after execution.
    - qubits: Objects with a ``name`` attribute (one per stream index).
    - measurement_axis: Sweep coordinates, e.g.
      ``{"sequence": range(...), "depths": [...], "shots": range(...)}``.
    - keys: Optional list of stream keys; default is all keys on ``handles``.
    """
    if keys is None:
        stream_handles = list(handles.keys())
    else:
        stream_handles = list(keys)

    meas_vars = sorted(
        {extract_string(h) for h in stream_handles if extract_string(h) is not None}
    )
    values = [
        [handles.get(f"{meas_var}{i + 1}").fetch_all() for i, _q in enumerate(qubits)]
        for meas_var in meas_vars
    ]
    values_arr = np.array(values)
    if values_arr.shape[-1] == 1:
        values_arr = np.squeeze(values_arr, axis=-1)
    measurement_axis = dict(measurement_axis)
    measurement_axis["qubit"] = [qubit.name for qubit in qubits]
    measurement_axis = {key: measurement_axis[key] for key in reversed(measurement_axis.keys())}

    return xr.Dataset(
        {
            f"{meas_var}": ([key for key in measurement_axis.keys()], values_arr[i])
            for i, meas_var in enumerate(meas_vars)
        },
        coords=measurement_axis,
    )
