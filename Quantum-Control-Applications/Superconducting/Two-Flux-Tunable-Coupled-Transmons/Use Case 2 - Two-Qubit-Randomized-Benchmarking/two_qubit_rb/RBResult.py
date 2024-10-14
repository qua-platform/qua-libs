import dataclasses
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
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
            data_vars={"state": (["circuit_depth", "repeat", "average"], self.state)},
            coords={
                "circuit_depth": self.circuit_depths,
                "repeat": range(self.num_repeats),
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
        The fitted curve is overlaid with the raw data points.
        """
        A, alpha, B = self.fit_exponential()
        fidelity = self.get_fidelity(alpha)

        plt.figure()
        plt.plot(self.circuit_depths, self.get_decay_curve(), "o", label="Data")
        plt.plot(
            self.circuit_depths,
            rb_decay_curve(np.array(self.circuit_depths), A, alpha, B),
            "-",
            label=f"Fidelity={fidelity*100:.3f}%\nalpha={alpha:.4f}",
        )
        plt.xlabel("Circuit Depth")
        plt.ylabel("Fidelity")
        plt.title("2Q Randomized Benchmarking Fidelity")
        plt.legend()
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

        popt, _ = curve_fit(rb_decay_curve, self.circuit_depths, decay_curve, p0=[0.75, -0.1, 0.25], maxfev=10000)
        A, alpha, B = popt

        return A, alpha, B

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
        r = 1 - alpha - (1 - alpha) / d
        fidelity = 1 - r

        return fidelity

    def get_decay_curve(self):
        """
        Calculates the decay curve from the RB data.

        Returns:
            np.ndarray: Decay curve representing the fidelity as a function of circuit depth.
        """
        return (self.data.state == 0).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)


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


def get_interleaved_gate_fidelity(num_qubits: int, reference_alpha: float, interleaved_alpha: float):
    """
    Calculates the interleaved gate fidelity using the formula from https://arxiv.org/pdf/1210.7011.

    Args:
        num_qubits (int): Number of qubits involved.
        reference_alpha (float): Decay constant from the reference RB experiment.
        interleaved_alpha (float): Decay constant from the interleaved RB experiment.

    Returns:
        float: Estimated interleaved gate fidelity.
    """
    return 1 - ((2**num_qubits - 1) * (1 - interleaved_alpha / reference_alpha) / 2**num_qubits)
