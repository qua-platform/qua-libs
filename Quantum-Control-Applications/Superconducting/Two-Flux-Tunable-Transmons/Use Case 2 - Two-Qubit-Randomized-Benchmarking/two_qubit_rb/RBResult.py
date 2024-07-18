import dataclasses
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


@dataclasses.dataclass
class RBResult:
    circuit_depths: list[int]
    num_repeats: int
    num_averages: int
    state: np.ndarray

    def __post_init__(self):
        self.data = xr.Dataset(
            data_vars={"state": (["circuit_depth", "repeat", "average"], self.state)},
            coords={
                "circuit_depth": self.circuit_depths,
                "repeat": range(self.num_repeats),
                "average": range(self.num_averages),
            },
        )

    def plot_hist(self, n_cols=3):
        if len(self.circuit_depths) < n_cols:
            n_cols = len(self.circuit_depths)
        n_rows = max(int(np.ceil(len(self.circuit_depths) / n_cols)), 1)
        plt.figure()
        for i, circuit_depth in enumerate(self.circuit_depths, start=1):
            ax = plt.subplot(n_rows, n_cols, i)
            self.data.state.sel(circuit_depth=circuit_depth).plot.hist(ax=ax, xticks=range(4))
        plt.tight_layout()

    def plot_decay(self):
        fidelity = (self.data.state == 0).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)
        fidelity.rename("fidelity").plot.line()

    def plot_fidelity(self):
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
        decay_curve = self.get_decay_curve()

        popt, _ = curve_fit(rb_decay_curve, self.circuit_depths, decay_curve, p0=[0.75, -0.1, 0.25], maxfev=10000)
        A, alpha, B = popt

        return A, alpha, B

    def get_fidelity(self, alpha):
        # Calculate the average error rate per Clifford
        n_qubits = 2  # Assuming 2 qubits as per the context
        d = 2**n_qubits
        r = 1 - alpha - (1 - alpha) / d
        fidelity = 1 - r

        return fidelity

    def get_decay_curve(self):
        return (self.data.state == 0).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)


def rb_decay_curve(x, A, alpha, B):
    return A * alpha**x + B


def get_interleaved_gate_fidelity(num_qubits: int, reference_alpha: float, interleaved_alpha: float):
    """Formula from: https://arxiv.org/pdf/1210.7011"""
    return 1 - ((2**num_qubits - 1) * (1 - interleaved_alpha / reference_alpha) / 2**num_qubits)
