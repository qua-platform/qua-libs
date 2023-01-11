import dataclasses

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


@dataclasses.dataclass
class RBResult:
    sequence_depths: list[int]
    num_repeats: int
    num_averages: int
    state: np.ndarray

    def __post_init__(self):
        self.data = xr.Dataset(
            data_vars={"state": (["sequence_depth", "repeat", "average"], self.state)},
            coords={"sequence_depth": self.sequence_depths,
                    "repeat": range(self.num_repeats),
                    "average": range(self.num_averages)}
        )

    def plot_hist(self, n_cols=3):
        if len(self.sequence_depths)< n_cols:
            n_cols = len(self.sequence_depths)
        n_rows = max(int(np.ceil(len(self.sequence_depths)/n_cols)), 1)
        plt.figure()
        for i, sequence_depth in enumerate(self.sequence_depths, start=1):
            ax = plt.subplot(n_rows, n_cols, i)
            self.data.state.sel(sequence_depth=sequence_depth).plot.hist(ax, xticks=range(4))
        plt.tight_layout()

    def plot_fidelity(self):
        fidelity = (self.data.state == 0).sum(("repeat", "average"))/(self.num_repeats*self.num_averages)
        fidelity.rename("fidelity").plot.line()