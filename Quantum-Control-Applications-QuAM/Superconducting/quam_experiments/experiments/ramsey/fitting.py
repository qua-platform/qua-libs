from dataclasses import dataclass
import numpy as np
import xarray as xr


@dataclass
class RamseyFit:
    """
    Holds the fit results from the Ramsey analysis.

    Attributes:
      freq_offset: Measured frequency offset (in Hz).
      decay: Measured T2 Ramsey decay time (in ns).
      success: Boolean flag indicating whether the fit is acceptable.
    """

    freq_offset: float
    decay: float
    success: bool

    def log_frequency_offset(self):
        print(f"Frequency offset: {self.freq_offset:.3f} Hz")

    def log_t2(self):
        print(f"T2 Ramsey: {self.decay:.3f} ns")


def fit_frequency_detuning_and_t2_decay(ds: xr.Dataset, qubits, params) -> dict:
    """
    Fit the Ramsey oscillations to extract the qubit frequency detuning and T2 Ramsey decay.

    Parameters:
        ds : xarray.Dataset
            Dataset containing the raw Ramsey measurement data.
        qubits : list
            List of qubit objects.
        params : RamseyParameters
            The node parameters.

    Returns:
        A dictionary mapping each qubit name to a RamseyFit object.
    """
    fits = {}
    # (Replace the dummy fitting below with your actual fitting routine.)
    for q in qubits:
        # Dummy fit values: use 10% of the set frequency_detuning as offset and a constant decay.
        freq_offset = 0.1 * params.frequency_detuning_in_mhz * 1e6  # dummy value in Hz
        decay = 1000.0  # dummy T2 Ramsey in ns
        fits[q.name] = RamseyFit(freq_offset=freq_offset, decay=decay, success=True)
    return fits
