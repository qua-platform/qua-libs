## Xarray Data Fetcher

The `XarrayDataFetcher` is used to fetch data from a QM job and return an `xarray.Dataset` with the acquired data.
It can both acquire data while a job is running, and when the job is done.

### Sweep axes

The `XarrayDataFetcher` expects a dictionary of `xarray.DataArray` with the sweep axes.
These are used to construct the data arrays and attach the correct coordinates.

In some cases, the outer dimension is the qubit axisThe first axis can be an extra dimension such as "qubits"

### Basic example

```python
import xarray as xr
import numpy as np
from qualang_tools.results import XarrayDataFetcher, progress_counter

axes = {
    "qubits": xr.DataArray(["q0", "q1"]),
    "num_pi_pulses": xr.DataArray([1, 2, 3]),
    "amplitudes": xr.DataArray([0.1, 0.2, 0.3]),
}

with program as prog:
    # QUA program with n_avg averaging iterations

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.execute(prog)

data_fetcher = XarrayDataFetcher(job, axes)
for dataset in data_fetcher:
    progress_counter(data_fetcher["n"], n_avg, start_time=data_fetcher.t_start)
```
