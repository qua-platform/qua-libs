import uuid

import numpy as np
import xarray as xr
from qiskit import QuantumCircuit
from qiskit.providers import Backend, Job, JobStatus
from qiskit.qobj import QobjExperimentHeader
from qiskit.quantum_info import Chi, DensityMatrix
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import ProcessTomography, StateTomography
from qiskit_experiments.library.tomography import StateTomographyAnalysis
from qiskit_experiments.library.tomography.basis import (
    Pauli6PreparationBasis, PauliMeasurementBasis, PauliPreparationBasis)


class QuaJob(Job):
    def __init__(self, result: Result):
        job_id = uuid.uuid4().hex  # todo: see if QmJob has an id
        self._job_id = 0
        self._backend = Backend()
        super().__init__()
        self._result = result
        self._status = JobStatus.DONE

    def result(self):
        """Return job result."""
        return self._result

    def job_id(self):
        return self._job_id

    def backend(self):
        return self._backend

    def status(self):
        return self._status


def hist_da_to_qiskit_state_tomo_results(da: xr.DataArray, qubit_names: list[str]) -> Result:
    """
    requirements on da:
    - must have dimension named 'mbasis' - this is the measurement basis, values should be [0, 1, 2] ^ the number of qubits,
    where 0 is the Z basis, 1 is the X basis and 2 is the Y basis. E.g, for 2 qubits, the values should be [0, 1, 2, 3, 4, 5, 6, 7, 8]
    which correspond to [ZZ, ZX, ZY, XZ, XX, XY, YZ, YX, YY]
    - must have dimension named 'value' (?????) with the values of the histogram where the keys are integers representing the
      state of the qubits in the measurement basis, e.g. for 2 qubits and measurement basis 'Z' the keys are 0, 1, 2, 3
      where 0 is the state |00>, 1 is |01>, 2 is |10> and 3 is |11>

    """
    num_qubits = len(qubit_names)
    results = []
    for mi, mb in enumerate(da.mbasis):
        header = QobjExperimentHeader(creg_sizes=[['c_tomo', num_qubits]],
                                      memory_slots=num_qubits,
                                      n_qubits=num_qubits,
                                      qreg_sizes=[['q', num_qubits]],
                                      name=f'StateTomography_' +
                                      str(tuple(mb.values.tolist())),
                                      metadata={
                                                'clbits': list(range(num_qubits)),
                                                'cond_clbits': None,
                                                'm_idx': list(mb.values.tolist())}
        )
        shots = da.sel(mbasis=mb).sum().values
        hist = da.sel(mbasis=mb).values
        results.append(ExperimentResult(shots=shots,
                                        success=True,
                                        data=ExperimentResultData(
                                            {hex(i): hist[i] for i in range(len(hist))}),
                                        meas_level=2,
                                        header=header
                                        ))

    job = QuaJob(Result('name', 0, 0, 0, True, results=results))

    experiment = StateTomography(QuantumCircuit(num_qubits), measurement_qubits=list(range(num_qubits)),
                                 measurement_basis=PauliMeasurementBasis(), physical_qubits=list(range(num_qubits)))
    # experiment.analysis.set_options(fitter='linear_lstsq')
    expdat = ExperimentData(experiment)
    expdat.add_jobs(job)
    expdat = experiment.analysis.run(expdat)
    return expdat
