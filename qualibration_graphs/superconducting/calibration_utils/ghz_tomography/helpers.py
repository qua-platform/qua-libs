import itertools

import numpy as np
import xarray as xr

PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_MATRICES = [PAULI_I, PAULI_X, PAULI_Y, PAULI_Z]


def generate_pauli_basis(n_qubits: int):
    """Generate the full Pauli index basis for N qubits."""
    return list(itertools.product(range(4), repeat=n_qubits))


def gen_inverse_hadamard(n_qubits: int):
    """Build the inverse N-qubit tomography transform matrix."""
    h_mat = np.array([[1, 1], [1, -1]]) / 2
    hn_mat = h_mat
    for _ in range(n_qubits - 1):
        hn_mat = np.kron(hn_mat, h_mat)
    return np.linalg.inv(hn_mat)


def get_pauli_data_nq(results_xr: xr.DataArray, n_qubits: int) -> xr.Dataset:
    """Estimate N-qubit Pauli coefficients from tomography outcomes."""
    pauli_basis = generate_pauli_basis(n_qubits)
    inverse_hadamard = gen_inverse_hadamard(n_qubits)

    labels = [",".join(map(str, op)) for op in pauli_basis]
    paulis_data = xr.Dataset(
        {
            "value": (["pauli_op"], np.zeros(len(labels))),
            "appearances": (["pauli_op"], np.zeros(len(labels), dtype=int)),
        },
        coords={"pauli_op": labels},
    )

    tomo_axes = list(results_xr.coords["tomo_axis"].values)

    for tomo_axis in tomo_axes:
        tomo_data = results_xr.sel(tomo_axis=tomo_axis).data
        pauli_data = inverse_hadamard @ tomo_data

        local_paulis = []
        for bits in itertools.product([0, 1], repeat=n_qubits):
            label = []
            for q in range(n_qubits):
                if bits[q] == 0:
                    label.append(0)
                else:
                    label.append(tomo_axis[q] + 1)
            local_paulis.append(",".join(map(str, label)))

        for i, pauli in enumerate(local_paulis):
            paulis_data.value.loc[{"pauli_op": pauli}] += pauli_data[i]
            paulis_data.appearances.loc[{"pauli_op": pauli}] += 1

    paulis_data["value"] = xr.where(
        paulis_data.appearances != 0,
        paulis_data.value / paulis_data.appearances,
        paulis_data.value,
    )

    return paulis_data


def get_density_matrix(paulis_data: xr.Dataset, n_qubits: int) -> np.ndarray:
    """Reconstruct a density matrix from N-qubit Pauli coefficients."""
    dim = 2**n_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    for op in itertools.product(range(4), repeat=n_qubits):
        p_mat = PAULI_MATRICES[op[0]]
        for k in op[1:]:
            p_mat = np.kron(p_mat, PAULI_MATRICES[k])

        key = ",".join(map(str, op))
        coeff = paulis_data["value"].sel(pauli_op=key).item()
        rho += coeff * p_mat

    rho /= 2**n_qubits
    return rho


def ghz_density_matrix(num_qubits: int, sign: int = +1):
    """Build the ideal GHZ density matrix."""
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)

    rho[0, 0] = 0.5
    rho[-1, -1] = 0.5
    rho[0, -1] = 0.5 * sign
    rho[-1, 0] = 0.5 * sign

    return rho


def ghz_state_vector(num_qubits: int, sign: int = +1):
    """Build the ideal GHZ state vector."""
    psi = np.zeros(2**num_qubits, dtype=complex)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = sign / np.sqrt(2)
    return psi
