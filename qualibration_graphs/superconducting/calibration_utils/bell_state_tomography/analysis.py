from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from calibration_utils.data_process_utils import reshape_control_target_val2dim

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Literal
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from calibration_utils.data_process_utils import reshape_control_target_val2dim


@dataclass
class FitParameters:
    """Stores the relevant qubit spectroscopy experiment fit parameters for a single qubit"""

    fidelity: float
    success: bool


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit xarray Dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the fitted results for all qubits.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if node.parameters.use_state_discrimination:
        ds = reshape_control_target_val2dim(
            ds, state_discrimination=node.parameters.use_state_discrimination
        )
    else:
        ds = reshape_control_target_val2dim(
            ds, state_discrimination=node.parameters.use_state_discrimination
        )
        ds = convert_IQ_to_V(ds, qubits=None, qubit_pairs=node.namespace["qubit_pairs"])
    return ds


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the frequency detuning and T2 decay of the Ramsey oscillations for each qubit.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    ds_fit = ds
    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add metadata (counts, density matrix, fidelity) and build fit_results."""
    # pick bits from each qubit
    qc = ds_fit.state.sel(control_target="c") # dims: (qubit_pair, n_shots, tomo_axis_control, tomo_axis_target)
    qt = ds_fit.state.sel(control_target="t") # same dims

    # build joint outcome in {0,1,2,3} and rename
    joint = (2 * qc + qt).astype(int).rename("state")

    # replace the old 'state' variable with the joint one
    ds_fit_mod = ds_fit.drop_vars("state").assign(state=joint)
    
    qubit_pairs = node.namespace["qubit_pairs"]
    num_shots = int(node.parameters.num_shots)
    states = [0,1,2,3]

    results = []
    for state in states:
        results.append((ds_fit_mod.state == state).sum(dim = "n_shots") / num_shots)
        
    results_xr = xr.concat(results, dim=xr.DataArray(states, name="state"))
    results_xr = results_xr.rename({"dim_0": "state"})
    results_xr = results_xr.stack(
            tomo_axis=['tomo_axis_target', 'tomo_axis_control'])

    # Get confusion matrix
    identity_4x4_matrix = np.eye(4).tolist()
    conf_mat = {qp.name: getattr(qp, "confusion", identity_4x4_matrix) for qp in qubit_pairs}

    corrected_results = []
    for qp in qubit_pairs:
        corrected_results_qp = [] 
        for tomo_axis_control in [0,1,2]:
            corrected_results_control = []
            for tomo_axis_target in [0,1,2]:
                results = results_xr.sel(
                    qubit_pair=qp.name,
                    tomo_axis_control=tomo_axis_control,
                    tomo_axis_target=tomo_axis_target, 
                )
                results = np.linalg.inv(np.array(conf_mat[qp.name])) @ results.data
                results = results * (results > 0)
                results = results / results.sum()
                corrected_results_control.append(results)
            corrected_results_qp.append(corrected_results_control)
        corrected_results.append(corrected_results_qp)

    # Convert corrected_results to an xarray DataArray
    corrected_results_xr = xr.DataArray(
        corrected_results,
        dims=['qubit_pair', 'tomo_axis_control', 'tomo_axis_target', 'state'],
        coords={
            'qubit_pair': [qp.name for qp in qubit_pairs],
            'tomo_axis_control': [0, 1, 2],
            'tomo_axis_target': [0, 1, 2],
            'state': ['00', '01', '10', '11']
        }
    )
    corrected_results_xr = corrected_results_xr.stack(
            tomo_axis=['tomo_axis_target', 'tomo_axis_control'])

    paulis_data = {}
    rhos = {}
    for qp in qubit_pairs:
        paulis_data[qp.name] = get_pauli_data(corrected_results_xr.sel(qubit_pair=qp.name))
        rhos[qp.name] = get_density_matrix(paulis_data[qp.name])
        
    # %%
    from scipy.linalg import sqrtm
    ideal_dat = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
    s_ideal = sqrtm(ideal_dat)
    for qp in qubit_pairs:
        fidelity = float(np.abs(np.trace(sqrtm(s_ideal @rhos[qp.name] @ s_ideal)))**2)
        print(f"Fidelity of {qp.name}: {fidelity:.3f}")
        purity = np.abs(np.trace(rhos[qp.name] @ rhos[qp.name]))
        print(f"Purity of {qp.name}: {purity:.3f}")
        print()
        node.results[f"{qp.name}_fidelity"] = fidelity
        node.results[f"{qp.name}_purity"] = purity

    # --- package results into xarray objects ------------------------------------

    # 1) Ensure the probs use a different dim name than the data var "state"
    #    (avoid name collision)
    results_xr_ms = (
        results_xr
        .rename({"state": "measured_state"})      # <-- key change
        .assign_coords(measured_state=["00", "01", "10", "11"])
    )

    raw_probs = (
        results_xr_ms
        .unstack("tomo_axis")
        .rename("raw_probs")
        .transpose("qubit_pair", "measured_state", "tomo_axis_target", "tomo_axis_control")
    )

    corrected_results_xr_ms = (
        corrected_results_xr
        .rename({"state": "measured_state"})      # <-- key change
        .assign_coords(measured_state=["00", "01", "10", "11"])
    )

    corrected_probs = (
        corrected_results_xr_ms
        .unstack("tomo_axis")
        .rename("corrected_probs")
    )

    # 2) Pauli transfer data per pair
    _pauli_list = []
    for qp in qubit_pairs:
        T = np.asarray(paulis_data[qp.name])  # e.g. (4,4)
        da = xr.DataArray(
            T,
            dims=("pauli_row", "pauli_col") if T.ndim == 2 else tuple(f"pauli_ax{i}" for i in range(T.ndim)),
        )
        _pauli_list.append(da.expand_dims(qubit_pair=[qp.name]))

    pauli_T = xr.concat(_pauli_list, dim="qubit_pair").rename("pauli_T")

    # 3) Density matrices ρ (4×4) per qubit_pair
    _rho_list = []
    for qp in qubit_pairs:
        rho = np.asarray(rhos[qp.name])
        da = xr.DataArray(
            rho,
            dims=("row", "col"),
            coords={"row": ["00","01","10","11"], "col": ["00","01","10","11"]},
        )
        _rho_list.append(da.expand_dims(qubit_pair=[qp.name]))

    density_matrix_xr = xr.concat(_rho_list, dim="qubit_pair").rename("density_matrix")

    # 4) Fidelity per qubit_pair (scalar)
    fidelity_xr = xr.DataArray(
        [float(np.abs(np.trace(sqrtm(s_ideal @ rhos[qp.name] @ s_ideal)))**2) for qp in qubit_pairs],
        dims=["qubit_pair"],
        coords={"qubit_pair": [qp.name for qp in qubit_pairs]},
        name="fidelity",
    )

    # Assign everything to the dataset
    ds_fit = ds_fit.assign(
        density_matrix_real=xr.DataArray(
            density_matrix_xr.values.real,
            dims=density_matrix_xr.dims,
            coords=density_matrix_xr.coords,
        ),
        density_matrix_imag=xr.DataArray(
            density_matrix_xr.values.imag,
            dims=density_matrix_xr.dims,
            coords=density_matrix_xr.coords,
        ),
        raw_probs=raw_probs,
        corrected_probs=corrected_probs,
        pauli_T=pauli_T,
        fidelity=fidelity_xr,
    )
    ds_fit.attrs["bell_state"] = str(node.parameters.bell_state)

    # Build FitParameters for each qubit_pair
    fit_results = {
        qp.name: FitParameters(
            fidelity=float(fidelity_xr.sel(qubit_pair=qp.name).item()),
            success=True,
        )
        for qp in qubit_pairs
    }

    return ds_fit, fit_results



####################
# Helper functions #
####################    

def flatten(data):
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0]) + flatten(data[1:])
    else:
        return (data,)
    
def generate_pauli_basis(n_qubits):    
    pauli = np.array([0,1,2,3])
    paulis = pauli
    for i in range(n_qubits-1):
        new_paulis = []
        for ps in paulis:
            for p in pauli:
                new_paulis.append(flatten((ps, p)))
        paulis = new_paulis
    return paulis
        
def gen_inverse_hadamard(n_qubits):
    H = np.array([[1,1],[1,-1]])/2
    for _ in range(n_qubits-1):
        H = np.kron(H, H)
    return np.linalg.inv(H)

def get_pauli_data(da):

    pauli_basis = generate_pauli_basis(2)

    inverse_hadamard = gen_inverse_hadamard(2)

    # Create an xarray Dataset with dimensions and coordinates based on pauli_basis
    paulis_data = xr.Dataset(
        {
            "value": (["pauli_op"], np.zeros(len(pauli_basis))),
            "appearances": (["pauli_op"], np.zeros(len(pauli_basis), dtype=int))
        },
        coords={'pauli_op': [','.join(map(str, op)) for op in pauli_basis]}
    )

    for tomo_axis in da.coords['tomo_axis'].values:
        tomo_data = da.sel(tomo_axis = tomo_axis)
        pauli_data = inverse_hadamard @ tomo_data.data
        paulis = ["0,0", f"{tomo_axis[0]+1},0", f"0,{tomo_axis[1]+1}", f"{tomo_axis[0]+1},{tomo_axis[1]+1}"]
        for i, pauli in enumerate(paulis):
            paulis_data.value.loc[{'pauli_op': pauli}] += pauli_data[i]
            paulis_data.appearances.loc[{'pauli_op': pauli}] += 1
        
    paulis_data = xr.where(paulis_data.appearances != 0, paulis_data.value / paulis_data.appearances, paulis_data.value)
    
    return paulis_data


def get_density_matrix(paulis_data):
    # 2Q
    # Define the Pauli matrices
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # Create a vector of the Pauli matrices
    pauli_matrices = [I, X, Y, Z]

    rho = np.zeros((4,4))

    for i, pauli_i in enumerate(pauli_matrices):
        for j, pauli_j in enumerate(pauli_matrices):
            rho = rho + 0.25*paulis_data.sel(pauli_op = f"{i},{j}").values * np.kron(pauli_i, pauli_j)
    
    return rho