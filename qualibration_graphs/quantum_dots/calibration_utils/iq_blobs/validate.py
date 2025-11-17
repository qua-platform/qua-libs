"""
Validation utilities for simulating realistic quantum dot readout data.

This module provides functions to simulate IQ blob data for quantum dot pairs
using the Barthel model, allowing for testing of analysis and plotting without
hardware execution.
"""

import numpy as np
import xarray as xr
import jax.numpy as jnp
from typing import List, Optional

from .readout_barthel.simulate import SimulationParamsIQ, simulate_readout_iq


def simulate_quantum_dot_readout_data(
    qubit_pairs: List,
    n_runs: int,
    p_triplet: float = 0.5,
    mu_S: tuple = (0.0, 0.0),
    mu_T: tuple = (1.0e-2, 0.25e-2),
    sigma_I: float = 0.12e-2,
    sigma_Q: float = 0.10e-2,
    rho: float = 0.0,
    tau_M: float = 1.0,
    T1: float = 2.0,
    add_noise_variation: bool = True,
    seed: Optional[int] = None,
) -> xr.Dataset:
    """
    Simulate realistic quantum dot readout data for multiple quantum dot pairs.

    This function generates simulated IQ data for both singlet (ground) and triplet
    (excited) states using the Barthel model. The data format matches what would be
    returned by the execute_qua_program function.

    Parameters
    ----------
    qubit_pairs : List
        List of quantum dot pair objects to simulate data for.
    n_runs : int
        Number of measurement repetitions (shots) to simulate.
    p_triplet : float, optional
        Probability of triplet state in mixed ensemble, by default 0.5.
    mu_S : tuple, optional
        Mean (I, Q) position of singlet state, by default (0.0, 0.0).
    mu_T : tuple, optional
        Mean (I, Q) position of triplet state, by default (1.0e-2, 0.25e-2).
    sigma_I : float, optional
        Standard deviation of I quadrature noise, by default 0.12e-2.
    sigma_Q : float, optional
        Standard deviation of Q quadrature noise, by default 0.10e-2.
    rho : float, optional
        Correlation between I and Q noise, by default 0.0.
    tau_M : float, optional
        Measurement time, by default 1.0.
    T1 : float, optional
        T1 relaxation time, by default 2.0.
    add_noise_variation : bool, optional
        If True, adds slight variations in noise parameters across quantum dot pairs
        to make the simulation more realistic, by default True.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    xr.Dataset
        Dataset containing simulated IQ data with variables:
        - Ig1, Qg1, Ie1, Qe1: Ground and excited state IQ data for qubit_pair 1
        - Ig2, Qg2, Ie2, Qe2: Ground and excited state IQ data for qubit_pair 2
        - etc.
        Dimensions: (qubit_pair, n_runs)

    Examples
    --------
    >>> # Assuming you have qubit_pairs defined
    >>> ds = simulate_quantum_dot_readout_data(
    ...     qubit_pairs=qubit_pairs,
    ...     n_runs=1000,
    ...     p_triplet=0.5
    ... )
    >>> # Now you can use this dataset for analysis
    >>> from .analysis import fit_raw_data, process_raw_dataset
    >>> ds_processed = process_raw_dataset(ds, node)
    >>> ds_fit, fit_results = fit_raw_data(ds_processed, node)
    """
    if seed is not None:
        np.random.seed(seed)

    num_qubit_pairs = len(qubit_pairs)
    qubit_pair_names = [q.name for q in qubit_pairs]

    # Storage for simulated data
    Ig_data = []
    Qg_data = []
    Ie_data = []
    Qe_data = []

    for i, qdp in enumerate(qubit_pairs):
        # Add realistic variations across quantum dot pairs if requested
        if add_noise_variation:
            # Add small random variations to noise parameters (�20%)
            sigma_I_var = sigma_I * (1 + 0.2 * (np.random.rand() - 0.5))
            sigma_Q_var = sigma_Q * (1 + 0.2 * (np.random.rand() - 0.5))

            # Add small variations to blob positions (�10%)
            mu_S_var = (
                mu_S[0] + 0.1 * mu_S[0] * (np.random.rand() - 0.5) if mu_S[0] != 0 else 0.0,
                mu_S[1] + 0.1 * mu_S[1] * (np.random.rand() - 0.5) if mu_S[1] != 0 else 0.0,
            )
            mu_T_var = (
                mu_T[0] + 0.1 * mu_T[0] * (np.random.rand() - 0.5),
                mu_T[1] + 0.1 * mu_T[1] * (np.random.rand() - 0.5),
            )
        else:
            sigma_I_var = sigma_I
            sigma_Q_var = sigma_Q
            mu_S_var = mu_S
            mu_T_var = mu_T

        # --- Simulate GROUND state (mostly singlet) ---
        # For ground state preparation, we expect mostly singlet with small triplet contamination
        params_ground = SimulationParamsIQ(
            n_samples=n_runs,
            p_triplet=0.05,  # Small triplet contamination in ground state
            mu_S=mu_S_var,
            mu_T=mu_T_var,
            sigma_I=sigma_I_var,
            sigma_Q=sigma_Q_var,
            rho=rho,
            tau_M=tau_M,
            T1=T1,
        )
        X_ground, _ = simulate_readout_iq(params_ground, return_labels=True)
        X_ground = np.array(X_ground)

        Ig_data.append(X_ground[:, 0])
        Qg_data.append(X_ground[:, 1])

        # --- Simulate EXCITED state (mostly triplet after x180 pulse) ---
        # For excited state, we expect mostly triplet after the x180 pulse
        params_excited = SimulationParamsIQ(
            n_samples=n_runs,
            p_triplet=0.95,  # High triplet probability after x180 pulse
            mu_S=mu_S_var,
            mu_T=mu_T_var,
            sigma_I=sigma_I_var,
            sigma_Q=sigma_Q_var,
            rho=rho,
            tau_M=tau_M,
            T1=T1,
        )
        X_excited, _ = simulate_readout_iq(params_excited, return_labels=True)
        X_excited = np.array(X_excited)

        Ie_data.append(X_excited[:, 0])
        Qe_data.append(X_excited[:, 1])

    # Create xarray Dataset matching the expected format from execute_qua_program
    # The format should match what comes out of stream processing in QUA
    # Reshape to match expected format: add qubit_pair dimension
    Ig_combined = np.array(Ig_data)
    Qg_combined = np.array(Qg_data)
    Ie_combined = np.array(Ie_data)
    Qe_combined = np.array(Qe_data)

    ds_final = xr.Dataset(
        {
            "Ig": (["qubit_pair", "n_runs"], Ig_combined),
            "Qg": (["qubit_pair", "n_runs"], Qg_combined),
            "Ie": (["qubit_pair", "n_runs"], Ie_combined),
            "Qe": (["qubit_pair", "n_runs"], Qe_combined),
            "n": (["n_runs"], np.arange(n_runs)),
        },
        coords={
            "qubit_pair": qubit_pair_names,
            "n_runs": np.arange(n_runs),
        },
    )

    return ds_final


def simulate_quantum_dot_readout_from_node(node, add_noise_variation: bool = True, seed: Optional[int] = None) -> xr.Dataset:
    """
    Convenience function to simulate quantum dot readout data directly from a QualibrationNode.

    This function extracts the necessary parameters from the node and calls
    simulate_quantum_dot_readout_data with appropriate defaults.

    Parameters
    ----------
    node : QualibrationNode
        The calibration node containing parameters and qubit_pairs.
    add_noise_variation : bool, optional
        If True, adds slight variations in noise parameters across quantum dot pairs,
        by default True.
    seed : int, optional
        Random seed for reproducibility, by default None.

    Returns
    -------
    xr.Dataset
        Dataset containing simulated IQ data ready for analysis.

    Examples
    --------
    >>> # In your simulate_qua_program function:
    >>> from calibration_utils.iq_blobs.validate import simulate_quantum_dot_readout_from_node
    >>> ds_simulated = simulate_quantum_dot_readout_from_node(node)
    >>> node.results["ds_raw"] = ds_simulated
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    n_runs = node.parameters.num_shots

    # You can add more parameters to node.parameters to control simulation if needed
    # For now, using reasonable defaults
    ds = simulate_quantum_dot_readout_data(
        qubit_pairs=qubit_pairs,
        n_runs=n_runs,
        p_triplet=0.5,  # Mixed ensemble for calibration data
        mu_S=(0.0, 0.0),  # Singlet at origin
        mu_T=(1.0e-2, 0.25e-2),  # Triplet offset in IQ plane
        sigma_I=0.12e-2,
        sigma_Q=0.10e-2,
        rho=0.0,
        tau_M=1.0,
        T1=2.0,
        add_noise_variation=add_noise_variation,
        seed=seed,
    )

    return ds