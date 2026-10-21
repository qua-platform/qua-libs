"""Plotting helpers for GHZ Z-basis population measurement."""

from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt

from calibration_utils.n_qubit_confusion_matrix import QubitGroup


def _figure_width(num_qubits: int) -> int:
    if num_qubits == 3:
        return 6
    if num_qubits == 4:
        return 10
    return 12


def plot_z_basis_populations(
    corrected_results: Dict[str, Iterable[float]],
    qubit_groups: Iterable[QubitGroup],
    fidelities: Dict[str, float],
    *,
    num_qubits: int,
    title_suffix: str,
    bar_color: str = "skyblue",
    edge_color: str = "navy",
    annotation_facecolor: str = "wheat",
    fidelity_differences: Optional[Dict[str, float]] = None,
) -> plt.Figure:
    """Plot mitigated Z-basis population distributions for each qubit group."""
    qubit_groups = list(qubit_groups)
    num_groups = len(qubit_groups)
    fig_width = _figure_width(num_qubits)
    states = list(range(2**num_qubits))
    state_labels = [format(state, f"0{num_qubits}b") for state in states]
    all_0_label = "0" * num_qubits
    all_1_label = "1" * num_qubits

    if num_groups == 1:
        fig, axs = plt.subplots(1, figsize=(fig_width, 3))
        axs = [axs]
    else:
        fig, axs = plt.subplots(num_groups, 1, figsize=(fig_width, 3 * num_groups))

    for i, qg in enumerate(qubit_groups):
        ax = axs[i]
        values = corrected_results[qg.name]
        ax.bar(state_labels, values, color=bar_color, edgecolor=edge_color)
        ax.set_ylim(0, 1)
        for j, value in enumerate(values):
            if value > 0.01:
                rotation = 90 if num_qubits >= 4 else 0
                ax.text(
                    j,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    rotation=rotation,
                    fontsize=8 if num_qubits >= 4 else 10,
                )
        ax.set_ylabel("Probability")
        if i == num_groups - 1:
            ax.set_xlabel("State")
        if num_qubits >= 4:
            ax.tick_params(axis="x", rotation=90)
        ax.set_title(f"GHZ state\n{qg.name}\n({title_suffix})")
        fidelity_text = f"Z-basis population fidelity ({all_0_label}+{all_1_label}): {fidelities[qg.name]:.4f}"
        if fidelity_differences is not None and qg.name in fidelity_differences:
            diff = fidelity_differences[qg.name]
            diff_sign = "+" if diff > 0 else ""
            fidelity_text += f"\nΔ (correlated − uncorrelated): {diff_sign}{diff:.4f}"
        ax.text(
            0.02,
            0.98,
            fidelity_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor=annotation_facecolor, alpha=0.5),
        )

    fig.tight_layout(pad=2.0)
    if fidelity_differences is not None:
        fig.subplots_adjust(top=0.82, hspace=0.4)
    else:
        fig.subplots_adjust(hspace=0.4)
    return fig


def plot_z_basis_populations_nq(
    corrected_results_nq: Dict[str, Iterable[float]],
    qubit_groups: Iterable[QubitGroup],
    fidelities_nq: Dict[str, float],
    fidelity_differences: Dict[str, float],
    *,
    num_qubits: int,
) -> Optional[plt.Figure]:
    """Plot NQ-mitigated Z-basis populations when NQ matrices are available."""
    groups_with_nq = [qg for qg in qubit_groups if qg.name in corrected_results_nq]
    if not groups_with_nq:
        return None

    return plot_z_basis_populations(
        corrected_results_nq,
        groups_with_nq,
        fidelities_nq,
        num_qubits=num_qubits,
        title_suffix="Correlated confusion matrix correction",
        bar_color="moccasin",
        edge_color="orange",
        annotation_facecolor="moccasin",
        fidelity_differences={qg.name: fidelity_differences[qg.name] for qg in groups_with_nq},
    )


def plot_ghz_z_basis(
    corrected_kron: Dict[str, Iterable[float]],
    qubit_groups: Iterable[QubitGroup],
    fidelities: Dict[str, float],
    *,
    num_qubits: int,
    corrected_nq: Optional[Dict[str, Iterable[float]]] = None,
    fidelities_nq: Optional[Dict[str, float]] = None,
    fidelity_differences: Optional[Dict[str, float]] = None,
) -> Dict[str, plt.Figure]:
    """Plot Kron- and NQ-mitigated Z-basis population distributions."""
    figures = {
        "figure": plot_z_basis_populations(
            corrected_kron,
            qubit_groups,
            fidelities,
            num_qubits=num_qubits,
            title_suffix="Uncorrelated confusion matrix correction",
        )
    }

    if corrected_nq is not None and fidelities_nq is not None and fidelity_differences is not None:
        fig_nq = plot_z_basis_populations_nq(
            corrected_nq,
            qubit_groups,
            fidelities_nq,
            fidelity_differences,
            num_qubits=num_qubits,
        )
        if fig_nq is not None:
            figures["figure_nq"] = fig_nq

    return figures
