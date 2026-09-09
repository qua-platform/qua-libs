import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Iterable, Literal, Mapping, Optional

from calibration_utils.n_qubit_confusion_matrix import QubitGroup

from .helpers import ghz_density_matrix


def _adaptive_tick_spec(n_qubits: int, dim: int):
    """Return tick indices and labels adapted to matrix size."""
    if n_qubits <= 3:
        idx = np.arange(dim)
        labels = [format(i, f"0{n_qubits}b") for i in idx]
        return idx, labels

    max_ticks = 10
    step = max(1, int(np.ceil(dim / max_ticks)))
    idx = np.arange(0, dim, step)
    if idx[-1] != dim - 1:
        idx = np.append(idx, dim - 1)

    if n_qubits <= 4:
        labels = [format(i, f"0{n_qubits}b") for i in idx]
    else:
        labels = [f"0x{i:X}" for i in idx]
    return idx, labels


def plot_3d_component(data, ideal, n_qubits, title="", component="real"):
    """Plot a 3D bar comparison of reconstructed vs ideal matrix component."""
    dim = 2**n_qubits
    tick_idx, tick_labels = _adaptive_tick_spec(n_qubits, dim)

    fig = plt.figure(figsize=(max(6, dim), 5))
    ax = fig.add_subplot(111, projection="3d")

    xpos, ypos = np.meshgrid(np.arange(dim) + 0.5, np.arange(dim) + 0.5, indexing="ij")
    xpos, ypos = xpos.ravel(), ypos.ravel()
    zpos = np.zeros_like(xpos)

    if component == "real":
        dz = np.real(data).ravel()
        dzi = np.real(ideal).ravel()
        component_title = "Real part"
    elif component == "imag":
        dz = np.imag(data).ravel()
        dzi = np.imag(ideal).ravel()
        component_title = "Imag part"
    else:
        raise ValueError(f"Unknown component '{component}', expected 'real' or 'imag'.")

    colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    gmin = min(np.min(dz), np.min(dzi))
    gmax = max(np.max(dz), np.max(dzi))

    ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, color=cmap((np.sign(dz) + 1) / 2), alpha=1)
    ax.bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha=0.15, edgecolor="k")

    ax.set_xticks(tick_idx + 1)
    ax.set_yticks(tick_idx + 1)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels, rotation=45)

    ax.set_zlim([gmin, gmax])
    ax.set_title(title + f"\n ({component_title})")
    return fig


def plot_density_heatmap(data, n_qubits, title="", component="real", annotate_values=None):
    """Plot an annotated heatmap of one density-matrix component."""
    dim = 2**n_qubits
    tick_idx, tick_labels = _adaptive_tick_spec(n_qubits, dim)

    if component == "real":
        rho_component = np.real(data)
        component_title = "Real part"
    elif component == "imag":
        rho_component = np.imag(data)
        component_title = "Imag part"
    else:
        raise ValueError(f"Unknown component '{component}', expected 'real' or 'imag'.")

    if annotate_values is None:
        annotate_values = n_qubits <= 3

    fig, ax = plt.subplots(figsize=(max(6, dim * 0.6), max(5, dim * 0.55)))
    ax.pcolormesh(rho_component, vmin=-0.5, vmax=0.5, cmap="RdBu")

    if annotate_values:
        for i in range(dim):
            for j in range(dim):
                value = rho_component[i][j]
                color = "k" if np.abs(value) < 0.1 else "w"
                ax.text(i + 0.5, j + 0.5, f"{value:.2f}", ha="center", va="center", color=color)

    ax.set_title(title + f"\n({component_title})")
    ax.set_xlabel("Computational basis")
    ax.set_ylabel("Computational basis")
    ax.set_xticks(tick_idx)
    ax.set_yticks(tick_idx)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_yticklabels(tick_labels)
    return fig


def plot_ghz_tomography(
    rhos_by_method: Mapping[str, Mapping[str, np.ndarray]],
    qubit_groups: Iterable[QubitGroup],
    fit_results: Mapping[str, Mapping[str, object]],
    *,
    num_qubits: int,
    plot_level: Literal["full", "minimal"] = "minimal",
) -> Dict[str, plt.Figure]:
    """Plot reconstructed GHZ density matrices for selected mitigation methods."""
    ideal_dat = ghz_density_matrix(num_qubits)
    fidelity_keys = {"kron": "fidelity_kron", "nq": "fidelity_nq"}
    figures: Dict[str, plt.Figure] = {}

    for qg in qubit_groups:
        fr = fit_results.get(qg.name, {})
        candidates = ("kron", "nq") if plot_level == "full" else ("nq", "kron")

        for method_name in candidates:
            rhos = rhos_by_method.get(method_name, {})
            fidelity = fr.get(fidelity_keys[method_name])
            if qg.name not in rhos or fidelity is None:
                continue

            mitigation_label = (
                "Uncorrelated confusion matrix correction"
                if method_name == "kron"
                else "Correlated confusion matrix correction"
            )
            plot_title = f"GHZ state\n{qg.name}\n({mitigation_label})"
            rho = rhos[qg.name]
            prefix = f"{qg.name}_{method_name}"

            figures[f"{prefix}_3d_real"] = plot_3d_component(
                rho, ideal_dat, num_qubits, title=plot_title, component="real"
            )
            figures[f"{prefix}_3d_imag"] = plot_3d_component(
                rho, ideal_dat, num_qubits, title=plot_title, component="imag"
            )

            if plot_level == "full":
                figures[f"{prefix}_heatmap_real"] = plot_density_heatmap(
                    rho,
                    num_qubits,
                    title=plot_title,
                    component="real",
                    annotate_values=num_qubits <= 3,
                )
                figures[f"{prefix}_heatmap_imag"] = plot_density_heatmap(
                    rho,
                    num_qubits,
                    title=plot_title,
                    component="imag",
                    annotate_values=num_qubits <= 3,
                )
            else:
                break

    return figures
