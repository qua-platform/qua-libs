from typing import List, Any
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


# --- main plotting wrapper (keeps your signature & outer structure) ----------
def plot_raw_data_with_fit(ds: xr.Dataset, qubit_pairs: List[Any], fits: xr.Dataset):
    """
    Plot reconstructed density matrix vs. ideal Bell state for each qubit pair.
    Expects `fits` to contain:
      - density_matrix: (qubit_pair, row, col) complex[4x4] per pair
      - fidelity: (qubit_pair,) scalar
      - bell_state: scalar or per-qubit_pair string indicating target Bell state
    """
    figs = []
    for qp in qubit_pairs:
        qc = qp.qubit_control
        qt = qp.qubit_target
        qp_name = qp.name

        # pull reconstructed rho (4x4) for this pair
        rho_real = fits["density_matrix_real"].sel(qubit_pair=qp_name).to_numpy()
        rho_imag = fits["density_matrix_imag"].sel(qubit_pair=qp_name).to_numpy()
        # pull fidelity if available
        F = float(fits["fidelity"].sel(qubit_pair=qp_name)) if "fidelity" in fits else np.nan
        # target bell state (accept scalar or per-pair)
        bell = str(fits.bell_state)
        if bell == "00-11":
            rho_ideal = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
        else:
            raise ValueError(f"Bell state tomography for {bell} is not supported!")

        title = f"Qc: {qc.name}, Qt: {qt.name}  |  target: {bell}  |  F={F:.3f}"
        fig = plot_3d_hist_with_frame(rho_real + 1j * rho_imag, rho_ideal, title=title)
        figs.append(fig)

    return figs

# --- 3D bar plot comparing data vs. ideal (real & imaginary parts) -----------
def plot_3d_hist_with_frame(data: np.ndarray, ideal: np.ndarray, title: str = ""):
    """
    Draw side-by-side 3D bar plots of real/imag parts of a 4x4 density matrix.
    - Opaque bars: experimental data
    - Translucent framed bars: ideal target
    """
    # sanity: ensure 4x4 complex arrays
    data = np.asarray(data, dtype=complex).reshape(4, 4)
    ideal = np.asarray(ideal, dtype=complex).reshape(4, 4)

    fig, axs = plt.subplots(1, 2, figsize=(10, 6), subplot_kw={"projection": "3d"})
    # grid positions
    xpos, ypos = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos, dtype=float)

    # simple 2-color map for sign; map {-1,0,1} -> {0,0.5,1}
    colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]  # dark→light blues
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # symmetric z-limits across real/imag, data/ideal
    vals = np.concatenate([
        np.real(data).ravel(), np.imag(data).ravel(),
        np.real(ideal).ravel(), np.imag(ideal).ravel()
    ])
    g = np.max(np.abs(vals))
    zlim = (-g if g > 0 else -1.0, g if g > 0 else 1.0)

    for i, part in enumerate(("real", "imag")):
        dz = (np.real(data) if part == "real" else np.imag(data)).ravel()
        dzi = (np.real(ideal) if part == "real" else np.imag(ideal)).ravel()

        # bars for experimental data
        face = cmap((np.sign(dz) + 1) / 2.0)
        axs[i].bar3d(xpos, ypos, zpos, dx=0.6, dy=0.6, dz=dz, alpha=1.0, color=face)
        # translucent wireframe-like bars for ideal
        axs[i].bar3d(xpos, ypos, zpos, dx=0.6, dy=0.6, dz=dzi, alpha=0.15, edgecolor="k", color=(1,1,1,0))

        axs[i].set_title(part)
        axs[i].set_zlim(zlim)
        axs[i].set_xticks(np.arange(1, 5))
        axs[i].set_yticks(np.arange(1, 5))
        labels = ["00", "01", "10", "11"]
        axs[i].set_xticklabels(labels, rotation=45)
        axs[i].set_yticklabels(labels, rotation=45)
        axs[i].set_xlabel("row (|··⟩)")
        axs[i].set_ylabel("col (⟨··|)")
        axs[i].set_zlabel(part)

    fig.suptitle(title)
    # fig.tight_layout()
    return fig
