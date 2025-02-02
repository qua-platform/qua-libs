import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.graph_objs import Figure
from plotly.subplots import make_subplots


def plot_fidelity_3d(ds: xr.Dataset) -> Figure:
    """ Browser-based 3D volume plotting with appropriate axis labels. """
    fig = make_subplots(
        rows=1,
        cols=len(ds.qubit),
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[f"Qubit {qubit.item()}" for qubit in ds.qubit]
    )

    for i, qubit in enumerate(ds.qubit):
        da = ds.fidelity.sel(qubit=qubit)

        # Convert frequency to MHz
        durations = ds.duration.data
        freqs_mhz = ds.freq.data / 1e6  # Convert Hz to MHz
        amps = ds.amp.data

        # Create a meshgrid ensuring correct shape
        D, F, A = np.meshgrid(durations, freqs_mhz, amps, indexing="xy")

        fig = fig.add_trace(go.Volume(
            x=F.flatten(),
            y=D.flatten(),
            z=A.flatten(),
            value=da.data.flatten(),
            isomin=float(da.min()),
            isomax=float(da.max()),
            opacity=0.1,
            opacityscale="max",
            surface_count=50,
            colorscale='picnic'
        ), row=1, col=i + 1)

        # Find the index of the maximum fidelity value
        fi, ai, di = np.unravel_index(np.argmax(da.data), da.shape)
        fig.add_trace(go.Scatter3d(
            x=[float(ds.freq[fi]) / 1e6],  # Convert to MHz
            z=[float(ds.amp[ai])],
            y=[float(ds.duration[di])],
            mode="markers+text",
            marker=dict(size=6, color="white", symbol="x"),
            text=[f"Max: {da.max():.3f}"],  # Display the value as text
            textposition="top center"
        ), row=1, col=i + 1)

        # Update layout for better axis labeling
        fig.update_layout(
            scene=dict(
                xaxis_title="Frequency (MHz)",
                yaxis_title="Duration (ns)",
                zaxis_title="Amplitude Factor (arb)"
            ),
            scene2=dict(  # Apply the same labels to the second subplot
                xaxis_title="Frequency (MHz)",
                yaxis_title="Duration (ns)",
                zaxis_title="Amplitude Factor (arb)"
            )
        )

    return fig


def plot_fidelity_2d(ds: xr.Dataset):
    figs = []
    for qubit in ds.qubit:
        fig, ax = plt.subplots(1, len(ds.duration), figsize=(15, 5), sharex=True, sharey=True)

        # Choose a good colormap for normalized data
        cmap = "RdYlGn"

        # Plot each duration slice and store the returned image for colorbar reference
        im_list = []
        for i, duration in enumerate(ds.duration):
            im = ds.fidelity.sel(qubit=qubit).sel(duration=duration).plot(
                ax=ax[i], add_colorbar=False, cmap=cmap, x="freq_mhz", y="amp",
                vmin=ds.fidelity.sel(qubit=qubit).min(),
                vmax=ds.fidelity.sel(qubit=qubit).max()
            )
            ax[i].set_title(f"Duration: {int(duration.values)}ns")  # Ensure titles for clarity
            ax[i].set_xlabel(None)
            ax[i].set_ylabel(None)
            im_list.append(im)

        # Figure titles and labels
        fig.suptitle(f"Qubit {qubit.item()}", fontsize=14)
        fig.supxlabel("Frequency (MHz)")
        fig.supylabel("Amplitude Factor (arb)")

        # Create a vertical colorbar on the right
        cax = fig.add_axes([0.88, 0.2, 0.02, 0.6])  # Shift colorbar left slightly
        cbar = fig.colorbar(im_list[0], cax=cax, orientation="vertical")
        cbar.set_label("Fidelity (%)")

        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Makes space for the colorbar

        figs.append(fig)

    return figs
