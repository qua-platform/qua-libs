from typing import List
import plotly.express as px

import plotly
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import axes
from matplotlib import pyplot as plt
from qm.QmJob import QmJob


def plot_channel(ax, data: np.ndarray, title: str):
    if data.dtype == np.complex:
        ax.plot(data.real)
        ax.plot(data.imag)
    else:
        ax.plot(data)
    ax.set_title(title)
    ax.set_ylim((-0.6, 0.6))


def get_simulated_samples_by_element(element_name: str, job: QmJob, config: dict):
    element = config["elements"][element_name]
    sample_struct = job.get_simulated_samples()
    if "mixInputs" in element:
        port_i = element["mixInputs"]["I"]
        port_q = element["mixInputs"]["Q"]
        samples = (
            sample_struct.__dict__[port_i[0]].analog[str(port_i[1])]
            + 1j * sample_struct.__dict__[port_q[0]].analog[str(port_q[1])]
        )
    else:
        port = element["singleInput"]["port"]
        samples = sample_struct.__dict__[port[0]].analog[str(port[1])]
    return samples


def plot_simulator_output(
    plot_axes: List[List[str]], job: QmJob, config: dict, duration_nsec: int
):
    """
    generate a plot of simulator output by elements

    :param plot_axes: a list of lists of elements. Will open
    multiple axes, one for each list.
    :param job: The simulated QmJob to plot
    :param config: The config file used to create the job
    :param duration_nsec: the duration to plot in nsec
    """
    time_vec = np.linspace(0, duration_nsec - 1, duration_nsec)
    samples_struct = []
    for plot_axis in plot_axes:
        samples_struct.append(
            [get_simulated_samples_by_element(
                pa, job, config) for pa in plot_axis]
        )

    fig = go.Figure().set_subplots(rows=len(plot_axes), cols=1, shared_xaxes=True)

    for i, plot_axis in enumerate(plot_axes):
        for j, plotitem in enumerate(plot_axis):
            if samples_struct[i][j].dtype == np.float:
                fig.add_trace(
                    go.Scatter(
                        x=time_vec, y=samples_struct[i][j], name=plotitem),
                    row=i + 1,
                    col=1,
                )
                print(samples_struct[i][j])
            else:
                fig.add_trace(
                    go.Scatter(
                        x=time_vec, y=samples_struct[i][j].real, name=plotitem + " I"
                    ),
                    row=i + 1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=time_vec, y=samples_struct[i][j].imag, name=plotitem + " Q"
                    ),
                    row=i + 1,
                    col=1,
                )
    fig.update_xaxes(title="time [nsec]")
    return fig


def plot_ar_attempts(ar_data, **hist_kwargs):
    qubits = list(ar_data.keys())
    fig, axes = plt.subplots(1, len(qubits), sharex='all', figsize=(14, 4))
    for i, q in enumerate(qubits):
        ax = axes[i]
        ax.hist(ar_data[q], **hist_kwargs)
        ax.set_title(f'q = {q}')
        ax.set_ylabel('prob.')
        ax.set_xlabel('no. of attempts')
    fig.tight_layout()


def plot_spectrum(signal: np.ndarray, t_s_usec: float, num_zero_pad: int = 0) -> tuple[np.ndarray, np.ndarray, plotly.graph_objs.Figure]:
    """
    plot the spectrum of a signal

    signal - 1D array to plot
    t_s_usec - sampling time interval in usec
    num_zero_pad - how much to zero pad the signal

    returns: the spectrum (abs^2), the frequency vector and the plotly figure object
    """
    signal_ac = signal - signal.mean()
    signal_pad = np.hstack((signal_ac, np.zeros(num_zero_pad)))
    n = len(signal_pad)
    signal_fft = np.abs(np.fft.fft(signal_pad))**2

    freq_ax = np.fft.fftfreq(len(signal_pad), d=t_s_usec)
    f_s = 0.5/t_s_usec

    fig = px.line(x=freq_ax[1:], y=signal_fft[1:],
                  labels={'x': 'frequency [MHz]', 'y': 'power'}
                  )
    fig.update_layout(xaxis_range=(0, f_s))
    return signal_fft, freq_ax, fig
