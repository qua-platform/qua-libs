import re
from typing import List, Tuple
import plotly.express as px

import plotly
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib import axes
from matplotlib import pyplot as plt
import matplotlib
from qm import QmJob
import xarray as xr


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
            if samples_struct[i][j].dtype == float:
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


def plot_ar_attempts(ar_data: dict[str, np.typing.NDArray], **hist_kwargs):
    """
    plot the distribution of active reset attempts acquired when running `reset_active(save_qua_var=...)`

    ar_data - a dictionary of the form {qubit_name: ar_attempts}
    hist_kwargs - kwargs to pass to matplotlib.pyplot.hist

    Example:
        # in the QUA program, save the number of active reset attempts in a variable as follows:

        q.reset_active(save_qua_var=f'ar_{q.name}')

        # then in the python script, fetch the data and plot it as follows:
        from quam_libs.lib.plot_utils import plot_ar_attempts
        ar_dat = {}
        for q in machine.active_qubits:
        for q in machine.active_qubits:
            qn = q.name
            ar_dat[qn] = job.result_handles.get(f'ar_{qn}').fetch_all()['value']

        I_ar = {}
        for q in machine.active_qubits:
        for q in machine.active_qubits:
            qn = q.name
            I_ar[qn] = job.result_handles.get(f'I_ar_{qn}').fetch_all()['value']

        plot_ar_attempts(ar_dat, bins=100, log=True)
    """
    qubits = list(ar_data.keys())
    fig, axes = plt.subplots(
        1, len(qubits), sharex='None', figsize=(14, 4), squeeze=False)
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

def grid_pair_names(machine) -> List[str]:
    """"
    Runs over active qubit pairs in a QUAM object and returns a list of the grid_name attribute of each qubit
    """
    return  [ qp.qubit_target.extras['grid_name']+':'+qp.qubit_control.extras['grid_name'] for qp in machine.active_qubit_pairs]
class QubitPairGrid():
    """Creates a grid object where qubit pairs are placed on a grid. 
    The grid is builtb with references to the qubit pair names, 
    which should of the form: 'q-i_j-q-n_m' where i,j and n,m are 
    integeres describing the x and y coordinates of the qubits of 
    the pair on a qubit grid.

    Iteration of the resuting grid can be done using 'grid_iter' 
    defined in lib.qua_datasets

    :param ds: The ds containing the names of the qubit in ds.qubit

    :var fig: the created figure object
    :var all_axes: all of the created axis, used and unused
    :var axes: a list of the axes relevant for the grid
    :name_dicts: a list containing the names of the qubit, taken from ds.qubit, in the 
                convention of FacetGrid dict_names

    usage example:
    Assume we have a dataset with a data variable I, and a data coordinate q formated
    according to the naming convention. A way to plot the data on the grid would be

    '''
    from quam_libs.lib.qua_datasets import grid_iter
    grid_names = [ dict(zip(q.extras_[0:len(q.extras_):2],q.extras_[1:len(q.extras_):2]))['grid_name'] for q in machine.active_qubits]
    grid = coupler_grid(ds, grid_names)

    for ax, coupler in grid_iter(grid):
        ds.loc[coupler].I.plot(ax = ax)
    '''

    """

    def _convert_to_int(self, incoming_string):
        return tuple(map(int, incoming_string.split('_')))

    def _list_clean(self, list_input_string):
        return [self._clean_up(input_string) for input_string in list_input_string]

    def _clean_up(self, input_string):
        return re.sub("[^0-9]", "", input_string)
    

    def __init__(self, ds : xr.DataArray, qubit_pair_names : list[str], size : int = 4):


        if len(qubit_pair_names)>1:
            qubit_indices = [tuple([tuple(map(int,self._list_clean(gp.split(':')[0].split('_')))),tuple(map(int,self._list_clean(gp.split(':')[1].split('_'))))]) for gp in qubit_pair_names]
        else:
            qubit_indices = [tuple([tuple(map(int,self._list_clean(gp.split(':')[0].split('_')))),tuple(map(int,self._list_clean(gp.split(':')[1].split('_'))))]) for gp in qubit_pair_names]


        col_diffs = [pair[1][0]-pair[0][0] for pair in qubit_indices]
        row_diffs = [pair[1][1]-pair[0][1] for pair in qubit_indices]
        coupler_indices = [[2*pair[0][0], 2*pair[0][1]]
                           for pair in qubit_indices]
        for k, (col_diff, row_diff) in enumerate(zip(col_diffs, row_diffs)):
            coupler_indices[k][0] += col_diff
            coupler_indices[k][1] += row_diff
        coupler_indices = [tuple(coupler) for coupler in coupler_indices]

        grid_row_idxs = [idx[0] for idx in coupler_indices]
        grid_col_idxs = [idx[1] for idx in coupler_indices]
        min_grid_row = min(grid_row_idxs)
        min_grid_col = min(grid_col_idxs)
        shape = (max(grid_row_idxs) - min_grid_row + 1,
                 max(grid_col_idxs) - min_grid_col + 1)
            
        figure, all_axes = plt.subplots(*shape, figsize = (shape[1] * size, shape[0] * size), squeeze=False)

        if shape == (1, 1):
            # If (1, 1), subplots returns a single axis, which we convert into
            # a nested array
            axes = np.array(((all_axes,),))
        else:
            # If (1, N) or (N, 1), subplots returns a 1D array of axes, which we
            # convert into a 2D array.
            axes = all_axes.reshape(shape)

        axes = []
        qubit_names = []

        for row, axis_row in enumerate(all_axes):
            for col, ax in enumerate(axis_row):
                grid_row = max(grid_row_idxs) - row
                grid_col = col + min_grid_col

                if (grid_row, grid_col) in coupler_indices:

                    axes.append(ax)
                    qubit_names.append(
                        ds.qubitp.values[coupler_indices.index((grid_row, grid_col))])
                else:
                    ax.axis('off')
        self.fig = figure
        self.all_axes = all_axes
        self.axes = [axes]
        self.name_dicts = [[{ds.qubitp.name: value} for value in qubit_names]]

def grid_names(machine) -> List[str]:
    """"
    Runs over active qubits in a QUAM object and returns a list of the grid_name attribute of each qubit
    """
    return  [ q.extras['grid_name'] for q in machine.active_qubits]




class QubitGrid():
    """Creates a grid object where qubits are placed on a grid. 
    Accepts a dataset whose dimension 'qubit is used as the dimension on which the grid is built.
    It also accepts a parameter "grid_names" that specifies the positon of each wubit on a grid. If none
    it assumes that qubit names are of the form: 'q-i_j' where i,j are integeres describing the x and y coordinates of the grid.

    Iteration of the resuting grid can be done using 'grid_iter' defined in lib.qua_datasets

    :param ds: The ds containing the names of the qubit in ds.qubit
    :params grid_names: a list of names in the required qubit names, in case the qubits names 
                        given in a different format. Defalut is None

    :var fig: the created figure object
    :var all_axes: all of the created axis, used and unused
    :var axes: a list of the axes relevant for the grid
    :name_dicts: a list containing the names of the qubit, taken from ds.qubit, in the 
                convention of FacetGrid dict_names

    usage example:
    Assume we have a dataset with a data variable I, and a data coordinate q formated
    according to the naming convention. A way to plot the data on the grid would be

    '''
    from quam_libs.lib.qua_datasets import grid_iter, QubitGrid
    grid = QubitGrid(ds)

    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].I.plot(ax = ax)
    '''

    If the names of the qubits are not of the acceptable form it is possible to use:

    '''
    from quam_libs.lib.qua_datasets import grid_iter, QubitGrid, grid_names
    grid = QubitGrid(ds, grid_names(machine))

    for ax, qubit in grid_iter(grid):
        ds.loc[qubit].I.plot(ax = ax)

    '''

    """

    def _list_clean(self, list_input_string):
        return [self._clean_up(input_string) for input_string in list_input_string]

    def _clean_up(self, input_string):
        return re.sub("[^0-9]", "", input_string)

    def __init__(self, ds: xr.DataArray, grid_names: list = None, size : int = 3):
        if grid_names:
            if type(grid_names) == str:
                grid_names = [grid_names]
            grid_indices = [tuple(map(int, self._list_clean(
                grid_name.split('_')))) for grid_name in grid_names]
        else:
            grid_indices = [tuple(map(int, self._list_clean(
                ds.qubit.values[q_index].split('_')))) for q_index in range(ds.qubit.size)]

        if len(grid_indices) > 1:
            grid_name_mapping = dict(zip(grid_indices, ds.qubit.values))
        else:
            try:
                grid_name_mapping = dict(
                    zip(grid_indices, [str(ds.qubit.values[0])]))
            except:
                grid_name_mapping = dict(zip(grid_indices, [str(ds.qubit.values)]))
        grid_row_idxs = [idx[1] for idx in grid_indices]
        grid_col_idxs = [idx[0] for idx in grid_indices]
        min_grid_row = min(grid_row_idxs)
        min_grid_col = min(grid_col_idxs)
        shape = (max(grid_row_idxs) - min_grid_row + 1,
                 max(grid_col_idxs) - min_grid_col + 1)
            
        figure, all_axes = plt.subplots(*shape, figsize = (shape[1] * size, shape[0] * size), squeeze=False)

        axes = []

        qubit_names = []

        for row, axis_row in enumerate(all_axes):
            for col, ax in enumerate(axis_row):
                grid_row = max(grid_row_idxs) - row
                grid_col = col + min_grid_col
                if (grid_col, grid_row) in grid_indices:
                    axes.append(ax)
                    name = grid_name_mapping.get((grid_col, grid_row))
                    if name is not None:
                        qubit_names.append(
                            grid_name_mapping[(grid_col, grid_row)])
                else:
                    ax.axis('off')

        self.fig = figure
        self.all_axes = all_axes
        self.axes = [axes]
        self.name_dicts = [[{ds.qubit.name: value} for value in qubit_names]]


def grid_iter(grid: xr.plot.FacetGrid) -> Tuple[matplotlib.axes.Axes, dict]:
    """Create a generator to iterate over a facet grid.
    For each iteration, return a tuple of (axis object, name dict of this axis).

    This is useful for adding annotations and additional data to facet grid figures.

    :param grid: The grid to iterate over
    :type grid: xr.plot.FacetGrid
    :yield: a tuple with the axis and name of the facet
    :rtype: _type_
    """
    for axr, ndr in zip(grid.axes, grid.name_dicts):
        for ax, nd in zip(axr, ndr):
            yield ax, nd
