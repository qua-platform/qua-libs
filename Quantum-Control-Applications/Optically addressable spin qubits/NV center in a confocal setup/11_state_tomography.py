"""
       STATE TOMOGRAPHY
The goal of this program is to measure the projection of the Bloch vector of the qubit along the three axes of
the Bloch sphere in order to reconstruct the full qubit state tomography.
The qubit state preparation is left to user to define.

Prerequisites:
    - Ensure calibration of the different delays in the system (calibrate_delays).
    - Having updated the different delays in the configuration.
    - Having updated the NV frequency, labeled as "NV_IF_freq", in the configuration.
    - Having set the pi pulse amplitude and duration in the configuration
"""
from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
from qualang_tools.loops import from_array


######################################
# Set-up a Bloch sphere for plotting #  (can be removed if not used)
######################################
def bra_tex(s):
    return rf"$\left| {s} \right\rangle$"


class BlochSpherePlot:
    North = np.array((0, 0, 1))
    South = np.array((0, 0, -1))
    East = np.array((1, 0, 0))
    West = np.array((0, 1, 0))

    def __init__(self, elev=20, azim=15, sphere_style=None, circles_style=None, axes_style=None, *args, **kwargs):
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"}, *args, **kwargs)
        self.ax.view_init(elev=elev, azim=azim)

        self.sphere_style = {
            "color": "#ccf3ff",
            "alpha": 0.1,
        }
        if sphere_style:
            self.sphere_style.update(sphere_style)

        self.circles_style = {
            "color": "#333",
            "alpha": 0.2,
            "lw": 1.0,
        }
        if circles_style:
            self.circles_style.update(circles_style)

        self.axes_style = {
            "color": "#333",
            "alpha": 0.2,
            "lw": 1.0,
        }
        if axes_style:
            self.axes_style.update(axes_style)

        self.add_sphere()
        self.add_circles()
        self.add_axes()

        self._prepare_axes()

    def plot_vector(
        self,
        v,
        label=None,
        **kwargs,
    ):
        self.ax.quiver(
            0,
            0,
            0,
            v[0],
            v[1],
            v[2],
            normalize=True,
            arrow_length_ratio=0.08,
            **kwargs,
        )
        if label:
            vn = 1.1 * np.array(v) / np.linalg.norm(v)
            self.ax.text(*vn, label, fontsize="large")

    def label(self, position, label, **kwargs):
        self.ax.text(*position, label, fontsize="large", **kwargs)

    def label_bra(self, position, label, **kwargs):
        self.label(position, bra_tex(label), **kwargs)

    def add_circles(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        c = np.cos(theta)
        s = np.sin(theta)
        z = np.zeros_like(theta)

        self.ax.plot(c, s, z, **self.circles_style)
        self.ax.plot(z, c, s, **self.circles_style)
        self.ax.plot(s, z, c, **self.circles_style)

    def add_sphere(self):
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        self.ax.plot_surface(x, y, z, **self.sphere_style)

    def add_axes(self):
        u = np.linspace(-1, 1, 2)
        z = np.zeros_like(u)
        self.ax.plot(u, z, z, **self.axes_style)
        self.ax.plot(z, z, u, **self.axes_style)
        self.ax.plot(z, u, z, **self.axes_style)

    def _prepare_axes(self):
        self.ax.set(
            aspect="equal",
        )

        self.ax.tick_params(
            labelbottom=False,
            labelleft=False,
        )
        self.ax.set_axis_off()
        self.ax.grid(False)
        self.ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        self.ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))


bloch_sphere = BlochSpherePlot(
    sphere_style={"alpha": 0.03, "color": "#333", "shade": False},
    axes_style={"alpha": 0.8},
    elev=20,
    azim=-30,
    dpi=100,
    figsize=(8, 6),
)
bloch_sphere.label_bra(bloch_sphere.North * 1.1, "e")
bloch_sphere.label_bra(bloch_sphere.South * 1.1, "g")
bloch_sphere.label_bra(bloch_sphere.East * 1.1, "X")
bloch_sphere.label_bra(bloch_sphere.West * 1.1, "Y")
# bloch_sphere.plot_vector((1, 1, 0), 'Test', color='r')
# bloch_sphere.plot_vector((1, 0, 1), bra_tex('k'), color='g')

###################
# The QUA program #
###################

n_avg = 1_000_000  # Number of averaging iterations

with program() as state_tomography:
    times = declare(int, size=100)  # QUA vector for storing the time-tags
    counts = declare(int)  # variable for number of counts
    counts_st = declare_stream()  # stream for counts
    counts_dark_st = declare_stream()  # stream for counts
    f = declare(int)  # frequencies
    n = declare(int)  # number of iterations
    n_st = declare_stream()  # stream for number of iterations
    c = declare(int)  # QUA variable for switching between projections

    with for_(n, 0, n < n_avg, n + 1):
        with for_(c, 0, c <= 2, c + 1):
            # Add here whatever state you want to characterize
            with switch_(c):
                with case_(0):  # projection along X
                    play("-y90", "NV")
                with case_(1):  # projection along Y
                    play("x90", "NV")
                with case_(2):  # projection along Z
                    wait(pi_len_NV * u.ns, "NV")
            align()  # Play the laser pulse after the mw sequence
            # Measure and detect the photons on SPCM1
            play("laser_ON", "AOM1")
            measure("readout", "SPCM1", None, time_tagging.analog(times, meas_len_1, counts))
            save(counts, counts_st)  # save counts
            wait(wait_between_runs * u.ns, "AOM1")
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        counts_st.buffer(3).average().save("counts")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, state_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(state_tomography)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["counts", "iteration"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        state, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot the Bloch vector on the Bloch sphere
        bloch_sphere.plot_vector((state[0], state[1], state[2]), "", color="r")
        plt.pause(0.1)

    # Derive the density matrix
    I = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    # Zero order approximation
    rho = 0.5 * (I + state[0] * sigma_x + state[1] * sigma_y + state[2] * sigma_z)
    print(f"The density matrix is:\n{rho}")
