from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery import baking
from scipy import signal, optimize
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
import warnings

warnings.filterwarnings("ignore")

pi_half_amp = 0.2
pi_amp = 0.4
pi_len = 7
tau = 18
N = 10

wf = []
# wf += [0.0] * 15
# wf += [pi_half_amp] * pi_len
for i in range(N):
    wf += [0.0] * tau + [pi_amp] * pi_len

wf += [0.0] * tau + [pi_half_amp] * pi_len
wf += [0.0] * 15

time = np.arange(0, len(wf), 1)
plt.figure()
plt.plot(time, wf)
plt.grid("on")

baked_segments = []
for k in range(32):
    wf = [0.0] * k + [pi_amp] * pi_len + [0.0] * (4 - (pi_len + k) % 4)
    if len(wf) < 16:
        wf += [0.0] * (16 - len(wf))
    baked_segments.append(wf)

gap = 0
btot = []
for i in range(N):
    # plt.plot(baked_segments[tau-gap1])
    btot += baked_segments[tau - gap]
    gap = 4 - (tau - gap + pi_len) % 4


plt.plot(btot, "k-")
gap0 = 4 - (tau + pi_len) % 4
b1 = [0.0] * tau + [pi_amp] * pi_len + [0.0] * gap0
gap1 = 4 - (tau - gap0 + pi_len) % 4

b2 = [0.0] * (tau - gap0) + [pi_amp] * pi_len + [0.0] * gap1
gap2 = 4 - (tau - gap1 + pi_len) % 4
b3 = [0.0] * (tau - gap1) + [pi_amp] * pi_len + [0.0] * gap2
plt.plot(b1 + b2 + b3)
