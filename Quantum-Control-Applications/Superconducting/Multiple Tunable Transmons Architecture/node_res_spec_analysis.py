# ==================== DEFINE NODE ====================
import time

import nodeio

nodeio.context(
    name="ResSpecAnalysisNode",
    description="finds resonator spectroscopy"
)

inputs = nodeio.Inputs()
inputs.stream(
    'IQ',
    units='list',
    description='measurement data'
)

outputs = nodeio.Outputs()
outputs.define(
    'state',
    units='JSON',
    description='updated state'
)

nodeio.register()

# ==================== DRY RUN DATA ====================

import numpy as np

# set inputs data for dry-run of the node
num_qubits = 4

f_min = [-70e6, -110e6, -150e6, -210e6]
f_max = [-40e6, -80e6, -130e6, -190e6]
df = 0.05e6

freqs = [np.arange(f_min[i], f_max[i] + 0.1, df) for i in range(num_qubits)]
I = np.arange(f_min[0], f_max[0] + 0.1, df)
Q = np.arange(f_min[0], f_max[0] + 0.1, df)

freqs = [freqs[i].tolist() for i in range(num_qubits)]

data = [I.tolist(), Q.tolist(), freqs]
inputs.set(IQ=data)
# =============== RUN NODE STATE MACHINE ===============

import matplotlib.pyplot as plt
from qualang_tools.units import unit
from scipy import signal
from quam import QuAM

u = unit()

while nodeio.status.active:

    outputs.set(state="quam_bootstrap_state.json")

    IQ = inputs.get('IQ')

    print('Doing resonator spec analysis...')

    I = np.array(IQ[0])
    Q = np.array(IQ[1])
    freqs_dem = np.array(IQ[2][0])

    fig = plt.figure()
    # Plot results
    plt.subplot(211)
    plt.cla()
    plt.title("resonator spectroscopy amplitude")
    plt.plot(freqs_dem / u.MHz, np.sqrt(I ** 2 + Q ** 2), ".")
    plt.xlabel("frequency [MHz]")
    plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
    plt.subplot(212)
    plt.cla()
    # detrend removes the linear increase of phase
    phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
    plt.title("resonator spectroscopy phase")
    plt.plot(freqs_dem / u.MHz, phase, ".")
    plt.xlabel("frequency [MHz]")
    plt.ylabel("Phase [rad]")
    plt.pause(0.1)
    plt.tight_layout()

    time.sleep(2)

    print('Res spec analysis finished...')