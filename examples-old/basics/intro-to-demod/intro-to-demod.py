"""
intro_to_integration.py: Demonstrate usage of the integration in the measure statement
Author: Gal Winer - Quantum Machines
Created: 31/12/2020
Created on QUA version: 0.6.393
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import SimulationConfig, LoopbackInterface
from configuration import *
import matplotlib.pyplot as plt

# Open communication with the server.
QMm = QuantumMachinesManager()


num_segments = 25

seg_length = pulse_len // (4 * num_segments)

samples_per_chunk = 25
chunks_per_window = 3

config["integration_weights"]["xWeights"]["cosine"] = [0.1] * num_segments * seg_length
config["integration_weights"]["yWeights"]["sine"] = [0.0] * num_segments * seg_length

# Create a quantum machine based on the configuration.
QM1 = QMm.open_qm(config)

with program() as measureProg:
    ind = declare(int)
    I = declare(fixed)

    int_stream = declare_stream()
    acc_int_stream = declare_stream()
    mov_int_stream = declare_stream()

    sliced_demod_res = declare(fixed, size=int(num_segments))
    acc_demod_res = declare(fixed, size=int(num_segments))
    mov_demod_res = declare(fixed, size=int(10))
    measure("readoutOp", "qe1", "raw", demod.full("x", I))

    save(I, "full")

    reset_phase("qe1")
    measure("readoutOp", "qe1", None, demod.sliced("x", sliced_demod_res, seg_length))

    with for_(ind, 0, ind < num_segments, ind + 1):
        save(sliced_demod_res[ind], int_stream)

    reset_phase("qe1")
    measure("readoutOp", "qe1", None, demod.accumulated("x", acc_demod_res, seg_length))
    with for_(ind, 0, ind < num_segments, ind + 1):
        save(acc_demod_res[ind], acc_int_stream)

    measure(
        "readoutOp",
        "qe1",
        None,
        demod.moving_window("x", mov_demod_res, samples_per_chunk, chunks_per_window),
    )
    with for_(ind, 0, ind < num_segments, ind + 1):
        save(mov_demod_res[ind], mov_int_stream)

    with stream_processing():
        int_stream.save_all("demod_sliced")
        acc_int_stream.save_all("demod_acc")
        mov_int_stream.save_all("demod_mov")

job = QM1.simulate(
    measureProg,
    SimulationConfig(4000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)

res = job.result_handles
full = res.full.fetch_all()["value"]
print(f"Result of full demodulation: {full}")
sliced = res.demod_sliced.fetch_all()["value"]
print(f"Result of sliced demodulation in {num_segments} segments:{sliced}")

[f, (ax1, ax2)] = plt.subplots(nrows=1, ncols=2)
ax1.plot(res.demod_sliced.fetch_all(), "o-")
ax1.set_title("sliced demod")
ax1.set_xlabel("slice number")

ax2.plot(res.demod_acc.fetch_all(), "o-")
ax2.set_title("acc demodulation")
ax2.set_xlabel("slice number")

plt.figure()
plt.plot(res.raw_input1.fetch_all()["value"] / 2**12)
plt.xlabel("t[nS]")
plt.ylabel("output [V]")
plt.title("Raw output")

plt.figure()
plt.plot(res.demod_mov.fetch_all()["value"] / 2**12, "o-")
plt.xlabel("sample number")
plt.title("moving windows")
