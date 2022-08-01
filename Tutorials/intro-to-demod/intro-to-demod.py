"""
intro_to_integration.py: Demonstrate usage of the integration in the measure statement
Author: Gal Winer - Quantum Machines
Created: 31/12/2020
Revised by Tomer Feld - Quantum Machines
Revision date: 24/04/2022
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
from configuration import *
import matplotlib.pyplot as plt

# Open communication with the server.
qop_ip = None
qmm = QuantumMachinesManager(host=qop_ip)

# Sliced demodulation parameters
num_segments = 25
seg_length = readout_len // (4 * num_segments)

# Moving window demodulation parameters
samples_per_chunk = 25
chunks_per_window = 3

with program() as measureProg:
    ind = declare(int)
    I = declare(fixed)

    int_stream = declare_stream()
    acc_int_stream = declare_stream()
    mov_int_stream = declare_stream()
    raw_adc = declare_stream(adc_trace=True)

    sliced_demod_res = declare(fixed, size=int(num_segments))
    acc_demod_res = declare(fixed, size=int(num_segments))
    mov_demod_res = declare(fixed, size=10)

    measure("readout", "qe1", raw_adc, demod.full("cos", I))
    save(I, "full")

    reset_phase("qe1")
    measure("readout", "qe1", None, demod.sliced("cos", sliced_demod_res, seg_length))
    with for_(ind, 0, ind < num_segments, ind + 1):  # save a QUA array
        save(sliced_demod_res[ind], int_stream)

    reset_phase("qe1")
    measure("readout", "qe1", None, demod.accumulated("cos", acc_demod_res, seg_length))
    with for_(ind, 0, ind < num_segments, ind + 1):  # save a QUA array
        save(acc_demod_res[ind], acc_int_stream)

    measure(
        "readout",
        "qe1",
        None,
        demod.moving_window("cos", mov_demod_res, samples_per_chunk, chunks_per_window),
    )
    with for_(ind, 0, ind < num_segments, ind + 1):
        save(mov_demod_res[ind], mov_int_stream)

    with stream_processing():
        int_stream.save_all("demod_sliced")
        acc_int_stream.save_all("demod_acc")
        mov_int_stream.save_all("demod_mov")
        raw_adc.input1().save("raw_input")

job = qmm.simulate(
    config,
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
plt.plot(res.raw_input.fetch_all() / 2**12)
plt.xlabel("t[nS]")
plt.ylabel("output [V]")
plt.title("Raw output")

plt.figure()
plt.plot(res.demod_mov.fetch_all()["value"] / 2**12, "o-")
plt.xlabel("sample number")
plt.title("moving windows")
