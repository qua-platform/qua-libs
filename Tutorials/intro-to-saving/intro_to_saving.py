"""
intro_to_saving.py: Demonstrate saving and data processing
Author: Gal Winer - Quantum Machines
Created: 7/11/2020
Revised by Tomer Feld - Quantum Machines
Revision date: 24/04/2022
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import Math
from configuration import *
from qm import SimulationConfig

# Open communication with the server.
qop_ip = None
qmm = QuantumMachinesManager(host=qop_ip)

# 1. Assigning values to variables and saving variables to tags
with program() as saving_a_var:
    a = declare(int, value=5)
    b = declare(fixed, value=0.23)
    save(a, "a_var")
    assign(a, 7)
    save(a, "a_var")
    assign(a, a + 1)
    save(a, "a_var")
    assign(b, Math.sin2pi(b))
    save(b, "b_var")

job = qmm.simulate(config, saving_a_var, SimulationConfig(1000))

res = job.result_handles
a = res.a_var.fetch_all()["value"]
b = res.b_var.fetch_all()["value"]

print("##################")
print("1: Assigning values to variables and saving variables to tags")
print(f"a={a}")
print(f"b={b}")
print("##################")


# 2. Saving variables to streams and using stream processing
with program() as streamProg:
    out_str = declare_stream()
    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str)

    with stream_processing():
        # Average all of the data and save only the last value into "out".
        out_str.average().save("out")

job = qmm.simulate(config, streamProg, SimulationConfig(500))

res = job.result_handles
out = res.out.fetch_all()

print("##################")
print("2: Saving variables to streams and using stream processing")
print(f"out={out}")
print("##################")


# 3. Using the buffer operator in stream processing
with program() as streamProg_buffer:
    out_str = declare_stream()

    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str)

    with stream_processing():
        # Group output into vectors of length 3. Since only full buffers are used,
        # the last 2 data points [99 100] are discarded.
        # Perform a running average over the data, in group of 3:
        # The first vector is [0 1 2] and it averages only with itself.
        # The second vector is [3 4 5] and it averages with the 1st vector, giving [1.5 2.5 3.5].
        # etc...
        # This time 'save_all' is used, so all of the data is saved ('save' would have only saved [48 49 50])
        out_str.buffer(3).average().save_all("out")

job = qmm.simulate(config, streamProg_buffer, SimulationConfig(500))

res = job.result_handles
out = res.out.fetch_all()["value"]

print("##################")
print("3: Using the buffer operator in stream processing")
print(f"out={out}")
print("##################")

# 4. Saving a stream to multiple tags
with program() as multiple_tags:
    out_str1 = declare_stream()

    a = declare(int)
    with for_(a, 0, a <= 100, a + 1):
        save(a, out_str1)

    with stream_processing():
        # Two separate streams are used on the data:
        # 1. Put the data into vectors of length 2, average and save only the last one
        # 2. Save all of the raw data directly
        out_str1.buffer(2).average().save("out_avg")
        out_str1.save_all("out_raw")

job = qmm.simulate(config, multiple_tags, SimulationConfig(500))

res = job.result_handles
out_avg = res.out_avg.fetch_all()
out_raw = res.out_raw.fetch_all()["value"]
print("##################")
print("4:Saving a stream to multiple tags")
print(f"out_avg={out_avg}")
print(f"out_raw={out_raw}")
print("##################")

# 5. Using multi-dimensional buffer operator in stream processing
with program() as streamProg_buffer:
    out_str = declare_stream()

    a = declare(int)
    b = declare(int)
    with for_(a, 0, a <= 10, a + 1):
        with for_(b, 10, b < 40, b + 10):
            save(b, out_str)

    with stream_processing():
        out_str.buffer(11, 3).save("out")

job = qmm.simulate(config, streamProg_buffer, SimulationConfig(500))

res = job.result_handles
out = res.out.fetch_all()

print("##################")
print("5: Using the multi-dimensional buffer operator in stream processing")
print(f"out={out}")
print("##################")
