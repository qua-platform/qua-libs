"""
intro_to_integration.py: Demonstrate usage of the integration in the measure statement
Author: Gal Winer - Quantum Machines
Created: 31/12/2020
Revised by Tomer Feld - Quantum Machines
Revision date: 04/04/2022
Created on QUA version: 0.6.393
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
from configuration import *

# Open communication with the server.
qmm = QuantumMachinesManager()

with program() as measureProg:
    ind = declare(int)
    r = Random()
    temp = declare(int)
    stream1 = declare_stream()
    stream2 = declare_stream()

    with for_(ind, 0, ind < 100, ind + 1):
        save(ind, stream1)
        assign(temp, Random().rand_int(10))
        save(temp, stream2)

    with stream_processing():
        stream1.save_all("stream1")
        stream1.buffer(10).save_all("stream2")
        stream1.buffer(10).average().save_all("stream2avg")
        stream1.buffer(10).average().save("stream2avg_single")
        stream1.buffer(3).map(FUNCTIONS.average()).save_all("buffer_average")
        stream2.zip(stream1).save_all("zipped_streams")
        stream1.buffer(10, 10).save_all("two_d_buffer")

# Simulate the program on the server
job = qmm.simulate(
    config,
    measureProg,
    SimulationConfig(
        4000,  # Duration of simulation in units of clock cycles (4 ns)
        simulation_interface=LoopbackInterface(
            [("con1", 1, "con1", 1)]
        ),  # Simulate a loopback from analog output 1 to analog input 1
    ),
)

# Fetch the results of the simulation
res = job.result_handles
str1 = res.stream1.fetch_all()
str2 = res.stream2.fetch_all()
str3 = res.stream2avg.fetch_all()
str4 = res.stream2avg_single.fetch_all()
str5 = res.buffer_average.fetch_all()
str6 = res.zipped_streams.fetch_all()
str7 = res.two_d_buffer.fetch_all()
