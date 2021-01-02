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

# Create a quantum machine based on the configuration.
QM1 = QMm.open_qm(config)

with program() as measureProg:
    ind = declare(int)
    r=Random()
    temp = declare(int)
    stream1 = declare_stream()
    stream2 = declare_stream()

    with for_(ind,0,ind<100,ind+1):
        save(ind,stream1)
        assign(temp,Random().rand_int(10))
        save(temp,stream2)

    with stream_processing():
        stream1.save_all('stream1')
        stream1.buffer(10).save_all('stream2')
        stream1.buffer(10,10).save_all('2d_buffer')
        stream1.buffer(10).average().save_all('stream2avg')
        stream1.buffer(10).average().save('stream2avg_single')
        stream1.buffer(3).map(FUNCTIONS.average()).save_all('buffer_average')
        stream2.zip(stream1).save_all('zipped_streams')

job = QM1.simulate(measureProg,
                   SimulationConfig(4000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])))

res = job.result_handles
str1=res.stream1.fetch_all()
str2=res.stream2.fetch_all()
str3=res.stream2avg.fetch_all()
str4=res.stream2avg_single.fetch_all()
str5=res.zipped_streams.fetch_all()
