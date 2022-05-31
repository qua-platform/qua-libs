"""
A script used to look at the raw ADC data, this allows checking that the ADC is not saturated and defining the
threshold for time tagging.
"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import matplotlib.pyplot as plt
from configuration import *

###################
# The QUA program #
###################

with program() as TimeTagging_calibration:
    raw_adc_st = declare_stream(adc_trace=True)
    play("laser_ON", "AOM")
    measure("long_readout", "SPCM", raw_adc_st)
    wait(1000, "SPCM")

    with stream_processing():
        raw_adc_st.input1().save("raw_adc")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

qm = qmm.open_qm(config)

job = qm.execute(TimeTagging_calibration)
job.result_handles.wait_for_all_values()
res_handles = job.result_handles
raw_data = res_handles.get("raw_adc").fetch_all()
plt.plot(raw_data)
plt.title("ADC Trace Check ADCs saturation and define threshold")
