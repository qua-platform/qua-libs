from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from scipy.optimize import minimize

from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

###################
# The QUA program #
###################

# %matplotlib qt

n_runs = 10000

cooldown_time = 5 * qubit_T1 // 4

with program() as IQ_blobs:
    n = declare(int)
    I_g = declare(fixed)
    Q_g = declare(fixed)
    I_g_st = declare_stream()
    Q_g_st = declare_stream()
    I_e = declare(fixed)
    Q_e = declare(fixed)
    I_e_st = declare_stream()
    Q_e_st = declare_stream()
    i = declare(int)
    
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    
    I_th = declare(fixed, value=0.0)
    
    adc_st = declare_stream(adc_trace=True)

    # frame_rotation_2pi(0.0, 'squeeze_rise','squeeze_drive','squeeze_fall')
    # frame_rotation_2pi(0.25, 'cqr_drive')

    with for_(n, 0, n < n_runs, n + 1):
        
        # update_frequency('squeeze_drive', int(0))
        # update_frequency('squeeze_rise', int(0))
        # update_frequency('cqr_drive', int(0))
        # update_frequency('resonator', int(0))
        
        
        play('on', 'squeeze_switch', duration=int((4e6+1e3)//4))
        play('ftc_rise', 'squeeze_rise')
        align('squeeze_rise', 'squeeze_drive')
        play('cw', 'squeeze_drive', duration=int(4e6//4))
        align('squeeze_drive', 'squeeze_fall')
        play('ftc_fall', 'squeeze_fall')
        
        wait(cooldown_time, 'cqr_drive', 'resonator')
        wait(int(2e2//4), 'cqr_drive', 'resonator')
        
        with for_(i, 0, i < 1, i+1):
            play('on', 'cqr_switch', duration=(cqr_len//4))
            play('cqr', 'cqr_drive')
            play('on', 'SPC_pump', duration=(passive_len//4))
            measure('passive_readout', 'resonator', None, demod.full('rotated_cos', I, 'out1'), demod.full('rotated_sin', Q, 'out1'))
            save(I, I_st)
            save(Q, Q_st)

        align()
        
        wait(10*cooldown_time)

    with stream_processing():
        # adc_st.input1().save_all("adc1")
        I_st.save_all('I')
        Q_st.save_all('Q')
    #     I_g_st.save_all("I_g")
    #     Q_g_st.save_all("Q_g")
    #     I_e_st.save_all("I_e")
    #     Q_e_st.save_all("Q_e")

#####################################
#  Open Communication with the QOP  #
#####################################
# qmm = QuantumMachinesManager()


qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)
# job = qmm.simulate(config, IQ_blobs, SimulationConfig(duration=int(20e3//4)), flags=['auto-element-thread'])
# job.get_simulated_samples().con1.plot()

qm = qmm.open_qm(config)

job = qm.execute(IQ_blobs)
res_handles = job.result_handles
res_handles.wait_for_all_values()

I = res_handles.get('I').fetch_all()['value']
Q = res_handles.get('Q').fetch_all()['value']
# adc1 = res_handles.get("adc1").fetch_all()['value'] / 2**12


plt.figure()
# plt.plot(I, Q,'o')
# plt.hist2d(I,Q, bins = 50, norm = mpl.colors.LogNorm())
plt.hist2d(I,Q, bins = 50)
plt.axis('equal')

