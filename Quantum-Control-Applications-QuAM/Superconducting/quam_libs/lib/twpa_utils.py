import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erf

def voltTOdbm(volt):
    p_w=(volt**2)/50
    dbm=10*np.log10(p_w*1000)
    return dbm
################ 250922 V2 #####################################
def dBm(full_scale_power_dbm,daps):
    v=np.sqrt((2*50*10**(full_scale_power_dbm/10))/1000)*daps*1 # 1 : twpa readout amplitude  #opx1000 documentation
    p_w=(v**2)/50
    dbm=10*np.log10(p_w*1000)-10
    return dbm +5.68 #5.68 calibrated through SA on 14/09
def pump_maxgain(gain, dfps, daps):
    avg_gain=np.mean(gain,axis=0)
    max_gain_idx=np.unravel_index(np.argmax(avg_gain), avg_gain.shape)
    max_gain_pump=np.array(np.array([dfps[max_gain_idx[0]],daps[max_gain_idx[1]]]))
    return max_gain_pump, max_gain_idx
def pump_maxdsnr(dsnr, dfps, daps):
    avg_dsnr=np.mean(dsnr,axis=0)
    max_dsnr_idx=np.unravel_index(np.argmax(avg_dsnr), avg_dsnr.shape)
    max_dsnr_pump=np.array(np.array([dfps[max_dsnr_idx[0]],daps[max_dsnr_idx[1]]]))
    return max_dsnr_pump, max_dsnr_idx
def mvTOdbm(mv):
    v=mv*1e-3
    rms_v=v/np.sqrt(2)
    p_watt=((rms_v)**2)/50
    dbm=10*np.log10(p_watt*1000)
    return dbm
def snr(ds, qubits, dfps, daps):
    noise=np.zeros((len(qubits),len(dfps),len(daps),1))
    signal=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                noise[i,j,k]=mvTOdbm(np.mean(ds.IQ_abs_noise.values[i][j][k]))
                signal[i,j,k]=mvTOdbm(ds.IQ_abs_signal.values[i][j][k][len(ds.IQ_abs_signal.values[i][j][k])//2])
    return signal-noise

def gain(ds_pumpoff,ds_pumpon, qubits, dfps, daps):
    signal_pumpoff=np.zeros((len(qubits),len(dfps),len(daps),1))
    signal_pumpon=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
        for j in range(len(dfps)):
            for k in range(len(daps)):
                signal_pumpoff[i,j,k]=voltTOdbm(np.mean(ds_pumpoff.IQ_abs_signal.values[i][j][k]))
                signal_pumpon[i,j,k]=voltTOdbm(np.mean(ds_pumpon.IQ_abs_signal.values[i][j][k]))
    return signal_pumpon-signal_pumpoff
################ 250928 V3 #####################################
def signal(ds):
    avg_signal=ds.IQ_abs_signal.values.mean(axis=-1, keepdims=True)
    return voltTOdbm(avg_signal)
def noise(ds, qubits, dfps, daps, n_avg):
    I=np.zeros((len(qubits),len(dfps),len(daps),n_avg))
    Q=np.zeros((len(qubits),len(dfps),len(daps),n_avg))
    for i in range(len(qubits)):
            for j in range(len(dfps)):
                for k in range(len(daps)):
                    for n in range(n_avg):
                        I[i][j][k][n]=ds.I.values[i][n][j][k]
                        Q[i][j][k][n]=ds.Q.values[i][n][j][k]
    I_noise=np.zeros((len(qubits),len(dfps),len(daps),1))
    Q_noise=np.zeros((len(qubits),len(dfps),len(daps),1))
    for i in range(len(qubits)):
            for j in range(len(dfps)):
                for k in range(len(daps)):
                    I_noise[i][j][k]=np.std(I[i][j][k])
                    Q_noise[i][j][k]=np.std(I[i][j][k])
    return (voltTOdbm(I_noise)+voltTOdbm(Q_noise))/2 #is it ok to define the noise as avg of IQ std
######################## optimizer
def lin(db_value):
        return 10 ** (db_value / 10)
def fidelity(ro,t1,snr,pgg): #snr=distance/(distribution of g,e)
    fidelity=np.exp(-ro/(2*t1))*erf(snr/(np.sqrt(2)))*pgg
    return fidelity  
def pa(qubits, dfps, daps):
    target_shape = (len(qubits), len(dfps), len(daps), 1)
    total_elements = np.prod(target_shape)
    pa = np.resize(daps, total_elements).reshape(target_shape)
    return pa
def fp(qubits, dfps, daps):
    fp = np.zeros((len(qubits), len(dfps), len(daps), 1))
    fp[:, :, :, 0] = dfps[np.newaxis, :, np.newaxis]
    return fp
def min_dsnr(qubits,dsnr_avg,snr_off,pgg, dfps, daps): 
    twpa_fct=np.sqrt(lin(dsnr_avg))
    f=fidelity(qubits[0].resonator.operations["readout"].length*1e-3,
               qubits[0].T1*1e6, 
               snr_off*twpa_fct,
               pgg)
    f_=f.reshape(len(dfps)*len(daps))
    sat_idx=np.where((np.max(f_)-f_)*100<0.1)
    sat_dsnr = dsnr_avg.reshape(len(dfps)*len(daps))[sat_idx[0]]
    return min(sat_dsnr)
def min_gain(qubits,  twpas):
    ##### 1.max a_ro_on_max s.t) #(mtpx Ro)*Pon < Psat-10 : total readout power should be smaller than saturation power
    ##### get the da , theoretically can be used 
    a_ro=np.linspace(0,1,500)
    ps_on=dBm(qubits[0].resonator.opx_output.full_scale_power_dbm,
            a_ro*qubits[0].resonator.operations["readout"].amplitude)-60-6-5
    mtpx_ps_on= ps_on + 10*np.log10(len(qubits))
    idx=np.where(mtpx_ps_on<twpas[0].p_saturation-20)[0][-1]
    a_ro_on_max=a_ro[idx]
    ###### 2. a_ro_off~a_ro_off*a_ro_on*sqrt(linG) : 
    ###### pick the Gain which compensate the lowered readout amplitude up to the readout amplitude when twpa is off
    g=np.linspace(0,25,500)
    a_ro_off=1 # value doesnt matter
    idx_=np.where(a_ro_off<a_ro_off*a_ro_on_max*np.sqrt(10**(g/10)))
    minimum_gain=g[idx_[0][0]]
    # to guarantee TWPA gain suppress HEMT noise
    if minimum_gain<10:
        minimum_gain=10
    elif minimum_gain>=10:
        minimum_gain=minimum_gain
    return minimum_gain

def optimizer(mingain, mindsnr, gain_avg, dsnr_avg, daps, dfps, p_lo,p_if):
    mask = gain_avg > mingain
    masked_dsnr = np.where(mask, dsnr_avg, -np.inf)
    flat_index = np.argmax(masked_dsnr)
    idx = np.unravel_index(flat_index, dsnr_avg.shape)
    print(f"Optimized ap={np.round(daps[idx[1]],5)},fp={np.round((p_lo+p_if+dfps[idx[0]])*1e-9,3)}GHz ")
    print(f"gain_avg :{np.round(gain_avg[idx],2)}dB")
    print(f"dsnr_avg :{np.round(dsnr_avg[idx],2)}dB")
    return idx
#############################
def resonator_fit(f, f0, gamma, a, c0, c1):
    # f0 [GHz], gamma [GHz], a > 0 is dip depth, baseline c0 + c1*(f - f0)
    return (c0 + c1*(f - f0)) - a / (1.0 + ((f - f0)/gamma)**2)
def fit_resonator_dBm(f_GHz, s21_dBm): 
    f = np.asarray(f_GHz, dtype=float)
    s21 = np.asarray(s21_dBm, dtype=float)
    # ---- initial guesses ----
    idx_min = np.argmin(s21)
    f0_0 = f[idx_min]
    span = f.max() - f.min()
    gamma_0 = 0.15 * span if span > 0 else 0.001
    a_0 = float(np.median(s21) - s21[idx_min])  # positive dip depth in dB
    c0_0 = float(np.median(s21))
    c1_0 = (s21[-1] - s21[0]) / (f[-1] - f[0]) if span > 0 else 0.0

    p0 = [f0_0, gamma_0, max(a_0, 0.1), c0_0, c1_0]
    bounds = (
        [f.min(), 1e-6*span if span > 0 else 1e-6, 0.0, -np.inf, -np.inf],
        [f.max(), 2.0*span if span > 0 else 1.0, np.inf, np.inf, np.inf]
    )

    # ---- fit ----
    popt, pcov = curve_fit(resonator_fit, f, s21, p0=p0, bounds=bounds, maxfev=20000)
    f0_fit, gamma_fit, a_fit, c0_fit, c1_fit = popt
    fwhm = 2 * gamma_fit *1e3

    # ---- plot ----
    ff = np.linspace(f.min(), f.max(), 800)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(f, s21, 'b.', label='Data (dBm)')
    plt.plot(ff, resonator_fit(ff, *popt), 'r-', lw=1.5,
             label=f"Fit (f0={f0_fit:.3f} GHz, kappa≈{fwhm:.2f}, depth≈{a_fit:.2f} dB)")
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("|S21| (dBm)")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
    print(f"f0 = {f0_fit:.3f} GHz,kappa≈{fwhm:.2f}MHz, Dip depth = {a_fit:.2f} dB")
    return a_fit
def signal_size(f_GHz, s21_dBm):
 
    f = np.asarray(f_GHz, dtype=float)
    s21 = np.asarray(s21_dBm, dtype=float)
    # ---- initial guesses ----
    idx_min = np.argmin(s21)
    f0_0 = f[idx_min]
    span = f.max() - f.min()
    gamma_0 = 0.15 * span if span > 0 else 0.001
    a_0 = float(np.median(s21) - s21[idx_min])  # positive dip depth in dB
    c0_0 = float(np.median(s21))
    c1_0 = (s21[-1] - s21[0]) / (f[-1] - f[0]) if span > 0 else 0.0

    p0 = [f0_0, gamma_0, max(a_0, 0.1), c0_0, c1_0]
    bounds = (
        [f.min(), 1e-6*span if span > 0 else 1e-6, 0.0, -np.inf, -np.inf],
        [f.max(), 2.0*span if span > 0 else 1.0, np.inf, np.inf, np.inf]
    )

    # ---- fit ----
    popt, pcov = curve_fit(resonator_fit, f, s21, p0=p0, bounds=bounds, maxfev=20000)
    f0_fit, gamma_fit, a_fit, c0_fit, c1_fit = popt

    return a_fit
def _fit_one_trace(f, s21):
    f = np.asarray(f, dtype=float)
    s21 = np.asarray(s21, dtype=float)

    idx_min = np.argmin(s21)
    f0_0 = f[idx_min]
    span = float(f.max() - f.min())
    gamma_0 = 0.15 * span if span > 0 else 1e-3
    a_0 = float(np.median(s21) - s21[idx_min])  # positive dip depth in dB
    c0_0 = float(np.median(s21))
    c1_0 = (s21[-1] - s21[0]) / (f[-1] - f[0]) if span > 0 else 0.0

    p0 = [f0_0, gamma_0, max(a_0, 0.1), c0_0, c1_0]
    bounds = (
        [f.min(), 1e-6*span if span > 0 else 1e-6, 0.0, -np.inf, -np.inf],
        [f.max(), 2.0*span if span > 0 else 1.0,  np.inf,  np.inf,  np.inf]
    )

    popt, pcov = curve_fit(resonator_fit, f, s21, p0=p0, bounds=bounds, maxfev=20000)
    f0_fit, gamma_fit, a_fit, c0_fit, c1_fit = popt
    return a_fit  # dip depth in dB
def signal_size_multi(f_arr, s21_arr):
    """
    Return array of a_fit (dip depth in dB) for multiple traces.

    Accepts:
      - 2D numpy arrays: f_arr.shape == s21_arr.shape == (n_traces, n_points)
      - lists/tuples of 1D arrays: len(f_arr) == len(s21_arr) == n_traces
        (use this when traces have different lengths)

    Returns:
      a_fits: np.ndarray of shape (n_traces,)
    """
    # Case 1: list/tuple of per-trace arrays (variable lengths OK)
    if isinstance(f_arr, (list, tuple)):
        assert isinstance(s21_arr, (list, tuple)) and len(f_arr) == len(s21_arr), \
            "f_arr and s21_arr must be lists/tuples of the same length"
        a_list = []
        for f, s21 in zip(f_arr, s21_arr):
            a_list.append(_fit_one_trace(f, s21))
        return np.asarray(a_list, dtype=float)

    # Case 2: numpy arrays
    f_arr = np.asarray(f_arr, dtype=float)
    s21_arr = np.asarray(s21_arr, dtype=float)
    assert f_arr.shape == s21_arr.shape and f_arr.ndim == 2, \
        "For numpy arrays, use shape (n_traces, n_points) for both f_arr and s21_arr"

    n_traces = f_arr.shape[0]
    a_out = np.empty(n_traces, dtype=float)
    for i in range(n_traces):
        a_out[i] = _fit_one_trace(f_arr[i], s21_arr[i])
    return a_out
def delta_s(RF_freq, ds_off_s, ds_on_s, qubits, dfps, daps):
    f_stack = np.stack([RF_freq[i]*1e-9 for i in range(len(qubits)) for j in range(len(dfps)) for k in range(len(daps))], axis=0)
    s21_off_stack = np.stack([
        voltTOdbm(ds_off_s.IQ_abs_signal.values[k][i][j])
        for k in range(len(qubits)) for i in range(len(dfps)) for j in range(len(daps))
    ], axis=0) 
    s21_on_stack = np.stack([
        voltTOdbm(ds_on_s.IQ_abs_signal.values[k][i][j])
        for k in range(len(qubits)) for i in range(len(dfps)) for j in range(len(daps))
    ], axis=0) 
    signalsize_off = signal_size_multi(f_stack, s21_off_stack)
    signalsize_on = signal_size_multi(f_stack, s21_on_stack)
    return signalsize_on-signalsize_off

