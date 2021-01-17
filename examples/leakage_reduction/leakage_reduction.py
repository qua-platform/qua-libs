from lr_lib import *
import numpy as np
import matplotlib.pyplot as plt

##we start with a standard DRAG pulse
ts = np.linspace(0.0, 4.0*gauss_len, gauss_len)
config["pulses"]["DRAG_PULSE"]["length"] = gauss_len
config["waveforms"]["gauss_wf"]["samples"] = [DRAG_I(1.0, 2.0*gauss_len, 4.0*gauss_len)(t) for t in ts]
config["waveforms"]["DRAG_gauss_wf"]["samples"] = [DRAG_I(1.0, 2.0*gauss_len, 4.0*gauss_len)(t) for t in ts]
config["waveforms"]["DRAG_gauss_der_wf"]["samples"] = [DRAG_Q(1.0, 2.0*gauss_len, 4.0*gauss_len)(t) for t in ts]

##we then optimize a regular DRAG pulse
np.random.seed(3)
es1 = cma.CMAEvolutionStrategy(np.random.rand(3), 0.5)
es1.optimize(cost_DRAG)
es1.result_pretty()

##Finally, we use the optimized DRAG pulse to add more degrees of freedom
## use A, B, freq as initial guess for the full optimization
init = list(es1.result.xbest) + list(np.random.rand(n_params))
es2 = cma.CMAEvolutionStrategy(init, 0.5)
es2.optimize(cost_optimal_pulse)
es2.result_pretty()

#We can now draw the optimal pulse
opt_pulse=np.array(get_DRAG_pulse('x',es2.result.xbest,40))
plt.plot(opt_pulse[0,:])
plt.plot(opt_pulse[1,:])