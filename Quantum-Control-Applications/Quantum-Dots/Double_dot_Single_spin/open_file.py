# %%
import numpy as np

with np.load('CSD_P2P1.csv.npz') as data:

    I = data['arr_0']
    Q = data['arr_1']
    P2 = data['arr_2']
    P1 = data['arr_3']

import matplotlib.pyplot as plt

# plt.subplot(121)
# plt.pcolor(P2, P1, I)
plt.plot(P2, 1*P2 - 0.003)
plt.plot(P2, -1*P2 + 0.006)
P2_aux = P2[len(P1):]
I_aux = I[:,-len(P1):]
plt.pcolor(P2_aux, P1, I_aux)
epsilon = np.linspace(-0.005, 0.005, 40) + 0.0045
energy = np.linspace(-0.005, 0.005, 40) + 0.0045
for i in epsilon:
    plt.plot(i, -i+0.006, 'rx')
    plt.plot(i, i-0.003, 'bx')
    plt.axvline(x=(-0.005+0.0045)*np.sqrt(2))
# plt.subplot(122)
# plt.pcolor(P2, P1, Q)
# plt.tight_layout()
# %%

from qm.qua import *

with program() as prog:

    align()
    play('bias'*amp(-epsilon_min+0.006/P1_amp), 'P1_sticky')
    play('bias'*amp(epsilon_min/P2_amp), 'P2_sticky')
    align()
    
# %%
