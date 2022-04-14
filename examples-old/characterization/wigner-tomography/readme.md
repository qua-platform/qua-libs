---
id: index
title: Wigner tomography
sidebar_label: Wigner tomography
slug: ./
---

Wigner tomography is the process of extracting the Wigner function $W(\alpha)$ 
(a quasi-probabilistic distribution) of the quantum state of a cavity.
The Wigner function is related to the density matrix of a state, therefore the
tomography procedure will allow for the construction of the density matrix and a full 
reconstruction of the quantum state. Through a standard procedure one can encode the state of a qubit
in a superposition of coherent states in a cavity. Therefore, using Wigner
tomography it'll be possible to extract the full density matrix of the qubit.

The Wigner function is defined as follows $$W(\alpha) = 2/\pi \langle P\rangle_\alpha$$,
where $P$ is the photon parity operator and $\alpha$ is the same parameter in the coherent states
and represents a complex vector in the IQ plane.

Using a qubit coupled to the cavity it's straightforward to extract the photon parity of the cavity form
a repeated measurement of the qubit through an additional readout resonator. 
The parity is related to the qubit state as such: $\langle P\rangle \propto P_e - P_g$m where $P_e$ and $P_g$ are the probabilities of finding the qubit in the
excited and ground state respectively, which can be extracted with repeated measurement.

Notice: The example describes the tomography process assuming the cavity was encoded prior.

## Config

The configuration consists of  4 quantum elements:
* `cavity_I` and `cavity_Q` define single input elements and are the I and Q components of the cavity
 that we'll perform the tomography on.
* `qubit` is the qubit that's coupled to the cavity
* `rr` is the readout resonator that's coupled to the qubit and used to read its state

Each element has its own IF and LO frequencies, and connection ports. Next, for each element we define the relevant
operation and pulse:
* For `cavity_I` and `cavity_Q` we define the `displace_pulse`, which will be the real and imaginary parts of the displace 
pulse. These were separated due to a needed 2d parameter sweep over the amplitudes of the pulses for the tomography.
* For the `qubit` we define the `x_pi/2_pulse` which is simply a $\pi/2$ rotation around the x axis
* For the `rr` we define the `readout_pulse` - the pulse used for measuring the resonator.

The waveforms used for the `displace_pulse` and `x_pi/2_pulse` are Gaussians with different parameters.
Generally to displace a cavity one needs to apply a pulse such that it integrates to the desired $\alpha$.


## Program

We first calculate the revival time of the qubit coupled to the cavity. Then, we decide of the $\alpha$ range 
we want to sample for constructing the Wigner function, and the spacing. Once we defined the required parameters,
we proceed to the QUA program.

We first define the QUA fixed variable for the amplitude scaling required to shift the cavity by the desired $\alpha$
We than create 2 QUA `for_` loops to iterate over the points of the IQ grid. The inner-most `for_` loop is for repeated
measurement of the same point in the IQ plane.

Then, in each cycle we perform the tomography procedure:
* We align both cavity components in order to be played simultaneously. We displace the I and Q components by the real and 
imaginary parts of $\alpha$, respectively, this is done using realtime amplitude modulation, by multiplying the pulse
with the function `amp(x)`, where `x` is the scaling parameter.
* Next, we align the cavity with the qubit to ensure the pulses meant for the qubit start after reaching the desired coherent state for the cavity.
On the qubit we apply a `x_pi/2` operation to bring it to the equator, wait for the revival time, and eventually apply
a second `x_pi/2` operation to project the qubit to the excited or ground state.
* Finally, we measure the state using the readout resonator and demodulate the reflected signals to get the
qubits state on the IQ plane which can then determine its state.

```python
amp_displace = list(-alpha / np.sqrt(2 * np.pi) / 4)
amp_dis = declare(fixed, value=amp_displace)
with for_(r, 0, r < points, r + 1):
   with for_(i, 0, i < points, i + 1):
       with for_(n, 0, n < shots, n + 1):
           align("cavity_I", "cavity_Q")
           play("displace_I" * amp(amp_dis[r]), "cavity_I")
           play("displace_Q" * amp(amp_dis[i]), "cavity_Q")

           align("cavity_I", "cavity_Q", "qubit")
           play("x_pi/2", "qubit")
           wait(revival_time, "qubit")
           play("x_pi/2", "qubit")

           align("qubit", "rr")
           measure(
               "readout",
               "rr",
               "raw",
               demod.full("integW_cos", I1, "out1"),
               demod.full("integW_sin", Q1, "out1"),
               demod.full("integW_cos", I2, "out2"),
               demod.full("integW_sin", Q2, "out2"),
           )
           assign(I, I1 + Q2)
           assign(Q, -Q1 + I2)
           save(I, "I")
           save(Q, "Q")

           wait(
               10, "cavity_I", "cavity_Q", "qubit", "rr"
           )  # wait and let all elements relax
```                    
## Post processing

Having the I,Q results of repeated measurement of the qubit for different $\alpha$ we can extract the parity of the cavity
at each point by counting the excited and ground state measurements. We can display the results using a heatmap
which represents the IQ plane, with the axes being the real and imaginary parts of $\alpha$.    



## Sample output


## Script
[download script](wigner_tomography.py)
