---
title: Suppressing qubit dephasing using real-time Hamiltonian estimation
sidebar_label: QD Bayesian Estimation
slug: ./
id: index
---
The system under consideration here is a S-T0 qubit residing in a pair of coupled quantum dots in GaAs/AlGaAs
[This paper](https://www.nature.com/articles/ncomms6156).

In this qubit, there are two control knobs for setting the state of the qubit. The more intuitive of the two is an
electric knob: this controls the relative depth of the two potential wells
(ε, the detuning) and the coupling between the wells (J). The second knob is due to the magnetic field experienced by
the quantum dots. An external field is indeed applied by (typically) a superconducting electromagnet, but this is not
the only contribution. There is also a fluctuating magnetic field $\Delta B_z$ due to the spin of the nuclear spins in the
substrate hosting the quantum dots.

The qubit state precesses around the axis of this magnetic field, and moving to the rotating frame is therefore
desirable to mitigate this decohering effect.  
However, due to fluctuations of the field, it is not possible to simply switch to the rotating frame by modulating the
level of detuning. It is possible, by manipulations of the qubit state, to polarize the substrate (so called -
“pumping”) thus minimizing the fluctuations but this has limited effect as there are residual fluctuations.
Nevertheless, as the rate of fluctuation is slower than the experimental cycle timescale, a protocol to estimate the
magnetic field just before the experiment can be used to switch to the correct rotating frame.

This can be done, for example, by employing Bayesian estimation of the magnetic field. This technique uses a prior by
which all magnetic field values in some range are equally probable and uses Bayes theorem to repeatedly update the (
posterior) distribution. The most probably field of the final distribution is assumed to be the external field for the
duration of the experiment.

To perform Bayesian estimation, a single-shot state evaluation needs to first be performed. This is done in the first
row of the snippet below. The state is measured by measuring the reflection of an RF signal from a Quantum Point
Contact (QPC) near the double-dot. The reflected signal is demodulated and the demodulated value is saved to a variable.
All this is accomplished with just a single statement.

The following expression then needs to be repeatedly evaluated (in real time) for each possible value of magnetic field:

$$P(m_k|\Delta B_z)=\frac{1}{2}[1+r_k (\alpha+\beta cos(2\pi \Delta B_z t_k)]$$

Where $m_k$ is the qubit state ($S$ or $T_0$), $r_k$ is a factor equal to 1 or -1 depending on the state, 
and $t_k$ is the evolution time of the $k^{th}$ measurement. $\alpha$ and $\beta$ are numeric
constants.

This is done in the first `for` loop. Note the usage of casting and trigonometric functions which are efficiently
implemented on the QOP processor. The second `for` loop a normalization of the resulting probability distribution.
The following section implements equations (1)-(4) in the paper: 
```python
measure('measure', 'RF-QPC', None, demod.full('integW1', I))
assign(state[k - 1], I > 0)
save(state[k - 1], state_str)
assign(rk, Cast.to_fixed(state[k - 1]) - 0.5)
with for_(fB, fB_min, fB < fB_max, fB + dfB):
    assign(C, Math.cos2pi(Cast.mul_fixed_by_int(fB, t_samp * k)))
    assign(Pf[ind1], (0.5 + rk * (alpha + beta * C)) * Pf[ind1])
    assign(ind1, ind1 + 1)

assign(norm, 1 / Math.sum(Pf))
with for_(ind1, 0, ind1 < Pf.length(), ind1 + 1):
    assign(Pf[ind1], Pf[ind1] * norm)
```

The maximum value of the resulting vector is then taken as the most probable magnetic
field value, and a target procedure can be run, at a frame rotating with the qubit.
In this script, a time Rabi experiment is run, but any other procedure can be selected.  





