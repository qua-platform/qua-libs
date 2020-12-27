---
id: index
title: Frame rotation and phase reset - An introduction 
sidebar_label: Frame&Phase
slug: ./
---

This simple script is designed to showcase how phase and framework in QUA.

## Basics of phase in QAU

The phase of a signal output by the OPX is dependent on two factors: 

1. Time elapsed from the start of the program
2. The frame matrix

This topic is covered in full detail in the QUA docs, but we repeat some of this material here
for clarity. 

The first of the two phase factors is fairly straight forward and is essentially the `t` parameter 
in a complex exponential $$e^{-i\omega t}$$. The second phase contribution allows to 
change the phase such that the `I` and `Q` channels are experience a relative rotation:

$$\begin{pmatrix}
cos(\phi_F) & -sin(\phi_F)\\ 
sin(\phi_F) & cos(\phi_F)
\end{pmatrix} \begin{pmatrix}
I\\Q 

\end{pmatrix}
$$

## Script

This scripts plays a 5MHz pulse followed by a 10MHz pulse in 4 cases:
1. No phase change between them.
2. A pi rotation is applied to the 10MHz pulse.
3. A phase reset is applied to the 10MHz pulse.
4. A phase reset, and a pi rotation are applied to the 10MHz pulse.

A fit to the 10MHz pulse is done in order for the difference to be easily visualized.
In the 1st case, it is clear that the phase is maintained because the peaks of the 10MHz fit coincide with the peaks of the 5Mhz pulse.
In the 2nd case, the phase is maintained but with an extra Pi phase.
In the 3rd case, the phase is being reset, this causes a small delay before the 2nd pulse, and the phase is being reset to an arbitrary phase calculated from the start of the program.

[download script](reset_phase_demo.py)
 