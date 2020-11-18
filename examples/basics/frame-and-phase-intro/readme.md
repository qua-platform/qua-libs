---
id: index
title: Frame rotation and phase reset - An introduction 
sidebar_label: Frame&Phase
slug: ./
---

This simple script is designed to showcase how phase and frame work in QUA.

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

This program includes several sets of pulse, where in some of the sequences a frame rotation has
been applied and in others it has been reset. When a frame rotation has been reset, 
the pulse plays as if it were accumulating phase without rotation from the beginning of the script. 
The phase reset acts to "zero the clock", setting the phase as if the program has restarted. 

## Script 

[download script](reset_phase_demo.py)
 