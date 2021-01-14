---
title: ALLXY
sidebar_label: ALLXY
slug: ./
id: index
---

# ALLXY Calibration sequence

This script presents an ALLXY calibration sequence for Pi pulses which is more robust than the usual power Rabi or Ramsey
procedures for finding the correct amplitude and detuning. This is done by applying all the combinations of pi and pi/2 
pulses around the X and Y axes.
This is based on work done in https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf
or https://rsl.yale.edu/sites/default/files/physreva.82.pdf-optimized_driving_0.pdf


## The program
We assume that we already have somewhat calibrated pi pulse `pi_gauss_op_qubit` in a guassian form.
Our goal is to find more precisely the amplitude and the detuning of our pulse. ALLXY sequence can do that because it
expolits rotation around different axes, which effectively samples errors more robustly. Moreover, the basic idea is that
the fidelity of each pair of pulses (rotation by pi or pi/2 around X or Y) has some first or second order dependence on 
the error in the amplitude and on the detuning. Thus, it enables us to systematically identify and calibrate a variety of errors such 
as: detuning, power, reflection and DRAG error.  
The program is simple and consists of measuring the expectation value of the Pauli Z operator after each pair of pulses.
The sequence of pulses is ordered in such a way that if our pulses were precise we would find the qubit in:  
- Ground state for the first 5 pairs  
- On the equator for the next 12 pairs  
- Excited state for the last 4 pairs  

In order to efficiently use the OPX capabilities we first map all the pulses to corresponding amplitudes and frame 
rotation angles. We suppose that we start from an almost pi pulse around the X axis `Pi_pulse`. We'll use that pulse to achieve the
rest of the operations.
- Pi/2  around X is achieved by taking `Pi_pulse` and multiplying the amplitude by 1/2
- Pi around Y is achieved by applying frame rotation of Pi/2 (virtual Z rotation), making the Y axis our new "X axis", 
  then applying `Pi_pulse`, and then rotating back the frame by -pi/2
- Pi/2 around Y, same as pi pulse but with 1/2 the amplitude  

Now, all we do is play each pair of pulses in the sequence, and measure the qubits state through the readout resonator. 
Using the measurement, we assume we know the threshold value of the Q components to distinguish the excited and ground 
states, we assign the Pauli Z value. We buffer and average the results for all pairs.  
Note: One must use active reset to high fidelity between each pair of pulses in order to achieve good results.

## Optimization
The program above is inserted into an optimization function, which calculates how far the measured expectation value is 
from the optimal one. Using that we optimize the IF frequency, and the amplitude of the pi pulse.