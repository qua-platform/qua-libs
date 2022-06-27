---
title: Introduction to Simulation
sidebar_label: Intro to Simulation
slug: ./
id: index
---

This example shows usage of the hardware simulator. The simulator mimics the output of the hardware, 
with its exact timing and voltage level, following the compilation of QUA to the FPGA's low-level. 
It is a useful tool for predicting and debugging the outcome of a QUA program. 
Read more on the simulator, and it's capabilities in the [QUA docs](https://qm-docs.qualang.io/guides/simulator).

The Examples
============

The following script gives three examples for the usage of the simulator in the QOP.

First and second exampled
-------------

The first and second demonstrates basic usage od the simulator. Note that the simulation is done via the QuantumMachinesManager (qmm) object. 
The simulation function is given the configuration, the program and the `SimulationConfig` instance. The last sets the duration for the 
simulation. i.e. how many clock cycles should the simulator simulate. The outcome of the simulator is a `job` object. 

The examples also demonstrate how to obtain and plot the simulated output of the hardware. Finally, the first example also demonstrates
how the simulator can simulate saving variables to a stream, as would occur in the real hardware. 

Third Example
==============

The third example demonstrates a slightly more advanced usage. It shows how a loop-back connection can be defined, 
to simulate acquisition and demodulation of ADC signals. In the example, a connection from analog output 1 of controller 1
is connected to analog input 1 of controller 1. The example also shows that the demodulation and adc input can be simulated 
and saved to the stream processing, which later can be fetched and analyzed. It is important to note that the data will only
be available if the simulation duration was long enough to simulate it. In the current example, to "fill" the buffer in the stream processing, 
the simulation must simulate the entire program with all the loop's iterations.  

[download script](intro-to-simulation.py)
