---
id: index
title: Hardware-efficient quantum random access memory with hybrid quantum acoustic systems
sidebar_label: QRAM
slug: ./
---

To run complex quantum algorithms it is often required to save and retrieve arbitrary quantum states.  
The concept of Quantum RAM is entirely like that of classical Random Access Memory (RAM) apart from 
the face it is able to save and retrieve qubits rather than bits. It also differs by supporting 
a string of qubits as a memory address, rather than a classical bit string. This allows a QRAM
to save and retrieve entangled quantum states as may be required by an algorithm. 

In this example we showcase a bucket-brigade QRAM Implementation as described in [[1]](#1).
This implementation assumes a multi-mode resonator where energy transfer between modes is facilitated 
by a non-linear element supplied by a superconducting qubit. 

# References
<a id="1">[1]</a> Hann, C. T., Zou, C. L., Zhang, Y., Chu, Y., Schoelkopf, R. J., Girvin, S. M., & Jiang, L. (2019). Hardware-Efficient Quantum Random Access Memory with Hybrid Quantum Acoustic Systems. Physical Review Letters, 123(25), 1–24. https://doi.org/10.1103/PhysRevLett.123.250501

