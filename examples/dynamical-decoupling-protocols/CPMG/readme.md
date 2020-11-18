---
id: index
title: CPMG
sidebar_label: CPMG
slug: ./
---

The CPMG sequence is a dynamical decoupling scripts (see [[1]](#1)) . 

We follow the pulse sequence outlined in [[2]](#2) and implement a macro for this pulse sequence. 


## Config

The system implements one control quantum element `qe1` and one readout quantum element `rr`

## Program

We nest an averaging loop and a delay loop which goes over `tau` values. 

## Post processing

None

## Script

[download script](CPMG.py)

## References

<a id="1">[1]</a> Wang, Z. H., De Lange, G., Ristè, D., Hanson, R., & Dobrovitski, V. V. (2012). Comparison of dynamical decoupling protocols for a nitrogen-vacancy center in diamond. Physical Review B - Condensed Matter and Materials Physics, 85(15), 1–16. https://doi.org/10.1103/PhysRevB.85.155204

<a id="2">[2]</a> Ali Ahmed, M. A., Álvarez, G. A., & Suter, D. (2013). Robustness of dynamical decoupling sequences. Physical Review A - Atomic, Molecular, and Optical Physics, 87(4), 1–6. https://doi.org/10.1103/PhysRevA.87.042309

