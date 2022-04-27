---
id: index
title: XY-n
sidebar_label: XY-n
slug: ./
---

XY-n represnets a family of dynamical decoupling scripts (see [[1]](#1)) where X and Y rotations are interchanged at equal temporal spacing. 
We follow the pulse sequence outlined in [[2]](#2) and implement a macro which can generate a general number of integer 
repetitions.  
```python
def XY_n_sym(tau, n):
    ind = declare(int)
    wait(tau / 2, "qe1")
    with for_(ind, 0, ind < n, ind + 1):
        play("X", "qe1")
        wait(tau, "qe1")
        play("Y", "qe1")
        wait(tau, "qe1")
    wait(tau / 2, "qe1")
```

## Config

The system implements one control quantum element `qe1` and one readout quantum element `rr`

## Program

We nest an averaging loop and a delay loop which goes over `tau` values. 
```python
with for_(n, 0, n < NAVG, n + 1):
    with for_each_(tau, tau_vec):
        XY_n_sym(tau, 8)
        align("qe1", "rr")
        measure("readout","rr",None,
            demod.full("integW1", I),
            demod.full("integW2", Q),
        )
        save(I, out_str)
        with if_(I > th):
            save(s1, out_str)
        with else_():
            save(s0, out_str)
```
## Post processing

None

## Script


[download script](XY-n.py)


## References

<a id="1">[1]</a> Wang, Z. H., De Lange, G., Ristè, D., Hanson, R., & Dobrovitski, V. V. (2012). Comparison of dynamical decoupling protocols for a nitrogen-vacancy center in diamond. Physical Review B - Condensed Matter and Materials Physics, 85(15), 1–16. https://doi.org/10.1103/PhysRevB.85.155204

<a id="2">[2]</a> Ali Ahmed, M. A., Álvarez, G. A., & Suter, D. (2013). Robustness of dynamical decoupling sequences. Physical Review A - Atomic, Molecular, and Optical Physics, 87(4), 1–6. https://doi.org/10.1103/PhysRevA.87.042309
