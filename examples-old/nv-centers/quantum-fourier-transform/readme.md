---
id: index
title: Quantum Fourier Transform
sidebar_label: Quantum Fourier Transform
slug: ./
---

This code implements a simple QFT protocol sensing protocol as described in [[1]](#1).
It assumes that all the pulses are implemented with optimal control theory, which in practice means that they are arbitrary pulses. For the sake of simplicity, in this code, they are all constants. 

## Setup
![QFT](setup.png "QFT")

## Script
[download script](QFT.py)

## References

<a id="1">[1]</a> V. Vorobyov, et al., Quantum Fourier transform for quantum sensing, arXiv:2008.09716. <br />
<a href="https://arxiv.org/abs/2008.09716">https://arxiv.org/abs/2008.09716</a> <br />