---
id: index
title: IQ Mixer Calibration - From theory to practice
sidebar_label: IQ Mixer Calibration
slug: ./
---

## IQ Mixer mathematical model

### Up-conversation

Suppose we have a local oscillator (LO) with a frequency of $\Omega$, it can be described as:

$$
A_{LO}(t) = \text{Re}\left\{A_0 e^{i \Omega t} \right\}
$$

When we pass it through an ideal IQ mixer, the output will be:

$$
A_{RF}(t) = \text{Re}\left\{z(t) A_0 e^{i \Omega t} \right\}
$$

With $z(t)$ is defined according to the inputs at the I & Q ports according to:

$$
z(t) = z_I(t) + i z_Q(t)
$$