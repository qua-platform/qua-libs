---
id: index
title: IQ Mixer Calibration - From theory to practice
sidebar_label: IQ Mixer Calibration
slug: ./
---

## IQ Mixer mathematical model

### Up-conversation

#### Ideal Mixer

Suppose we have a local oscillator (LO) with a frequency of $\Omega$, it can be described as:

$$
A_{LO}(t) = \text{Re}\left\{A_0 e^{-i \Omega t} \right\}
$$

When we pass it through an ideal IQ mixer, the output will be:

$$
A_{RF}(t) = \text{Re}\left\{z(t) A_0 e^{-i \Omega t} \right\}
$$

With $z(t)$ is defined according to the inputs at the I & Q ports according to:

$$
z(t) = z_I(t) + i z_Q(t)
$$

------------------------------------------------------------
> **_Example:_** : 
> It is common have $z_I(t) = \text{cos} \left( \omega_{IF} t + \phi \right)$ and $z_Q(t) = -\text{sin}\left(\omega_{IF}t + \phi \right)$.
> This makes $z(t) = e^{-i (\omega_{IF}t + \phi)}$ and:
>
> $$
> A_{RF}(t) = \text{Re}\left\{A_0 e^{-i (\Omega + \omega_{IF}) t - \phi} \right\}
> $$
>
> As can be seen, applying a Cosine & Sine at the I & Q ports will shift the frequency of the LO and add a phase.

----------------------------------------------------------------
Rewriting the equation, and assuming $A_0=1$, we get write:

$$
A_{RF}(t) = \frac{1}{2} \left(z(t) e^{-i \Omega t} + z^*(t) e^{i \Omega t}\right) 
$$


In the frequency domain we can write it in matrix form:

$$
\begin{pmatrix}
a[\omega] \\
a^*[-\omega] 
\end{pmatrix} 
=
\frac{1}{2}
\begin{pmatrix}
1 & 0 \\
0 & 1 
\end{pmatrix}
\begin{pmatrix}
z[\omega+\Omega] \\
z^*[\omega-\Omega] 
\end{pmatrix}
$$

When we defined $A_{RF}[\omega] = a[\omega] + a^*[-\omega]$.
Note that generally speaking, this creates two sidebands at the two sides of $\Omega$. 
We will treat the upper sideband as the signal, and the lower as the image, which can be removed by a proper choice of $z(t)$.

------------------------------------------------------------
> **_Example:_** : 
> Looking back at the previous example: 
> 
> $$
> z(t) = e^{-i \omega_{IF}t} \rightarrow z[\omega] = \delta[\omega+\omega_{IF}]
> $$
> $$
> z^*(t) = e^{i \omega_{IF}t} \rightarrow z^*[\omega] = \delta[\omega - \omega_{IF}]
> $$
> $$
> A_{RF}[\omega] = \frac{1}{2} \left(\delta[\omega+(\Omega+\omega_{IF})] + \delta[\omega-(\Omega+\omega_{IF})]\right)
> $$ 
> $$
> A_{RF}(t) = \text{cos} \left((\Omega+\omega_{IF})t\right)
> $$
> The choice above removed the image sideband and kept only the signal sideband.

----------------------------------------------------------------

#### Non-ideal Mixer
The math we described above arise from the mixing of two branches:

1. The signal at the I port is multiplied by the cosine of the LO
2. The signal at the Q port is multiplied by the sine of the LO

In an ideal mixer, it is assumed that these two branches are identical both in amplitude and in phase. When the mixer is not ideal this can be modeled as:

$$
A_{RF}(t) = \text{Re}\left\{z(t) A_0 \left[\text{cos}(\Omega t) + i r_{up} \text{sin}(\Omega t+\phi_{up}) \right] \right\}
$$

Where $r_{up}$ and $\phi_{up}$ are the relative amplitude and phase mismatch between the two branches. Note that $r_{up}=1$ and $\phi_{up}=0$ restore the ideal mixer equation.

In addition to the branches' imbalance, non-ideal mixers also have LO leakage which can be modeled as:

$$
A_{RF}(t) = \text{Re}\left\{z(t) A_0 \left[\text{cos}(\Omega t) + i r_{up} \text{sin}(\Omega t+\phi_{up}) \right] + \epsilon_{up} A_0 e^{-i \Omega t} \right\}
$$

In the frequency domain, this takes the form of:

$$
\begin{pmatrix}
a[\omega] \\
a^*[-\omega] 
\end{pmatrix} 
=
\frac{1}{4}
\begin{pmatrix}
d^*_{up} & o_{up} \\
o^*_{up} & d_{up} 
\end{pmatrix}
\begin{pmatrix}
z[\omega+\Omega] \\
z^*[\omega-\Omega] 
\end{pmatrix}
+
\frac{\epsilon_{up}}{2}
\begin{pmatrix}
\delta[\omega + \Omega] \\
\delta[\omega - \Omega] 
\end{pmatrix}
$$

With $d_{up} = 1 + r_{up} e^ {-i \phi_{up}}$ and $o_{up} = 1 - r_{up} e^ {i \phi_{up}}$.

Note that the non-ideal mixer will have leakage terms at $\Omega$ and at the image sideband. 
Adding a constant term to $z(t)$ can cancel the LO leakage term and applying the appropriate gain and phase offsets to the I & Q channels can remove the image term.