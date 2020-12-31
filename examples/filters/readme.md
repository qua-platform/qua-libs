---
id: index
title: Filtering
sidebar_label: Filters
slug: ./
---

This folder contains various examples for applying filters on the data, both for real time processing and for saving the
data.

## filters_with_sp

Using the StreamProcessor one can apply any IIR or FIR filter by convoluting the data with the filter's impulse response.
In this example we are applying two RF sources at 5.5 MHz and 15 MHz to the input, and using a 5th order Butterworth LPF filter in order to filterout the 15 MHz signal.
The filter is created using the scipy package in python, more filters can be found here:
https://docs.scipy.org/doc/scipy/reference/signal.html

Note that we specifically choose two frequencies at close frequencies in order showcase the strength of this approach.
We also use the built-in fft ability of the StreamProcessor in order to show the frequency response.

## filters_with_windows

In this example we showcase how to use the builtin 'moving window' filter in order to apply a rectangular window (also known as the boxcar or Dirichlet window) directly on the incoming data.
More information about the moving window can be found here:
https://qm-docs.s3.amazonaws.com/v0.7/python/features.html#measure-statement-features

This method is very efficient and can be used for realtime processing and feedback

### DC Filtering
Using "integration.moving_window" can be used for directly applying an LPF on the data, with an integration window of length (4*chunk_size) ns. This creates an LPF filter at a frequency of $$\frac{f_s}{4C}$$ where $$C$$ is the chuck_size.
This also decimates the data with decimation of (4*chunk_size).
It also scales the data by (4*chunk_size) (due to the integration).
In the example given here, we are applying two RF sources at 20 MHz and 100 kHz to the input, and using 5 MHz filter (C = 50) to remove the 20 MHz.

### IF Filtering
Using "demod.moving_window" can be used for extracting the baseband from an incoming modulated RF.
The signal is multiplied by a cosine and/or a sine (according to the integration weights). This shifts the IF to be around DC and to $$2 f_{IF}$$. The data is then being LPF filtered as above.
with an integration window of length (4*chunk_size) ns. This creates an LPF filter at a frequency of $$\frac{f_s}{4C}$$ where $$C$$ is the chuck_size.
This also decimates the data with decimation of (4*chunk_size).
It also scales the data by (2*chunk_size) (due to the integration).
In the example given here, we are applying two RF sources at 20 MHz and 20.1 MHz to the input. This creates a beating with an envelope of 100 kHz. We are a using 5 MHz filter (C = 50) to filter the remaining 40 MHz oscillation.