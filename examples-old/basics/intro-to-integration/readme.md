---
title: Intro to integration
sidebar_label: Intro to integration
slug: ./
id: index
---

This script demonstrates usage of the `integration` operator in the `measure` statement.
Integration collects measurement samples into a variable (or a vector of variables) by performing a
**weighted sum**. This means you need to specify the weights vector in the configuration
dictionary to use it. The details of how this feature is implemented can be found in the QUA
documentation, but this script shows a working usage example.

To showcase usage, we play a ramp pulse and use the `LoopbackInterface` to feed that signal 
back into the input of the OPX. This signal is shown below

![raw_out](raw_out.png "raw_out")

There are then several options for integration. The most basic one is to integrate the 
entire signal into a single point. A more involved example, is to use `sliced` or `accumulated` 
integration. In these, you either integrate is slices and generate an array of individual points, 
or do the same but sum the points one by one (which is essentially integration using the trapeze rule
when using uniform integration weights). 
Sliced integration follows the signal we play (but with fewer points) and acuumulated integration (in the case of a ramp)
produces a quadratic signal. Both cases are shown below

![slice_and_acc](slice_and_acc.png "slice_and_acc")

Finally, there is a `moving_window` option, were you specify a window composed of chunks.
Windows overlap, as shown below. This, in effect, implements a fairly flexible filter. 

![mov_win_scheme](mov_win_scheme.png "mov_win_scheme")

The resulting output is shown below

![mov_win](mov_win.png "mov_win")

