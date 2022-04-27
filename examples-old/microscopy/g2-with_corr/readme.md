---
id: index
slug: ./
---

# Calculating g2 correlation on a beam splitter output

This folder contains files for simulation of g2 correlation calculation on an OPX system.
The folder contains the .py simulation and config files, and plots of simulated results.

## Experiment setting
A beam splitter's input is connected to an emitter, with it's two outputs connected to an OPX's two ADCs.

## Experiment objective
We wish to measure the correlation function between the two outputs of the beam splitter.

## Experiment method
Using the OPX time tag function, we obtain two time tag vectors, one for each ADC, e.g.
| 1   | 2   |
|-----|-----|
| 5   | 8   |
| 24  | 14  |
| 137 | 180 |
| 156 | 215 |

First, we calculate the histogram of arrivals for each port, i.e., 
the i'th bin will contain the number of arrivals between time (i-1)*bin_size and time i*bin_size -1.
The bin size is set in the script (only powers of 2 are used to facilitate faster calculations).
Note - times in the script are given in units of nano seconds as this is the unit interval of the OPX ADCs.
Next, we calculate the correlation function between the two histograms.
The script supports either one or two sided correlation, using the boolean variable fold (True=one sided, False=two sided)

The results are verified using the numpy correlation function.

## Simulation
To verify the code, we model the emitter output as a Bernouli(p) i.i.d. process. 
The outputs of the beam splitter is given by the emitter output multiplied by a 
Bernouli(0.5) process X and its complement to one (1-X).
The two outputs are fed to the ADC using the simulation loopback interface.

Note that the supplied graphs demonstrate a local minimum in 0, as is expected.


## More Information
