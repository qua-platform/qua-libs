# Qubit Frequency Tracking

_Written by: Niv Drucker_

_Demonstrated in: the Lab of Prof. David Schuster in the University of Chicago._

_The experiment of: Ankur Agrawal._

_Important note: This is the exact code that was used for running the qubit frequency tracking measurement, and the code is tailored for a very specific setup and SW environment. Thus, the code is only for insipiration._

## The goal
The goal of this measurement is to track the frequency fluctuations of the transmon qubit, and update the frequency of the qubit element accordingly using a closed-loop feedback. This should enable us to stay in the reference frame of the qubit. More precisecly, our goal is to calibrate a frequency-tracking-macro (the two-point-Ramsey macro) that can be interleaved in a general experiment\routine, and correct actively for the frequency fluctuations.
 
## The device
The device consist of a single Transmon qubit coupled to a multimode 3D resonator. However, during the all experiment the 3D resonator is ideally in the vacuum state.


## Methods and results

The calibration of the macro consist of three steps -
1) **Time-domain Ramsey.**
In this step we are using the method `time_domain_ramesy_full_sweep(self, reps, f_ref, tau_min, tau_max, dtau, stream_name, correct=False)` to perform a Ramsey mesurement in the time domain. The probablity to measure the qubit in the excited state (as a function of tau) is:

<img src="https://latex.codecogs.com/svg.image?\mathcal{P}(e)\sim&space;A\exp\left(-\frac{\tau}{T_{2}}\right)\left(\frac{1&plus;\cos\text{(2\ensuremath{\pi}\ensuremath{\Delta}\ensuremath{\tau}&plus;\ensuremath{\phi})}}{2}\right)&space;" />

The parameter Delta is determining the oscillations frequency, and it is the shift of the drive from the real resonance frquency of the qubit (assuming the frequency of the qubit is constant during the measurement). Since we are introducing a shift of `f_ref` WRT to resonance, we expect the first peak to be found at `1/f_ref` (red star in the plot below). In practice, we get the red star in `1/Delta = 1/(fref-drift)' due to a drift in the resonance frequency of the qubit before we started the experiment.


![td_ramsey0](td_ramsey0.png)

The raw data is in blue and the purple curve is the fit. For the analysis we used the mathod `time_domain_ramesy_full_sweep_analysis(self, result_handles, stream_name)`.
Note that the fit method `_fit_ramsey` is slightly different from the equation above, so to enable better fitting. 
From the fit we extract T2, the new resonance frequency of the qubit and phi.
In order to verify that frequency calibration worked properly we are running the TD Ramsey again and check that now we get the red dot on the first peak:

![td_ramsey_corrected.png](td_ramsey_corrected.png)


2) Frequency-domain Ramsey measurement
some text3
![fd_ramsey.png](fd_ramsey.png)
3) Active tracking and correction based on two frequency points only in the frequency-domain Ramsey.  
some text 4
![active_frequency_tracking.PNG](active_frequency_tracking.PNG)