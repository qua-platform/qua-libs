# Intro to octave
This folder contains scripts for checking the octave and integrating it into your running experiments. 

**Please note that the following files are compatible with QOP222 and qm-qua==1.1.5 and newer.** 
**If using QOP and qm-qua version please follow this folder [qm-qua 1.1.4 and below](https://github.com/qua-platform/qua-libs/tree/main/Tutorials/intro-to-octave/qm-qua%201.1.4%20and%20below)**

### You can find here five python files 
1. `octave_introduction.py` contains the basic octave commands. This file shows the configuration the octave, sets the octave's clock, test the RF outputs and calibrate the mixers. 
2. `hello_octave.py` shows an example of running a program with the octave.
3. `configuration.py` is the configuration file for `hello_octave.py`.
4. `octave_calibration.py` is the octave configuration file for setting the clock and calibration.

## [octave_introduction.py](octave_introduction.py)
In this file you can find the basic octave commands.
It contains the configuration and its parameters, adds information about the octave and runs octave commands. 
It is organized in the following sections:
   1. Configuring the clock
   2. Looking at the octave RF outputs
   3. Configuring the digital switches
   4. Configuring the down converter modules
   5. Calibration

Here is how the OPX+, Octave and spectrum analyzer should be configured for each step:
![OPX+/Octave connectivity schematic](octave_introduction.png)
        
## [configuration.py](configuration.py) 
This file contains the config dictionary that the `hello_octave.py` file uses.

When initializing an Octave, the following parameters should be provided:
* __name__: the name of the Octave.
* __octave_calibration_db_path__: the path to the octave mixer calibration database. 
  * By default, it is set to the current working directory.

Note that while using your configuration file, there are three things to pay attention to:
   1. In the `elements` section, there is no `mixInputs` key, and `outputs` key. Instead, there are `RF_inputs` key and `RF_outputs` key correspondingly.
   2. There is a new key named `octaves`, where all the up-converter and down-converter parameters are configured. The connectivity between the OPX and the Octave is configured there as well. 
   3. There is no need for the `mixers` section. 

## [octave_calibration.py](octave_calibration.py)
This file is used to parametrize the Octave's clock and to calibrate the octave.

__This file must be run once prior to executing any QUA program in order to configure the Octave's clock and calibrate if needed.__ 

## [hello_octave.py](hello_octave.py) 
This file shows an example of how to run a program with a setup containing a set of OPX + Octave. 
The Octave config defined in ``configuration.py`` is passed to the Quantum Machine Manager to enable the communication with the previously declared Octave unit.

Then one can run the QUA program as usual assuming that the Octave has been calibrated by running `octave_calibration.py`.
