# Intro to octave
This folder contains scripts for checking the octave and integrating it into your running experiments.

**Please note that the following files are compatible with QOP202 and qm-qua==1.1.4 and below.**

### You can find here five python files
1. `octave_introduction.py` contains the basic octave commands. This file is a step-by-step guide on how to configure the Octave, test the RF outputs and calibrate the mixers.
2. `hello_octave.py` shows an example of running a program with the octave.
3. `configuration.py` is the configuration file for `hello_octave.py`.
4. `octave_configuration.py` is the octave configuration file for `hello_octave.py` where the octave parameters are set.
5. `set_octave.py` contains helper functions to set all octave parameters.

## [octave_introduction.py](octave_introduction.py)
In this file you can find the basic octave commands.
It contains the configuration and its parameters, adds information about the octave and runs octave commands.
It is organized in the following sections:
   1. Configuring the clock
   2. Configuring the lo source, lo frequency, rf output gain and rf output mode
   3. Looking at the octave RF outputs
   4. Configuring the digital switches
   5. Configuring the down converter modules
   6. Calibration

## [configuration.py](configuration.py)
This file contains the config dictionary that the `hello_octave.py` file uses.
It is also where the Octave units are declared using the ``OctaveUnit`` class.

When initializing a new Octave unit, a few parameters can be specified:
* __name__: the name of the Octave unit.
* __ip__: the IP address of the Octave or the router connected to it. It is usually the same as the one to connect to the OPX.
* __port__: the Octave port. Default is 50.
* __clock__: the clock setting that can be "Internal", "External_10MHz", "External_100MHz" or "External_1000MHz". Default is "Internal".
* __con__: the OPX controller connected to this Octave. Only used with default mapping and default is "con1".
* __port_mapping__: custom mapping between the Octave and the OPX. Default is "default". The expected structure is the following:
```python
port_mapping = [
    {
        ("con1", 1): ("octave1", "I1"),
        ("con1", 2): ("octave1", "Q1"),
        ("con1", 3): ("octave1", "I2"),
        ("con1", 4): ("octave1", "Q2"),
        ("con1", 5): ("octave1", "I3"),
        ("con1", 6): ("octave1", "Q3"),
        ("con1", 7): ("octave1", "I4"),
        ("con1", 8): ("octave1", "Q4"),
        ("con1", 9): ("octave1", "I5"),
        ("con1", 10): ("octave1", "Q5"),
    }
]
```
Then the list of the Octave units needs to be passed to the ``octave_declaration`` function.

Note that while using your configuration file, there are two things to pay attention to:
   1. In the `elements` section, in `mixInputs` key, the mixer name should have a specific structure, as can be seen in this file.
   2. In the `mixers` section, the keys should correspond to the mixers names given in the `elements` section.

## [octave_configuration.py](octave_configuration.py)
This file is used to parametrize the Octave synthesizers and up- and down- conversion modules linked to the elements
that will be used in the QUA programs.

This can be done be declaring the elements connected to the Octave with the ``ElementsSettings`` class and several parameters can be set:
* __name__: the name of the element as defined in the config.
* __lo_source__: the LO source that can be set to "Internal" if using the internal synthesizer, or "LO1", "LO2", "LO3" "LO4" or "LO5" if using an external LO connected to the back of the Octave. Default is "Internal".
* __gain__: the RF port gain within [-10 : 0.5: 20] dB. Default is 0 dB.
* __switch_mode__: the mode of the RF switch at the Octave RF ports. Can be "on", "off", "trig_normal" or "trig_inverse". Default is "on".
* __rf_in_port__: RF input port of the Octave if the element is used to measure signals from the fridge. The syntax is [octave_name, RF input port] as in ["octave1", 1].
* __down_convert_LO_source__: LO source for the down-conversion mixer if the element is used to measure signals from the fridge. Can be "Internal", "Dmd1LO" or "Dmd2LO".
* __if_mode__: Specify the IF down-conversion mode. Can be "direct" for standard down-conversion, "envelope" for reading the signals from the envelope detector inside the Octave, or "mixer" to up-convert the down-converted signals using the IF port in the back of the Octave.

Then a Quantum Machine Manager is opened and the element list is passed to the ``octave_settings`` function to configure the Octave.
This function is also used to calibrate all the specified elements at the (LO, IF) pairs defined in the config.

Note that the Octave Synthesizer frequencies are defined in the config in the mixInput element section.

__This file must be run once prior to executing any QUA program in order to configure the Octave.__

## [hello_octave.py](hello_octave.py)
This file shows an example of how to run a program with a setup containing a set of OPX + Octaves.
The Octave config defined in ``configuration.py`` is passed to the Quantum Machine Manager to enable the communication with the previously declared Octave units.

The one can run the QUA program as usual assuming that the Octave parameters have been set by running `octave_configuration.py`.


## [set_octave.py](set_octave.py)
This file contains all the relevant octave commands and should make it easier for you to integrate the Octave in your running experiments!

Let's talk about each function separately:

1. `OctaveUnit` class:
   * Define the name, IP address, port, clock setting and port mapping of each Octave in the cluster.

2. `ElementsSettings` class:
   * Define the LO source, gain, RF switch mode and down-conversion of each element connected to the Octaves.

3. `octave_declaration` function:
   1. Creates a `calibration_db.json` file where the calibration parameters will be updated
   2. Adds the octave devices to the octave_config object
   3. Sets the port mapping for each OPX-octave pair


4. `octave_settings` function:
   1. Sets the octave's clock in.
      * Note: The Octave's clock out is fixed to 1GHz.
   2. Sets the up-converters modules.
      * Note: the 4 options for the trigger are: "on", "off", "trig_normal" and "trig_inverse".
   3. Sets the down-converters modules
      * Note: If using down-converter module 2 don't forget to connect the LO to Dmd2LO in the back panel
   4. Calibration:
      * Note: It is set to True by default
