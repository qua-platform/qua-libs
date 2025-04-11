## How to generate the QUAM

Before starting to run experiments, it is necessary to build the Quantum Abstract Machine ([QUAM](https://github.com/qua-platform/quam)) for the desired
qubit chip architecture. More details about QUAM itself can be found in the [QUAM documentation](https://qua-platform.github.io/quam/).

The process can be divided into three sections:

1. First one needs to create the wiring specifying which quantum element (qubit, resonator, flux line...) is connected to
   which OPX channel.
2. Then, the QUAM components can be derived from the resulting wiring.json file.
   The attributes (methods or parameters) of these components can be customized at will.
3. Finally, a file called state.json, containing all the desired parameters that describe the full state of the system
   will be created.

### [1. Define the wiring](./configuration/make_wiring.py)

The wiring is generated from a python script called [make_wiring.py](./configuration/make_wiring.py).
It uses a tool called [wirer](https://github.com/qua-platform/py-qua-tools/tree/feature/auto_wiring/qualang_tools/wirer)
which helps to define the connectivity between the quantum elements (qubits, resonators, flux lines...) and the
channels of the control hardware (OPX+, Octave, MW-fem, LF-fem). It will create a json file called wiring.json which
contains the port mapping in a format requested by the QUAM builder, as well as the network settings.

1. First one needs to set up the instruments available in the QOP cluster

```python
from qualang_tools.wirer import Instruments

# Define static parameters
host_ip = "127.0.0.1"  # QOP IP address
cluster_name = "Cluster_1"  # Name of the cluster
# Desired location of wiring.json and state.json
# The folder must not contain other json files.
path = "./quam_state"

instruments = Instruments()
# instruments.add_opx_plus(controllers = [1])
# instruments.add_octave(indices = 1)
instruments.add_mw_fem(controller=1, slots=[1, 2])
instruments.add_lf_fem(controller=1, slots=[3, 4])
```

2. Then the port mapping between the different quantum elements and the available channels can be generated automatically
   based on the following hierarchy: resonators > qubit xy drive lines > qubit flux lines > qubit charge lines > tunable couplers.
   Some constrains can also be added in order to force all the resonators to be on the same line for instance.

```python
from qualang_tools.wirer import Connectivity, allocate_wiring
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec
# Define which qubits are present in the system
qubits = [1, 2, 3, 4, 5, 6]
# Allocate the wiring to the connectivity object based on the available instruments
connectivity = Connectivity()
# Define any custom/hardcoded channel addresses
q1_res_ch = mw_fem_spec(con=1, slot=1, in_port=1, out_port=1)
# Single feed-line for reading the resonators & individual qubit drive lines
connectivity.add_resonator_line(qubits=qubits, constraints=q1_res_ch)
connectivity.add_qubit_flux_lines(qubits=qubits)
connectivity.add_qubit_drive_lines(qubits=qubits)
allocate_wiring(connectivity, instruments)
```

3. Finally, the wiring and network information is serialized and store in wiring.json and the QUAM state is initiated in state.json.
   The wiring and port mapping can also be visualized in a matplotlib figure.

```python
from quam_builder.builder.machine import build_quam_wiring
from qualang_tools.wirer import visualize
# Build the wiring and network into a QUAM machine and save it as "wiring.json"
build_quam_wiring(connectivity, host_ip, cluster_name, path)

# View wiring schematic
visualize(connectivity.elements, available_channels=instruments.available_channels)
```

![opx1000_wiring](./.img/opx1000_wiring.PNG)

### [2. The QUAM components](./quam_builder.architecture)

Describe the structure of the QUAM components and how they can be customized, what to pay attention to...
Also, how to create a new one and update the wiring and build_quam accordingly.

The hierarchy and structure of QUAM can be detailed as follows:

1. [quam_root.py](./quam_builder.architecture/quam_root.py) represents the highest level in terms of hierarchy.
   It contains the qubits and qubit pairs objects, as well as the wiring, network and Octaves.
   Its methods are usually applied to all qubits or active qubits.
2. The definition of a QUAM transmon is defined in [transmon.py](./quam_builder.architecture/transmon.py). It contains the
   general transmon attributes (T1, T2, f_01...) as well as the QUAM components composing the transmon (`xy`, `resonator` and
   `z` in this case). Two-qubit gates can also be implemented by defining a specific qubit pair component as shown in
   [transmon_pair.py](./quam_builder.architecture/transmon_pair.py).
3. The QUAM components are either defined from the base QUAM components directly, such as the qubit xy drive which is
   directly defined as an `IQChannel`, or from user-defined components such as
   [readout_resonator](./quam_builder.architecture/readout_resonator.py) or [flux_line](./quam_builder.architecture/flux_line.py),
   which allows the customization of their attributes.

### [3. Generating the QUAM and state.json](./configuration/make_quam.py)

Once the QUAM root and the corresponding QUAM components are implemented, the QUAM state can be generated automatically and each
parameter of the QUAM components is initialized to its arbitrary default value.
All of these parameters can be updated programmatically based on the specs from the chip manufacturer for instance and
the process is described in the next section.

The script called [make_quam.py](./configuration/make_quam.py) takes care of generating the QUAM and filling the state
according to the wiring file. The Octaves connection parameters can also be edited here if relevant.

The QUAM generation happens in the `build_quam` function, which programmatically adds all the Octaves, ports, transmons and pulses
according to the wiring and the QUAM components. The default values used for the QUAM components and pulses can be found
under the [quam_builder](./quam_builder.builder) folder.

```python
from quam_config import Quam
from quam_builder.builder.machine import build_quam

path = "./quam_state"

machine = Quam.load(path)

# octave_settings = {"octave1": {"port": 11250} }  # externally configured: (11XXX where XXX are last three digits of oct ip)
# octave_settings = {"oct1": {"ip": "192.168.88.250"} }  # "internally" configured: use the local ip address of the Octave
octave_settings = {}

# Make the QUAM object and save it
quam = build_quam(machine, quam_state_path=path, octaves_settings=octave_settings)
```

Note that the set of default pulses, e.g. CosineDrag and Square, can be edited in [pulses.py](./quam_builder.builder/pulses.py).

For simplicity, or quick debugging/testing, the QUAM can also be generated "on-the-fly":

```python
import json
from qm import SimulationConfig
from qm.qua import program
from quam import QuamDict
from quam.components.ports import MWFEMAnalogOutputPort, MWFEMAnalogInputPort
from quam.components.channels import InOutMWChannel, MWChannel
from quam.components.pulses import SquarePulse, SquareReadoutPulse
from quam_config import Quam

machine = Quam()  # or, Quam.load() if the state already exists

# vvv  delete these if using Quam.load()
machine.network.host = "172.16.33.116"
machine.network.cluster_name = "Beta_8"
machine.wiring = QuamDict({})
# ^^^

mw_out = MWChannel(
    id="mw_out",
    operations={
        "cw": SquarePulse(amplitude=1, length=100),
        "readout": SquareReadoutPulse(amplitude=0.2, length=100), },
    opx_output=MWFEMAnalogOutputPort(
        controller_id="con1", fem_id=1, port_id=2, band=1, upconverter_frequency=int(3e9), full_scale_power_dbm=-14
    ),
    upconverter=1,
    intermediate_frequency=20e6
)
mw_in = InOutMWChannel(
    id="mw_in",
    operations={
        "readout": SquareReadoutPulse(amplitude=0.1, length=100), },
    opx_output=MWFEMAnalogOutputPort(
        controller_id="con1", fem_id=1, port_id=1, band=1, upconverter_frequency=int(3e9), full_scale_power_dbm=-14
    ),
    opx_input=MWFEMAnalogInputPort(
        controller_id="con1", fem_id=1, port_id=1, band=1, downconverter_frequency=int(3e9)
    ),
    upconverter=1,
    time_of_flight=28,
    intermediate_frequency=10e6
)

machine.qubits["dummy_out"] = mw_out
machine.qubits["dummy_in"] = mw_in

with program() as prog:
    mw_out.play("cw")
    mw_in.align()
    mw_in.play("readout")

config = machine.generate_config()
qmm = machine.connect()

simulation_config = SimulationConfig(duration=250)  # In clock cycles = 4ns
job = qmm.simulate(config, prog, simulation_config)
job.get_simulated_samples().con1.plot()

# save machine into state.json
machine.save("dummy_state.json")

# %%
# View the corresponding "raw-QUA" config
with open("dummy_qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)
```

### [4. Updating the parameters of state.json](./configuration/modify_quam.py)

Once the state is created, each parameter can be updated based on the desired initial values using
[modify_quam.py](./configuration/modify_quam.py).

```python
# %%
import numpy as np
import json
from quam_config import Quam
from quam_builder.builder.machine import save_machine

# Load QUAM
path = "./quam_state"
machine = Quam.load(path)

# %%
# Update the resonator parameters
rr_freq = np.array([4.395, 4.412, 4.521, 4.728, 4.915, 5.147]) * 1e9
rr_LO = 4.75e9
rr_if = rr_freq - rr_LO
rr_max_power_dBm = -8

for i, q in enumerate(machine.qubits):
    machine.qubits[q].resonator.opx_output.full_scale_power_dbm = rr_max_power_dBm
    machine.qubits[q].resonator.opx_output.upconverter_frequency = rr_LO
    machine.qubits[q].resonator.opx_input.downconverter_frequency = rr_LO
    machine.qubits[q].resonator.intermediate_frequency = rr_if[i]

# %%
# save into state.json
save_machine(machine, path)

# %%
# View the corresponding "raw-QUA" config
with open("qua_config.json", "w+") as f:
    json.dump(machine.generate_config(), f, indent=4)
```

Note that these parameters serve as a starting point before starting to calibrate the chip and their values will be
updated at the end of each calibration node.

### quam_builder

This folder contains all the utility functions necessary to create the wiring or build the QUAM, as well as QUA macros and data processing tools:

- [components](./quam_builder.architecture): this is where the QUAM root and custom QUAM components are defined. A set of basic QUAM components are already present, but advanced user can easily modify them or create new ones.
- [lib](./quam_libs): contains several utility functions for saving, fitting and post-processing data.
- [quam_builder](./quam_builder.builder): contains the main functions called in [machine.py](./quam_builder.builder/machine.py) and used to generate the wiring and build the QUAM structure from it and the QUAM components declared in the [components](./quam_builder.architecture) folder. It also contains the [pulses.py](./quam_builder.builder/pulses.py) file where the default qubits pulses are defined.

### configuration

The configuration folder contains the python scripts used to build the QUAM before starting the experiments.
It contains three files whose working principles are explained in more details below:

- **make_wiring**: create the port mapping between the control hardware (OPX+, Octave, OPX1000 LF fem, MW fem) and the quantum elements (qubits, resonators, flux lines...).
  - [make_wiring_lffem_mwfem.py](./configuration/make_wiring_lffem_mwfem.py) for a cluster made of LF and MW FEMs (OPX1000).
  - [make_wiring_lffem_octave.py](./configuration/make_wiring_lffem_octave.py) for a cluster made of LF-FEMs and Octaves (OPX1000).
  - [make_wiring_opxp_octave.py](./configuration/make_wiring_opxp_octave.py) for a cluster made of OPX+ and Octaves.
- [make_quam.py](./configuration/make_quam.py): create the state of the system based on the generated wiring and QUAM components and containing all the information necessary to calibrate the chip and run experiments. This state is used to generate the OPX configuration.
- [modify_quam.py](./configuration/modify_quam.py): update the parameters of the state programmatically based on defaults values (previous calibration, chip manufacturer specification...).
