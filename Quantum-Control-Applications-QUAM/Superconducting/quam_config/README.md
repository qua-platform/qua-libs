# Creating the QUAM State

This document explains the process of defining, generating, and initializing the QUantum Abstract Machine (QUAM) state, which serves as the central configuration object for your quantum system within the Qualibrate software ecosystem. The QUAM object holds information about hardware configuration, connectivity, elements (qubits, resonators, etc.), pulses, and operations.

## Folder Contents (`quam_config/`)

The `quam_config` library is organized into the following main files and directories:

```text
quam_config/
├── __init__.py             # Makes quam_config a Python package.
├── my_quam.py              # Defines the core QUAM Python class structure.
├── build_quam_wiring.py            # Script to generate the QUAM state file (e.g., JSON).
├── build_quam_wiring_onthefly.py   # Script for dynamic QUAM generation (inferred).
├── populate_quam_state_*.py    # Scripts to populate QUAM with initial parameters for specific hardware.
├── instrument_limits.py    # Defines operational limits for instruments.
├── wiring_examples/        # Example wiring/connectivity configurations.
│   ├── wiring_opxp_octave.py
│   ├── wiring_lffem_mwfem.py
│   └── ...                 # Other specific hardware wiring examples.
└── README.md               # Documentation for QUAM configuration (this file or original).
```

**`my_quam.py`**: Defines the root-level QUAM class (inheriting from base `QuamRoot`) representing your physical setup (qubits, instruments). This will be used in each \`QualibrationNode\`. Customize to match your lab.

**`build_quam_wiring.py`**:
Instantiates the structure from `my_quam.py` and generates the baseline QUAM state file (e.g., JSON) with default values.

**`build_quam_wiring_onthefly.py`**:
Alternative script for generating QUAM configurations dynamically at runtime.

**`populate_quam_state_*.py`** (e.g., `populate_quam_state_opxp_octave.py`, `populate_quam_state_lf_mw_fems.py`):
Loads a base QUAM state and populates it with initial parameters and connectivity details for specific hardware setups (e.g., OPX+Octave), providing a better starting point for calibration.

**`instrument_limits.py`**:
Defines instrument operational limits (e.g., power, frequency) used for validation or constraints during the analysis and subsequent QUAK state updates of experiments.

**`wiring_examples/`**:
Contains example scripts showing how to define hardware wiring/connectivity within QUAM for various setups, serving as templates.

## Workflow for Creating the QUAM State

The typical workflow to create a QUAM state that accurately represents your quantum setup involves the following steps:

### 1️⃣ Navigate to `Superconducting/quam_config` folder

All relevant scripts reside here.

### 2️⃣ Define the QUAM Root Class Structure (in `my_quam.py`)

Edit `my_quam.py` to define the Python classes representing your system's hierarchy. This involves specifying the types and number of components like qubits (including whether they are flux-tunable or fixed-frequency), resonators, instruments (OPX controllers, Octaves, FEMs), etc. This class should inherit from a base `QuamRoot`.

### 3️⃣ Generate Static Configuration & Wiring (using `build_quam_wiring.py`)

This step creates the static part of the QUAM state, primarily defining the hardware layout and connectivity.

- The `build_quam_wiring.py` script typically utilizes the external `quam-builder` Python package. This package provides helpful tools, including base QUAM classes and tools to simplify the process of programmatically defining your hardware structure and generating the initial configuration files. For detailed documentation on `quam-builder`, refer to its repository: [https://github.com/qua-platform/quam-builder](https://github.com/qua-platform/quam-builder).
- It is recommended to copy the contents from a relevant template in the `wiring_examples/` folder (matching your hardware) into `build_quam_wiring.py` and adjust details like the number of qubits, specific instrument connections (FEM ports to qubit channels), and network information (IP address, cluster name).
- Running `python build_quam_wiring.py` uses this wiring definition to generate two files in your QUAM state folder (default: `Superconducting/quam_state`):
  - `wiring.json`: Contains the static hardware connectivity information based on the script.
  - `state.json`: A skeleton file based on the structure in `my_quam.py`, but largely empty of specific operational parameters (frequencies, pulse details, etc.).
- _(Alternative: For direct, hardcoded QUAM structure creation without the builder, modify `build_quam_wiring_onthefly.py` instead)_.

### 4️⃣ Initialize Dynamic Parameters (using `populate_quam_state_*.py`)

This step populates the `state.json` file with initial operational parameters.

- Choose the correct initialization script based on your hardware:
  - For OPX+/Octave setups: Use `populate_quam_state_opxp_octave.py`.
  - For OPX1000/FEM setups: Use `populate_quam_state_lf_mw_fems.py`.
- Edit the chosen script: These files contain hardcoded initial guesses for parameters like qubit/resonator frequencies, pulse amplitudes/durations, gains, etc. **You must adjust these values** to be reasonable starting points for your specific qubits and setup.
- Run the script: Execute `python populate_quam_state_{hw_type}.py` (replacing `{hw_type}` accordingly). This loads the existing `state.json` and `wiring.json`, populates the dynamic parameters based on the script's logic and hardcoded values, and saves the updated, populated `state.json`.

### 5️⃣ (Optional) Specify Instrument Limits (using `instrument_limits.py`)

For device safety reasons, it may be necessary to impose certain limits on parameters such as waveform amplitudes. These can be set in the file `instrument_limits.py`. This step is optional, but it ensures that relevant parameters are not set above any defined limits.

## Saving and loading a QUAM state

After completing these steps, your QUAM state is ready. You can load and save it within your Python scripts or calibration nodes using:

```python
# Make sure QuamRoot is correctly imported from your my_quam.py
from quam_config.my_quam import QuamRoot

# Load QUAM state (assuming QUAM_STATE_PATH environment variable is set or using default)
machine = QuamRoot.load()

# Save QUAM state (updates the state.json file)
machine.save()
```

This populated QUAM state serves as the starting point for running calibrations via Qualibrate nodes, which will further refine these parameters.
