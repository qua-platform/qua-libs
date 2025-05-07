# Superconducting QUAlibration graphs

This repository provides a comprehensive library for calibrating superconducting transmon qubits using the Quantum Orchestration Platform (QOP), QUAM, and QUAlibrate.
This includes both flux-tunable and fixed-frequency Transmons.
It includes configurable experiment nodes, analysis routines, and tools for managing the quantum system state (QUAM).

This library is built upon **QUAlibrate**, an advanced, open-source software framework designed specifically for the automated calibration of Quantum Processing Units (QPUs). QUAlibrate provides tools to create, manage, and execute calibration routines efficiently. The configurable experiment nodes, analysis routines, and state management tools included here are designed to integrate seamlessly with the QUAlibrate ecosystem.
See the [QUAlibrate Documentation](https://qua-platform.github.io/qualibrate/) for more information.

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Getting Started](#getting-started)
    - [Downloading the Library](#downloading-the-library)
    - [Installation](#installation)
    - [Initial Setup (QUAlibrate Configuration)](#initial-setup-qualibrate-configuration)
    - [Verify Setup](#verify-setup)
3.  [Creating the QUAM State](#creating-the-quam-state)
4.  [Calibration Nodes and Graphs](calibration-nodes-and-graphs)
5.  [Project Structure](#project-structure)
6.  [Extending QUAM Components](#extending-quam-components)
7.  [Contributing](#contributing)
8.  [License](#license)

## Prerequisites

- **Python:** Version 3.9 to 3.12 is supported.
- **Python Virtual Environment:** Strongly recommended to avoid dependency conflicts.
  You can create one using:

  - **venv**: Python's built-in virtual environment handler
    ```bash
    python -m venv .venv
    source .venv/bin/activate ` (Linux/macOS)
    .venv\Scripts\activate` (Windows)
    ```
  - **conda**: Anaconda's virtual environment manager
    ```bash
    conda create -n qualibrate_env python=3.10
    conda activate qualibrate_env
    ```

- **Git:** (Optional but Recommended) For version control, easier updates (pulling changes), and collaboration (forking and contributing).
  [Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
- **Access to Quantum Orchestration Platform (QOP) hardware:** Required for running experiments on hardware.

## Getting Started

### Downloading the Library

You have a few options to get the code:

1.  **Customer Repository:** If provided as part of a customer installation, use the dedicated user repository.
2.  **Fork (Recommended for Staying Updated):** Forking the `qua-libs` repository on GitHub to your account (see GitHub's guide on [how to fork a repo](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo)) and then cloning your fork is the recommended way to stay periodically in sync with updates from the main repository.
    It also allows you to contribute changes back via pull requests.
3.  **Git Clone (Direct):** Clone the repository directly using Git.
    This allows you to pull updates but requires managing potential merge conflicts manually if you make local changes without forking.
    ```bash
    git clone https://github.com/qua-platform/qua-libs.git
    ```
4.  **Direct Download:** Navigate to the `qua-libs` repository on GitHub, download the ZIP file, and unzip it.
    This method doesn't require Git but makes updating and contributing harder.

### Installation

Once you have the code locally:

1.  **Navigate to the Directory:** Open a terminal or command prompt and change into the `superconducting` directory within the downloaded/cloned repository.
2.  **Activate Virtual Environment:** Ensure your dedicated Python virtual environment (see [Prerequisites](#prerequisites)) is activated.
3.  **Install the Package:** Run the following command to install the library and its dependencies in editable mode (`-e`), which means changes you make to the source code will be reflected immediately without reinstalling:

    ```bash
    pip install -e .
    ```

    _Note for `uv` users:_ If you are using `uv` instead of `pip`, you might need to allow pre-releases depending on the dependencies:

    ```bash
    uv pip install -e . --prerelease=allow
    ```

### Initial Setup (QUAlibrate Configuration)

The QUAlibrate framework needs some initial configuration to know where to find calibration scripts, store data, and manage the system state (QUAM).

1.  **Run the Configuration Script:** Execute the provided script from within the `Superconducting` directory:

    ```bash
    setup-qualibrate-config
    ```

    If this command does not work, you may need to first restart your terminal or IDE.

2.  **Follow Prompts:** The script will interactively ask for the following details:

    - `project name`: A unique name for your project or QPU chip (e.g., `MyQPU_Chip1`).  
      Default: `QPU_project`.
    - `storage location`: The root directory where measurement data will be saved.  
      Default: `data/{project_name}` relative to the current directory.
    - `calibration library folder`: The path to the directory containing calibration nodes/graphs.  
      Default: `./calibrations` relative to the current directory.
    - `QUAM state path`: The location where the QUAM state file (containing system parameters, connectivity, etc.) is stored.  
      Default: `./quam_state` relative to the current directory.

          You can press `Enter` or type `y` to accept the defaults, or `n` to provide custom paths.

3.  **Confirm Full Config:** The script will show the complete QUAlibrate configuration for final confirmation.
    For detailed explanations of all settings, refer to the [QUAlibrate Configuration File Documentation](https://qua-platform.github.io/qualibrate/configuration/).

### Verify Setup

To ensure QUAlibrate is installed and configured correctly:

1.  **Launch the Web Interface:** Run the following command in your terminal:

    ```bash
    qualibrate start
    ```

2.  **Open in Browser:** Navigate to [http://127.0.0.1:8001](http://127.0.0.1:8001).

You should see the QUAlibrate web UI, listing the calibration nodes found in your configured `calibrations` directory.

## Creating the QUAM State

QUAM (Quantum Abstract Machine) provides an abstraction layer over the low-level QUA configuration. It allows you to define your quantum system (hardware, connectivity, qubit parameters, pulses, etc.) in a structured, physicist-friendly way. The QUAM state is stored in the `./quam_state/` directory, separated into a static part `./quam_state/wiring.json` for the wiring and network, and the main contents in `./quam_state/state.json`. The QUAM state serves as a persistent digital model of your entire setup, one that is continuously updated with calibrations.

**Interaction with Calibration Nodes:**

- **Loading:** Calibration nodes (scripts in `calibrations/`) typically load the latest QUAM state at the beginning of their execution. This provides them with all the necessary parameters (e.g., frequencies, amplitudes, timings) required to run the specific calibration experiment.
- **Updating:** After a calibration node runs and analyzes the results, it often calculates updated parameters (e.g., a newly calibrated qubit frequency or an optimized pulse amplitude). The node then modifies the corresponding values within the loaded QUAM object.
- **Saving:** QUAlibrate nodes save the modified QUAM state, often alongside the experiment results. This ensures that subsequent nodes in a calibration graph or future runs use the most up-to-date, calibrated parameters. This also updates the latest QUAM state in the `./quam_state/` directory.

**How to Create the State:**

The process of creating the initial QUAM state file involves defining your specific hardware components (OPXs, Octaves, mixers, LOs), as well as the QPU layout that the hardware is attached to. Detailed instructions are found in **[quam_config/README.md](quam_config/README.md)**

This directory contains scripts (`generate_quam.py`, `populate_quam_xx.py`, examples, etc.) that demonstrate how to build the QUAM object programmatically.

## Calibration Nodes and Graphs

The scripts within the `calibrations` directory are the building blocks for automated calibration routines.
Each script typically performs a specific measurement (e.g., Resonator Spectroscopy, Rabi Oscillations, T1 measurement).
They are designed to be run via the QUAlibrate framework, either individually or as part of a larger calibration sequence (graph), but can also be executed as a standalone script from your favorite Python IDE (e.g. PyCharm, VScode...).

Refer to the [calibrations/README.md](calibrations/README.md) for detailed information on the structure and conventions used for these nodes.

## Project Structure

The library is organized into the following main directories:

TODO: modify after moving things to Qualibration-libs

```
superconducting/
├── calibrations/      # Individual calibration scripts (nodes) runnable by QUAlibrate.
│   ├── 00_hello_qua.py
│   ├── 01a_time_of_flight.py
│   └── ... (many calibration routines)
│
├── data/                   # Default location for storing experiment results.
│   └── {project_name}/     # Data organized by project name.
│       └── YYYY-MM-DD/     # Data organized by date.
│           └── #idx_{node_name}_HHMMSS/ # Data for a specific run.
│               └── quam_state/
│                   ├── state.json      # Contains the QUAM state except the wiring and network.
│                   └── wiring.json     # Contains the static part of the QUAM state (wiring and network).
│               ├── data.json       # Structure containing the data outpoutted by the node (fit results, figures,...).
│               ├── ds_raw.h5       # HDF5 dataset containing the raw data.
│               ├── ds_fit.h5       # HDF5 dataset containg the post-processed data.
│               ├── figures.png     # Generated figures.
│               └── node.json       # Metadat about the node used by QUAlibrate.
│
│── quam_state/         # Default location for the main QUAM state file.
│   ├── state.json      # Contains the QUAM state except the wiring and network
│   └── wiring.json     # Contains the static part of the QUAM state (wiring and network)
|
├── quam_config/            # Scripts and configurations for generating/managing QUAM state files.
│   ├── wiring_examples/    # Example configurations for different hardware setups.
│   ├── generate_quam.py        # Script to generate a QUAM file.
│   ├── populate_quam_xx.py     # Script to populate the newly generated QUAM file with initial values.
│   └── ...
│
│── calibration_utils/  # Specific experiment implementations (e.g., T1, Ramsey, Spectroscopy).
│   └── resonator_spectroscopy/
│   │   ├── analysis.py     # Contains all the analysis functions.
│   │   ├── parameters.py   # Contains node-specific parameters.
│   │   └── plotting.py     # Contains all the plotting functions.
│   └── ...
│
├── README.md               # This file.
└── pyproject.toml # Installation configuration for the package.
```

**calibrations**  
The `calibrations/` folder contains individual Python scripts, each representing a calibration "node".
These scripts typically import functionality from **calibration_utils**, define parameters, run a QUA program, analyze results, and update the QUAM state. See the README.md within this folder for more details on node structure.

**data**  
The `data/` folder is the default output directory where QUAlibrate saves results (plots, raw data, QUAM state snapshots) from calibration runs, organized by project, date, and run index/name.

**quam_state**  
The `quam_state/` directory is where the main QUAM state files are stored. These files are crucial for maintaining the current state of the quantum system, excluding the wiring and network configurations. The `state.json` file contains dynamic aspects of the QUAM state, while the `wiring.json` file holds static information about the system's wiring and network setup.

**quam_config**
Tools and examples for creating the quam_state.json file, which describes your specific hardware setup (instruments, connections, qubit parameters).

**calibration_utils**  
`calibration_utils/` contains the calibration-specific helper functions, such as specific fitting routines, parameter classes, and plotting functionality

## Extending QUAM Components

QUAM Builder provides a repository containing a standard set of components related to qubits, such as superconducting qubits (e.g., `FluxTunableTransmon`), resonators (e.g., `ReadoutResonatorIQ`), and associated pulses. While this provides a solid foundation, it should not be viewed as a fixed set. As you advance your calibration routines and develop custom calibration nodes and graphs, you may find it necessary to extend or modify these standard components.

There are several ways you might want to extend the QUAM components:

1.  **Adding Parameters:** You might need to add different parameters to the standard classes to accommodate specific characteristics of your hardware or calibration methods. For example, you may have a different coherence time metric you want to keep track of.
2.  **Adding Components:** You might want to introduce entirely new components. This could include custom pulse shapes tailored to your experiments or other quantum elements relevant to your setup.

### Method 1: Forking or Cloning QUAM Builder

One way to achieve these extensions is by creating a fork or a local clone of the main QUAM Builder repository.

1.  **Clone/Fork:** Obtain a local copy of the QUAM Builder source code.
2.  **Locate Components:** Navigate to the `architecture` folder within the repository. This folder contains the definitions for the different QUAM components.
3.  **Modify or Add:** You can now directly modify the existing Python classes for the components or add new Python files defining your custom components.

**Important Considerations:**

- **Compatibility:** When modifying existing components, be mindful of compatibility with existing calibration nodes. For example, if a calibration node expects a Transmon object to have a property named `T2echo`, renaming or removing this property in your modified class will break that node unless you also update the node's code to use the new property name. Try to maintain backward compatibility where possible or update your calibration nodes accordingly.
- **Synchronization:** If you intend to keep your local version synchronized with future updates from the main QUAM Builder repository, be aware that modifying the core component files can lead to merge conflicts when you try to pull the latest changes. This requires careful management of your version control.

### Method 2: Extension via QUAM Documentation

An alternative approach exists for extending QUAM components without directly cloning or forking the repository. This method is detailed in the QUAM documentation on [Custom Components](https://qua-platform.github.io/quam/components/custom-components/). Using this approach, you can subclass any existing classes in QUAM Builder, and add parameters and methods, as well as create new QUAM components. However, note that this approach is generally more limited in scope, typically allowing only for the extension of existing components rather than fundamental modifications or additions of entirely new component types in the same manner as direct code modification.

## Contributing

We welcome contributions! Please follow the standard fork-and-pull-request workflow. Ensure your code adheres to existing style conventions and includes appropriate tests and documentation.

## License

This project is licensed under the BSD-3 license.
