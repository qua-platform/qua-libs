# Superconducting Qubit Calibration Library

This repository provides a comprehensive library for calibrating N flux-tunable superconducting transmon qubits using the Quantum Orchestration Platform (QOP), QUAM, and QUAlibrate.
It includes configurable experiment nodes, analysis routines, and tools for managing the quantum system state (QUAM).

## Table of Contents

1.  [Prerequisites](#prerequisites)
2.  [Getting Started](#getting-started)
    * [Downloading the Library](#downloading-the-library)
    * [Installation](#installation)
3.  [Initial Setup (QUAlibrate Configuration)](#initial-setup-qualibrate-configuration)
4.  [Verify Setup](#verify-setup)
5.  [Usage](#usage)
6.  [Project Structure](#project-structure)
7.  [Calibration Nodes](#calibration-nodes)
8.  [Creating Custom Nodes](#creating-custom-nodes)
9.  [Contributing](#contributing)
10. [License](#license)

## Prerequisites

* **Python:** Version 3.9 to 3.12 is supported.
* **Python Virtual Environment:** Strongly recommended to avoid dependency conflicts.
You can create one using:
    * `venv`: `python -m venv .venv `` source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
    * `conda`: `conda create -n qualibrate_env python=3.10 `` conda activate qualibrate_env`
* **Git:** (Optional but Recommended) For version control, easier updates (pulling changes), and collaboration (forking and contributing).
[Install Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
* **Access to Quantum Orchestration Platform (QOP):** Required for running experiments on hardware.

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

1.  **Navigate to the Directory:** Open a terminal or command prompt and change into the `Superconducting` directory within the downloaded/cloned repository.
2.  **Activate Virtual Environment:** Ensure your dedicated Python virtual environment (see [Prerequisites](#prerequisites)) is activated.
3.  **Install the Package:** Run the following command to install the library and its dependencies in editable mode (`-e`), which means changes you make to the source code will be reflected immediately without reinstalling:

    ```bash
    pip install -e .
    ```

    *Note for `uv` users:* If you are using `uv` instead of `pip`, you might need to allow pre-releases depending on the dependencies:
    ```bash
    uv pip install -e . --prerelease=allow
    ```

### Initial Setup (QUAlibrate Configuration)

The QUAlibrate framework needs some initial configuration to know where to find calibration scripts, store data, and manage the system state (QUAM).

1.  **Run the Configuration Script:** Execute the provided script from within the `Superconducting` directory:

    ```bash
    python create_qualibrate_config.py
    ```

2.  **Follow Prompts:** The script will interactively ask for the following details:
    * `project name`: A unique name for your project or QPU chip (e.g., `MyQPU_Chip1`).
    * `storage location`: The root directory where measurement data will be saved.
Default: `data/{project_name}` relative to the current directory.
    * `calibration library folder`: The path to the directory containing calibration nodes/graphs.
Default: `calibration_graph`.
    * `QUAM state path`: The location where the QUAM state file (containing system parameters, connectivity, etc.) is stored.
Default: `quam_state`.

    You can press `Enter` or type `y` to accept the defaults, or `n` to provide custom paths.

3.  **Confirm Full Config:** The script will show the complete QUAlibrate configuration for final confirmation.
For detailed explanations of all settings, refer to the [QUAlibrate Configuration File Documentation](https://qua-platform.github.io/qualibrate/configuration/).

### Verify Setup

To ensure QUAlibrate is installed and configured correctly:

1.  **Launch the Web Interface:** Run the following command in your terminal (the ``` runs it in the background on Linux/macOS):

    ```bash
    qualibrate web `
    ```
    *(On Windows, you might run it without ``` in a separate terminal)*

2.  **Open in Browser:** Navigate to [http://127.0.0.1:8001](http://127.0.0.1:8001).

You should see the QUAlibrate web UI, listing the calibration nodes found in your configured `calibration_graph` directory.

## Usage

*(TODO: Add basic examples here)*

* Example: How to run a single calibration node (e.g., Resonator Spectroscopy) using the `qualibrate run` command or via the web UI.
* Example: How to construct and run a calibration graph.

```python
# Example placeholder: Running a node programmatically (Illustrative)
# from qualibrate import QualibrateClient
#
# client = QualibrateClient()
# result = client.run_node("02a_resonator_spectroscopy", quam_state_path="path/to/your/quam.json")
# print(result)

```

## Project Structure

The library is organized into the following main directories:

```
Superconducting/
├── calibration_graph/      # Individual calibration scripts (nodes) runnable by QUAlibrate.
│   ├── 00_hello_qua.py
│   ├── 01a_time_of_flight.py
│   └── ... (many calibration routines)
│
├── data/                   # Default location for storing experiment results.
│   └── {project_name}/     # Data organized by project name
│       └── YYYY-MM-DD/     # Data organized by date.
│           └── #idx_{node_name}_HHMMSS/ # Data for a specific run.
│               ├── quam_state.json
│               ├── results.npz
│               └── plot.png
│
├── quam_config/            # Scripts and configurations for generating/managing QUAM state files.
│   ├── quam_state/         # Default location for the main QUAM state file.
│   ├── wiring_examples/    # Example configurations for different hardware setups.
│   ├── make_quam.py        # Script to generate a QUAM file.
│   └── ...
│
├── quam_experiments/       # Reusable experiment logic, analysis, plotting, and parameter definitions.
│   ├── analysis/           # Core fitting and analysis functions.
│   ├── experiments/        # Specific experiment implementations (e.g., T1, Ramsey, Spectroscopy).
│   │   └── resonator_spectroscopy/
│   │       ├── analysis.py
│   │       ├── node.py     # (If structured this way) QUA program logic.
│   │       ├── parameters.py
│   │       └── plotting.py
│   │   └── ...
│   ├── parameters/         # Common parameter structures.
│   └── workflow/           # Execution, simulation, and data fetching logic.
│
├── create_qualibrate_config.py # Script for initial QUAlibrate setup.
├── README.md               # This file.
└── setup.py / pyproject.toml # Installation configuration for the package.
```

* **`calibration_graph/`**: Contains individual Python scripts, each representing a calibration "node". These scripts typically import functionality from `quam_experiments`, define parameters, run a QUA program, analyze results, and update the QUAM state. See the `README.md` within this folder for more details on node structure.
* **`data/`**: The default output directory where QUAlibrate saves results (plots, raw data, QUAM state snapshots) from calibration runs, organized by project, date, and run index/name.
* **`quam_config/`**: Tools and examples for creating the `quam_state.json` file, which describes your specific hardware setup (instruments, connections, qubit parameters).
* **`quam_experiments/`**: The core, reusable components for building experiments. This includes standardized ways to define parameters, run QUA programs, perform analysis (like fitting), and plot results, promoting modularity and consistency across different calibration nodes.

## Calibration Nodes

The scripts within the `calibration_graph` directory are the building blocks for automated calibration routines. Each script typically performs a specific measurement (e.g., Resonator Spectroscopy, Rabi Oscillations, T1 measurement). They are designed to be run via the QUAlibrate framework, either individually or as part of a larger calibration sequence (graph).

Refer to the `calibration_graph/README.md` for detailed information on the structure and conventions used for these nodes.

## Creating Custom Nodes

*`(TODO: Add guidelines or link to documentation on how to create new calibration nodes based on the existing structure and quam_experiments modules.)`*

Creating a new calibration typically involves:
1. Defining necessary parameters (potentially reusing/extending those in `quam_experiments/parameters`).
2. Writing the QUA program logic (often within a dedicated function or class, possibly in `quam_experiments/experiments`).
3. Implementing analysis and plotting functions (reusing `quam_experiments/analysis` and plotting utilities).
4. Creating a main script in `calibration_graph` that orchestrates these steps and interacts with QUAlibrate.

## Contributing

*(TODO: Add contribution guidelines - e.g., coding style, testing, pull request process)*

We welcome contributions! Please follow the standard fork-and-pull-request workflow. Ensure your code adheres to existing style conventions and includes appropriate tests and documentation.

## License

*(TODO: Add license information - e.g., MIT, Apache 2.0)*