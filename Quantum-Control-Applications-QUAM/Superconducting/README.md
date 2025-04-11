# N Flux-Tunable Transmon Qubits

## Table of Contents

## Installation

TODO Improve naming?

This folder contains an installable module called `quam_builder`, which provides a collection of tailored components for controlling flux-tunable qubits and experiment functionality. These components extend the functionality of QUAM, making it easier to design and execute calibration nodes.

### Requirements

- Python <= 3.12
- A dedicated Python virtual environment.
  Although this is not strictly required, it is highly recommended.
  TODO Add link to instructions
- (Optional) Git client for ...
  TODO Add reasoning

### Downloading the Superconducting Calibrations Folder

Three options

1. If you have a local github account or organization, it is recommended to fork the qua-libs repository and work from there. This enables you to pull periodic updates.
2. As part of a customer installation, the dedicated user repository will be provided for you
3. Navigate to GitHub and Download the qua-libs repository, then unzip the file and use that.
   This method does not require Git, but also does not provide advantages such as version control.

### Installation

Open a terminal, enabled the virtual environment. Then navigate to `Superconducting` folder, ensure you are in the correct virtual environment, then run

```bash
pip install -e .
```

### Configuring QUAlibrate

The calibration framework QUAlibrate needs to be configured with some details to be setup properly. The required details are:

- `project name`: The name of the project. This typically matches a specific QPU chip name, or the name of a project.
- `storage location`: This is where all the measurement data is stored.
  The default is a subfolder in the current working directory: `data/{project_name}`
- `calibration library folder`: This should point to the location of all the calibration nodes and graphs.
  The default points to the `calibration_graph` folder
- `QUAM state path`: The location where the QUAM state is stored.
  The QUAM state contains all the relevant information of your quantum setup, including qubit parameters, connectivity, etc.
  The default location is the subfolder `quam_state`.

These details can be configured by running

```shell
python create_qualibrate_config.py
```

This will interactively guide you through the process. You can hit `y` to accept all defaults, or `n` to manually override (some) default values.
You will also get a final confirmation for the full QUAlibrate config, which contains a lot of additional settings. These should typically be left as is, though you can find the descriptions in [QUAlibrate Configuration File](https://qua-platform.github.io/qualibrate/configuration/).

### Verify QUAlibrate configuration

After having set up the QUAlibrate configuration, it should be ready for use.
To verify that `qualibrate` installed correctly, you can launch the web interface:

Then, open a browser to http://127.0.0.1:8001, where you should see the list of calibration nodes stored in the
`calibration_graph` directory.

## Folder structure

The typical folder structure for the superconducting calibrations library is as follows:

```
├───calibration_graph
│   ├───01a_time_of_flight.py
│   └───...
│
├───data
│   └───{project_name}
│       └───2024-09-17
│           └───#1_01_Time_of_Flight_152438
│               └───quam_state
│
├───quam_config
│   └───quam_state
│
├───quam_experiments
│   └───analysis
│   └───experiments
│   └───parameters
│   └───workflows
```

### calibration_graph

This folder contains all the calibration scripts that can compose a qualibrate graph.
The structure of the nodes is described below.

### data

This folder contains the data that will be saved after the execution of each calibration node.
The [data handler](https://github.com/qua-platform/py-qua-tools/tree/main/qualang_tools/results#data-handler) is used to save data into an automatically generated folder with folder structure:
`<path_to_your_data_folder>/%Y-%m-%d/#{idx}_{name}_%H%M%S`

The saved data can have a different format depending on its type:

- The figures are saved as .png.
- The arrays are saved as .npz.
- The node parameters, state and wiring are saved as .json.

## Creating Custom Nodes
