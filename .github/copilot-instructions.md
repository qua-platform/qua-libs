# QUA Libraries - AI Coding Agent Instructions

This document provides guidance for AI coding agents working on the `qua-libs` repository.

## Project Overview

The `qua-libs` repository is a comprehensive resource for the QUA quantum control platform. It includes tutorials, quantum control applications, and a framework for calibration. The goal is to provide a "batteries-included" experience for QUA users.

The repository is organized into three main sections:

1.  **`Tutorials/`**: A collection of guides and examples for learning QUA and working with the OPX. These are great for understanding the basics.
2.  **`Quantum-Control-Applications/`**: Contains scripts and examples for various qubit types, from basic to advanced protocols. Many of these applications use the `py-qua-tools` library for more efficient QUA programming.
3.  **`qualibration_graphs/`**: A framework for building calibration nodes and graphs for different qubit architectures. This section relies on the `QUAM` framework and the `QUAlibrate` platform.

## Key Concepts and Technologies

- **QUA**: The core platform for quantum control. All code in this repository is designed to work with QUA.
- **QUAM**: The Quantum Abstract Machine, a framework for standardizing quantum machine configurations. The `qualibration_graphs` use QUAM, and the `quam-builder` library is used to create QUAM structures.
- **QUAlibrate**: A platform for qubit calibration. The `qualibration_graphs` are designed to be used with QUAlibrate.
- **`py-qua-tools`**: A helper library for writing QUA programs more efficiently. You'll find it used in the `Quantum-Control-Applications`.
- **`qualibration-libs`**: A library of utility functions that support the calibration nodes and graphs in `qualibration_graphs`.

## Developer Workflows

### Getting Started

- Before working on a specific section, familiarize yourself with its dependencies. For example, if you're working on `Quantum-Control-Applications`, you should understand `py-qua-tools`.
- The `pyproject.toml` and `requirements.txt` files list the project's dependencies.

### Working with Calibrations

- The `qualibration_graphs` are a central part of this repository. When adding a new calibration, you will likely need to:
    1.  Create a new calibration node in the appropriate subdirectory of `qualibration_graphs/superconducting/calibrations/`.
    2.  Update the corresponding `quam_config` if necessary.
    3.  Ensure that the new calibration integrates with the `QUAM` and `QUAlibrate` frameworks.

### Adding Tutorials and Applications

- When adding a new tutorial or application, follow the existing structure.
- Include a `README.md` file to explain the purpose of the example and how to run it.
- If your application uses external libraries, make sure they are documented.

## Important Files and Directories

- **`README.md`**: The main entry point for understanding the repository.
- **`CONTRIBUTING.md`**: Guidelines for contributing to the project.
- **`pyproject.toml`**: Defines project metadata and dependencies.
- **`qualibration_graphs/superconducting/`**: A key area of active development, containing calibration graphs for superconducting qubits.
- **`Quantum-Control-Applications/`**: A rich source of examples for different quantum computing platforms.

By following these guidelines, you can contribute effectively to the `qua-libs` repository and help advance the QUA ecosystem.
