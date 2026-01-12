# Intro to QUAlibrate

Welcome to your **QUAlibrate** introductory tutorial! This example demonstrates how to convert a standard QUA calibration protocol into a structured, modular **QualibrationNode**. We use `time_of_flight.py` protocol as a practical example.

This tutorial is ideal for users who are familiar with QUA and want to learn how to build reusable, automatable calibration nodes using the [QUAlibrate](https://qua-platform.github.io/qualibrate/) framework.



## 🎯 What You’ll Learn

- What QUAlibrate is and how it can simplify calibration workflows
- How to wrap an existing QUA protocol in a `QualibrationNode`
- How to run and test nodes through:
  - The QUAlibrate Web UI, or
  - Python scripts using the `Workflow` interface

---

## Table of Contents

1. [What is QUAlibrate?](#1--what-is-qualibrate)
2. [Installation Guide](#2--installation-guide)
3. [Project Structure](#3--project-structure)
4. [How to convert your QUA program to a QUAlibration Node?](#4--how-to-convert-your-qua-program-into-a-qualibration-node)

---

## 1. 🧠 What is QUAlibrate?

[QUAlibrate](https://qua-platform.github.io/qualibrate/#what-is-qualibrate) is a node-based framework that simplifies the design, execution, and automation of calibration and characterization workflows in QUA.

Using QUAlibrate, you can:

- Create QUAlibration Nodes that encapsulate calibration routines, which can be executed directly through Python or the QUAlibrate Web Interface

- Combine multiple nodes into calibration graphs that define dependencies and automate complex multi-step calibrations

This tutorial focuses on the first step: converting a QUA protocol into a structured, reusable QualibrationNode. We will not cover building calibration graphs in this guide.

📦 For a full example of calibration nodes and a calibration graph for superconducting qubits, see the  [qualibration_graphs/superconducting](https://github.com/qua-platform/qua-libs/tree/main/qualibration_graphs/superconducting) repository.

---

## 2. 🛠 Installation Guide

To install QUAlibrate and prepare your environment, follow the official instructions:
👉 [QUAlibrate Installation Guide](https://qua-platform.github.io/qualibrate/installation/)

---

## 3. 📁 Project Structure

To make the code more modular, reusable, and easier to maintain, we suggest organizing your project according to the following structure:

```
intro-to-QUAlibrate/
├── calibrations/      # Individual calibration scripts (nodes) runnable by QUAlibrate.
│   ├── Convert_QUA_program_to_QUAlibartionNode_Guide.md # Guide to convert your QUA program to QUAlibartionNode
│   ├── time_of_flight.py
│   └── ... (other calibration routines)

├── data/                   # Default location for storing experiment results.
│   └── {project_name}/     # Data organized by project name.
│       └── YYYY-MM-DD/     # Data organized by date.
│           └── #idx_{node_name}_HHMMSS/ # Data for a specific run.
│               ├── data.json       # Structure containing the data outpoutted by the node (fit results, figures,...).
│               ├── arrays.npz      # npz dataset containing the raw data.
│               ├── figures.png     # Generated figures.
│               └── node.json       # Metadat about the node used by QUAlibrate.

│── calibration_utils/  # Specific experiment implementations (e.g.,time_of_flight, Spectroscopy, T1, Ramsey).
│   └── time_of_flight/
│   │   ├── analysis.py     # Contains all the analysis functions.
│   │   ├── parameters.py   # Contains node-specific parameters.
│   │   └── plotting.py     # Contains all the plotting functions.
│   └── ...

├── configuration/          # configuration files.
│   ├── configuration_with_lf_fem_and_mw_fem/    # Example configuration for LF and MW FEM..
│   └── ...

├── README.md               # This file.
└── pyproject.toml # Installation configuration for the package.
```

💡 Why use this structure?
While you can start with everything in a single file, this layout offers several advantages:

- **Clarity** – Separates calibration logic, parameters, configuration, and results.

- **Reusability** – Enables shared analysis or plotting code across multiple nodes.

- **Scalability** – Makes it easier to add new calibrations.


This structure is recommended for clarity and scalability, feel free to adapt it to your needs.

---

## 4. 🔁 How to Convert Your QUA Program into a QUAlibration Node?

QUAlibrate makes it easy to transform a QUA-based calibration protocol into a structured `QualibrationNode`. A typical node includes:

- A `Parameters` class that defines user-configurable inputs, which are exposed in the QUAlibrate Web Interface—allowing parameters to be modified directly from the interface.

- A `QUA program` that uses those parameters

- Simulation or execution, followed by analysis, plotting, and result storage


📘 **Step-by-step guide**: A detailed walk-through of this conversion process is provided
[here](./calibrations/readme.md).
