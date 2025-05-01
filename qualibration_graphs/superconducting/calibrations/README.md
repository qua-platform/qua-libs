# Calibration Node & Graph Overview

This document provides an overview of the calibration nodes and the concept of calibration graphs within the Qualibrate framework used in this project.

## Collection of Calibration Nodes

The `calibrations` directory contains a collection of standardized Python scripts, each representing a **Calibration Node**. Think of each node as a single, well-defined step in the overall process of calibrating and characterizing qubits or other quantum elements.

A typical calibration node performs the following workflow:

1.  **Loads QUAM State:** Reads the current state of the quantum machine configuration (QUAM).
2.  **Executes QUA Program:** Runs a specific QUA sequence tailored to the calibration task (e.g., spectroscopy, Rabi, T1).
3.  **Performs Analysis:** Processes the acquired data and performs fitting or other analysis to extract relevant parameters.
4.  **Updates QUAM State:** Based on the analysis, it proposes changes to the relevant QUAM parameters (e.g., updating a frequency, amplitude, or duration).
5.  **Saves Results:** Persists the input parameters, raw/processed data, analysis results, generated plots, and the proposed QUAM state updates for review and tracking.

## Node Anatomy Explained

For a detailed breakdown of the internal structure of a typical calibration node, please refer to the [Anatomy of a QualibrationNode](./node_anatomy.ipynb) document. It dissects the `02a_resonator_spectroscopy.py` node section by section, explaining the purpose of the common components like imports, initialization, run actions (`@node.run_action`), QUA program creation, data handling, analysis, state updates, and saving.

## Extending the Calibration Library

You can easily extend this library by adding your own custom calibration nodes. To ensure compatibility and maintainability, new nodes should follow the same standardized structure and conventions outlined in the "Resonator Spectroscopy Node Explained" document. This includes:

- Using the `# %%` separators for code cells.
- Defining parameters in a separate `Parameters` class (usually imported).
- Structuring the workflow using functions decorated with `@node.run_action`.
- Loading and interacting with the `QUAM` object.
- Using `node.results`, `node.outcomes` for storing outputs.
- Using `with node.record_state_updates():` for proposing QUAM changes.
- Calling `node.save()` at the end.

## Creating a Calibration Graph

While individual nodes can be run standalone or via the Qualibrate UI for specific tasks, their real power comes from combining them into a **Calibration Graph**. A graph defines a sequence (or parallel execution) of nodes to perform a more complex calibration routine automatically.

Creating a typical calibration graph involves:

1.  **Using the Qualibrate UI:** The web interface provides a visual editor for building graphs.
2.  **Selecting Nodes:** Dragging and dropping nodes from the available library onto the graph canvas.
3.  **Connecting Nodes:** Defining the execution order by drawing connections between nodes.
4.  **Defining Dependencies:** Specifying conditions for running subsequent nodes (e.g., only run Node B if Node A was "successful") or passing results from one node as input parameters to another.
5.  **Saving the Graph:** Storing the graph configuration for later execution.
6.  **Running the Graph:** Initiating the execution of the entire sequence via the Qualibrate UI. Qualibrate manages the execution flow, parameter passing, and state updates according to the graph definition.

Calibration graphs allow for robust, automated calibration sequences that can adapt based on intermediate results. For a more detailed explanation of graph components, dependencies, and advanced features, please refer to the [Anatomy of a Calibration Graph](./graph_anatomy.ipynb) document.

## Running Calibration Nodes and Graphs

There are two primary ways to execute calibration nodes and graphs:

### Running via IDE / Standalone

Each calibration node script is designed to be runnable as a standalone Python file. The use of `# %%` separators allows you to treat the script like a Jupyter Notebook in compatible IDEs (such as VS Code with the Python/Jupyter extensions). You can run the script section by section (cell by cell) within an interactive kernel.

This workflow is ideal for development and debugging:

- Execute cells sequentially to understand the flow.
- Inspect variables and data structures after each step.
- Modify code within a cell and re-run only that cell.
- Test individual components (like QUA program generation, analysis functions) in isolation.

### Running via Qualibrate Frontend

The Qualibrate frontend (web UI) is designed for running stable, well-tested calibration nodes and graphs, particularly when you primarily need to adjust input parameters rather than modify the code itself.

- **Automatic Discovery:** Any calibration node script placed within the `calibrations` folder that follows the standard structure (including `QualibrationNode` instantiation) will automatically be discovered and made available in the Qualibrate UI.
- **Launching the UI:** Start the Qualibrate web application by running the command `qualibrate start` in your terminal within the correct environment. This launches a local web server.
- **Accessing Nodes/Graphs:** Open the provided URL (usually `http://localhost:8001` or similar) in your browser. The UI will list all discovered calibration nodes and saved calibration graphs.
- **Execution:** Select the desired node or graph, modify its input parameters through the UI form, and click "Run" to execute it. The UI will display progress, results, plots, and any proposed state updates for review.

3. Open http://localhost:8001/ on your browser:
   ![browser window](../.img/qualibrate_1.png)
4. Select the node you would like to run:
   ![select node](../.img/qualibrate_2.png)
5. Change the input parameters to your liking:
   ![change parameters](../.img/qualibrate_3.png)
6. Press "Run":
   ![change parameters](../.img/qualibrate_4.png)
