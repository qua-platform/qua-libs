import numpy as np
from qualibrate import QualibrationLibrary


def test_11a_single_qubit_randomized_benchmarking(library: QualibrationLibrary):
    node_template = library.nodes["11a_single_qubit_randomized_benchmarking"]

    executed_node, run_summary = node_template.run(load_data_id=1573)

    fit_results = executed_node.results["fit_results"]
