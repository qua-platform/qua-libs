import numpy as np
from qualibrate import QualibrationLibrary


def test_10a_Single_Qubit_Randomized_Benchmarking(library: QualibrationLibrary):
    node_template = library.nodes["10a_Single_Qubit_Randomized_Benchmarking"]

    executed_node, run_summary = node_template.run(load_data_id=1573)

    fit_results = executed_node.results["fit_results"]
