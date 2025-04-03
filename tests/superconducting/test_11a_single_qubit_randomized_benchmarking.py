import numpy as np
from qualibrate import QualibrationLibrary


def test_11a_single_qubit_randomized_benchmarking(library: QualibrationLibrary):
    node = library.nodes["11a_single_qubit_randomized_benchmarking"]

    run_summary = node.run(load_data_id=1573, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
