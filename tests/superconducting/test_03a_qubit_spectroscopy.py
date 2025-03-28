import numpy as np
from qualibrate import QualibrationLibrary


def test_03a_qubit_spectroscopy(library: QualibrationLibrary):
    node_template = library.nodes["03a_qubit_spectroscopy"]

    executed_node, run_summary = node_template.run(load_data_id=1583)

    fit_results = executed_node.results["fit_results"]
