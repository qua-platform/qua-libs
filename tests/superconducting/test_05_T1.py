import numpy as np
from qualibrate import QualibrationLibrary


def test_05_T1(library: QualibrationLibrary):
    node_template = library.nodes["05_T1"]

    executed_node, run_summary = node_template.run(load_data_id=1585)

    fit_results = executed_node.results["fit_results"]
