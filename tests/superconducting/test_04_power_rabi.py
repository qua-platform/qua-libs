import numpy as np
from qualibrate import QualibrationLibrary


def test_04_power_rabi(library: QualibrationLibrary):
    node_template = library.nodes["04_power_rabi"]

    executed_node, run_summary = node_template.run(load_data_id=1584)

    fit_results = executed_node.results["fit_results"]
