import numpy as np
from qualibrate import QualibrationLibrary


def test_08_Ramsey_vs_Flux_Calibration(library: QualibrationLibrary):
    node_template = library.nodes["08_Ramsey_vs_Flux_Calibration"]

    executed_node, run_summary = node_template.run(load_data_id=1566)

    fit_results = executed_node.results["fit_results"]
