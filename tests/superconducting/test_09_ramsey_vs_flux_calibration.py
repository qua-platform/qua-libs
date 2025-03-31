import numpy as np
from qualibrate import QualibrationLibrary


def test_09_ramsey_vs_flux_calibration(library: QualibrationLibrary):
    node_template = library.nodes["09_ramsey_vs_flux_calibration"]

    executed_node, run_summary = node_template.run(load_data_id=1566)

    fit_results = executed_node.results["fit_results"]
