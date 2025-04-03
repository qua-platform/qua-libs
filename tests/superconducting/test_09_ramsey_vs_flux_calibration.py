import numpy as np
from qualibrate import QualibrationLibrary


def test_09_ramsey_vs_flux_calibration(library: QualibrationLibrary):
    node = library.nodes["09_ramsey_vs_flux_calibration"]

    run_summary = node.run(load_data_id=1566, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
