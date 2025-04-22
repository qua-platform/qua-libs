import numpy as np
from qualibrate import QualibrationLibrary


def test_04b_power_rabi(library: QualibrationLibrary):
    node = library.nodes["04b_power_rabi"]

    run_summary = node.run(load_data_id=1584, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
