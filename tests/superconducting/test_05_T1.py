import numpy as np
from qualibrate import QualibrationLibrary


def test_05_T1(library: QualibrationLibrary):
    node = library.nodes["05_T1"]

    run_summary = node.run(load_data_id=1585, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
