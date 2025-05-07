import numpy as np
from qualibrate import QualibrationLibrary


def test_03a_qubit_spectroscopy(library: QualibrationLibrary):
    node = library.nodes["03a_qubit_spectroscopy"]

    run_summary = node.run(load_data_id=1583, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
