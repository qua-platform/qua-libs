import numpy as np
from qualibrate import QualibrationLibrary


def test_07_iq_blobs(library: QualibrationLibrary):
    node = library.nodes["07_iq_blobs"]

    run_summary = node.run(load_data_id=1586, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
