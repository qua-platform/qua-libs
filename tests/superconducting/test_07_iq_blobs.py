import numpy as np
from qualibrate import QualibrationLibrary


def test_07_iq_blobs(library: QualibrationLibrary):
    node_template = library.nodes["07_iq_blobs"]

    executed_node, run_summary = node_template.run(load_data_id=1586)

    fit_results = executed_node.results["fit_results"]
