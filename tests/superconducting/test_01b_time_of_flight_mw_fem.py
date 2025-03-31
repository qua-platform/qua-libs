import numpy as np
from qualibrate import QualibrationLibrary


def test_01b_time_of_flight_mw_fem(library: QualibrationLibrary):
    node_template = library.nodes["01b_time_of_flight_mw_fem"]

    executed_node, run_summary = node_template.run(load_data_id=1580)

    fit_results = executed_node.results["fit_results"]
