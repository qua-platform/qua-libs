import numpy as np
from qualibrate import QualibrationLibrary


def test_01b_time_of_flight_mw_fem(library: QualibrationLibrary):
    node = library.nodes["01b_time_of_flight_mw_fem"]

    run_summary = node.run(load_data_id=1580, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
