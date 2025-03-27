import numpy as np
from qualibrate import QualibrationLibrary


def test_02b_resonator_spectroscopy_vs_power(library: QualibrationLibrary):
    node_template = library.nodes["02b_resonator_spectroscopy_vs_power"]

    executed_node, run_summary = node_template.run(load_data_id=1582)

    fit_results = executed_node.results["fit_results"]
    # TODO Why is this unsuccessful?
    # assert not fit_results["q1"]["success"]
    # assert np.isnan(fit_results["q1"]["resonator_frequency"])
    assert fit_results["q3"]["success"]
    assert np.isclose(fit_results["q3"]["resonator_frequency"], 9951200000.0)
