import numpy as np
from qualibrate import QualibrationLibrary


def test_09a_Stark_Detuning(library: QualibrationLibrary):
    node_template = library.nodes["09a_Stark_Detuning"]

    executed_node, run_summary = node_template.run(load_data_id=1568)

    fit_results = executed_node.results["fit_results"]
