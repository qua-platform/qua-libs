from qualibrate import QualibrationLibrary


def test_09b_DRAG_Calibration_180_minus_180(library: QualibrationLibrary):
    node_template = library.nodes["09b_DRAG_Calibration_180_minus_180"]

    executed_node, run_summary = node_template.run(load_data_id=1570)

    fit_results = executed_node.results["fit_results"]
