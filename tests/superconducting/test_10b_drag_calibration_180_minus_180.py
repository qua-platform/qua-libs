from qualibrate import QualibrationLibrary


def test_10b_drag_calibration_180_minus_180(library: QualibrationLibrary):
    node = library.nodes["10b_drag_calibration_180_minus_180"]

    run_summary = node.run(load_data_id=1570, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
