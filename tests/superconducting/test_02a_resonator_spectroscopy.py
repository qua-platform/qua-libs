from qualibrate import QualibrationLibrary


def test_02a_resonator_spectroscopy(library: QualibrationLibrary):
    node = library.nodes["02a_resonator_spectroscopy"]

    run_summary = node.run(load_data_id=1581, skip_actions=["save_results"])
