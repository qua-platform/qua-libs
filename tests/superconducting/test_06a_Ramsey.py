from qualibrate import QualibrationLibrary


def test_06a_ramsey(library: QualibrationLibrary):
    node = library.nodes["06a_Ramsey"]

    run_summary = node.run(load_data_id=1575, skip_actions=["save_results"])
