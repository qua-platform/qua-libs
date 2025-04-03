from qualibrate import QualibrationLibrary


def test_05b_T2ey(library: QualibrationLibrary):
    node = library.nodes["05b_T2ey"]

    run_summary = node.run(load_data_id=1561, skip_actions=["save_results"])
