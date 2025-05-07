from qualibrate import QualibrationLibrary


def test_06b_echo(library: QualibrationLibrary):
    node = library.nodes["06b_echo"]

    run_summary = node.run(load_data_id=1561, skip_actions=["save_results"])
