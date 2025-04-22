from qualibrate import QualibrationLibrary


def test_08b_readout_power_optimization(library: QualibrationLibrary):
    node = library.nodes["08b_readout_power_optimization"]

    run_summary = node.run(load_data_id=1564, skip_actions=["save_results"])
