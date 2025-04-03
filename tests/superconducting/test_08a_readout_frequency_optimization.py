from qualibrate import QualibrationLibrary


def test_08a_readout_frequency_optimization(library: QualibrationLibrary):
    node = library.nodes["08a_readout_frequency_optimization"]

    run_summary = node.run(load_data_id=1563, skip_actions=["save_results"])
