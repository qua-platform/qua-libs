from qualibrate import QualibrationLibrary


def test_08c_readout_power_optimization(library: QualibrationLibrary):
    node = library.nodes["08c_readout_power_optimization"]

    run_summary = node_template.run(load_data_id=1564, skip_actions=["save_results"])
