from qualibrate import QualibrationLibrary


def test_07c_Readout_Power_Optimization(library: QualibrationLibrary):
    node_template = library.nodes["07c_Readout_Power_Optimization"]

    executed_node, run_summary = node_template.run(load_data_id=1564)
