from qualibrate import QualibrationLibrary


def test_05b_T2ey(library: QualibrationLibrary):
    node_template = library.nodes["05b_T2e"]

    executed_node, run_summary = node_template.run(load_data_id=1561)
