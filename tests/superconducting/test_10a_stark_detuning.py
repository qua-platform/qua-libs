from qualibrate import QualibrationLibrary


def test_10a_stark_detuning(library: QualibrationLibrary):
    node_template = library.nodes["10a_stark_detuning"]

    executed_node, run_summary = node_template.run(load_data_id=1568)

    fit_results = executed_node.results["fit_results"]
