from qualibrate import QualibrationLibrary


def test_10a_stark_detuning(library: QualibrationLibrary):
    node = library.nodes["10a_stark_detuning"]

    run_summary = node.run(load_data_id=1568, skip_actions=["save_results"])

    fit_results = node.results["fit_results"]
