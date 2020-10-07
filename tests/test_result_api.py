from qualibs.results.api import *


def test_save_single_result(results_connector):
    now = datetime.datetime.now
    expected = Result(graph_id=1, node_id=1, result_id=1, user_id='Dan', start_time=now() + datetime.timedelta(days=-2),
                      end_time=now(),
                      res_name="this",
                      res_val='that')
    results_connector.save(expected)
    reader = DataReader(results_connector)  # Gal - this should receive the same connector
    assert expected == reader.fetch_first(DataReaderQuery(graph_id=1))
