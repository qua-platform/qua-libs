import pytest

from qualibs.results.api import *
from qualibs.results.impl.sqlalchemy import Results, SqlAlchemyResultsConnector



@pytest.fixture()
def results_connector():
    return SqlAlchemyResultsConnector(backend=':memory:', echo=True)  # this is memory sqlite


def test_save_single_result(results_connector):
    now = datetime.datetime.now
    expected = Result(graph_id=1, node_id=1, user_id='Dan', start_time=now() + datetime.timedelta(days=-2),
                      end_time=now(),
                      name="this",
                      val='that')
    results_connector.save(expected)
    reader = DataReader(results_connector)  # Gal - this should receive the same connector
    assert expected == reader.fetch_first(DataReaderQuery(graph_id=1))
