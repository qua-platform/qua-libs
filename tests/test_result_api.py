import string

from qualibs.results.api import *
from random import choice


def test_save_single_result(results_connector):
    now = datetime.datetime.now
    expected = Result(graph_id=1, node_id=1, result_id=1, user_id='Dan', start_time=now(),
                      end_time=now(),
                      res_name="this",
                      res_val='that')
    results_connector.save(expected)
    reader = DataReader(results_connector)
    assert expected == reader.fetch_first(DataReaderQuery(graph_id=1))


def test_save_multiple_results(results_connector):
    now = datetime.datetime.now

    expected = [Result(graph_id=1, node_id=x, result_id=1, user_id='Name', start_time=now(),
                       end_time=now(),
                       res_name="this",
                       res_val='that') for x in range(10)]
    results_connector.save(expected)
    reader = DataReader(results_connector)
    assert expected == reader.fetch(DataReaderQuery(graph_id=1))


def test_fetch_by_graph_id(results_connector):
    now = datetime.datetime.now

    expected = Result(graph_id=100, node_id=1, result_id=1, user_id='Name', start_time=now(),
                      end_time=now(),
                      res_name="this",
                      res_val='that')
    results_connector.save(expected)
    reader = DataReader(results_connector)
    assert expected == reader.fetch_first(DataReaderQuery(graph_id=100))


def test_fetch_by_graph_and_node_id(results_connector):
    now = datetime.datetime.now

    expected = Result(graph_id=100, node_id=2, result_id=1, user_id='Name', start_time=now(),
                      end_time=now(),
                      res_name="this",
                      res_val='that')
    results_connector.save(expected)
    reader = DataReader(results_connector)
    assert expected == reader.fetch_first(DataReaderQuery(graph_id=100, node_id=2))


def test_save_node(results_connector):
    now = datetime.datetime.now

    expected = Node(graph_id=100, node_id=2, node_name='me')
    results_connector.save(expected)
    reader = DataReader(results_connector)
    assert expected == reader.fetch_first(DataReaderQuery(table='Nodes', node_id=2))


def test_fetch_result_by_node_name(results_connector):
    now = datetime.datetime.now

    node = Node(graph_id=1, node_id=1, node_name='test_name')
    res = Result(graph_id=1, node_id=1, result_id=1, user_id='Name', start_time=now(),
                 end_time=now(),
                 res_name="this",
                 res_val="that")

    results_connector.save([node, res])
    reader = DataReader(results_connector)
    assert res == reader.fetch_first(DataReaderQuery(table='Results', node_name='test_name'))


def test_fetch_metadata_by_node_name(results_connector):
    now = datetime.datetime.now
    metadat = [Metadatum(graph_id=1, node_id=1, data_id=x, name='a', val='2') for x in range(10)]

    node = Node(graph_id=1, node_id=1, node_name='test_name')

    results_connector.save([*metadat, node])
    reader = DataReader(results_connector)
    assert metadat == reader.fetch(DataReaderQuery(table='Metadata', node_name='test_name'))

def test_fetch_metadata_by_res_size(results_connector):
    now = datetime.datetime.now


    val=''.join(choice(string.digits + string.ascii_letters) for i in range(int(1e4)))
    res = Result(graph_id=1, node_id=1, result_id=1, user_id='Name', start_time=now(),
                 end_time=now(),
                 res_name="this",
                 res_val=val)

    results_connector.save(res)
    reader = DataReader(results_connector)
    assert res == reader.fetch_first(DataReaderQuery(min_size=int(1e4)))


