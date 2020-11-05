import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, TypeVar, List, Union
import abc


@dataclass
class DataReaderQuery:
    table: str = 'Results'
    node_name: Optional[str] = None
    graph_name: Optional[str] = None
    # object to collect arguments to the query
    graph_id: Optional[int] = None
    node_id: Optional[int] = None
    result_id: Optional[int] = None
    id_range: Optional[Tuple[int, int]] = None
    start_time: Optional[datetime.datetime] = None
    end_time: Optional[datetime.datetime] = None
    user_id: Optional[str] = None
    min_size: Optional[int] = None


@dataclass
class Result:
    graph_id: int
    node_id: int
    # result_id: int
    start_time: datetime.datetime
    end_time: datetime.datetime
    name: str
    val: str
    user_id: str


@dataclass
class Node:
    graph_id: int
    node_id: int
    node_name: str
    node_type: int
    version: str
    points_to: str
    program: str
    input_vars: str
    node_as_dict: str


@dataclass
class Metadatum:
    graph_id: int
    node_id: int
    # data_id: int
    name: str
    val: str


@dataclass
class Graph:
    graph_id: int
    graph_name: str
    graph_script: str
    graph_dot_repr: str


class BaseResultsConnector(abc.ABC):
    def __init__(self):
        super(BaseResultsConnector, self).__init__()
        pass

    @abc.abstractmethod
    def save(self, items: Union[Result, Iterable[Result]]):
        pass

    @abc.abstractmethod
    def fetch(self, query_obj: DataReaderQuery) -> List[Result]:
        pass

    @abc.abstractmethod
    def fetch_first(self, query_obj: DataReaderQuery) -> Optional[Result]:
        pass


class NodeDataWriter:
    def __init__(self, node_id, graph_id, node_name, connector: BaseResultsConnector):
        self._node_id = node_id
        self._graph_id = graph_id
        self._node_name = node_name
        self._connector = connector

    def save(self):
        self._connector.save(DataReaderQuery(graph_id=self._graph_id, node_id=self._node_id))


class DataReader:
    """ A class to run queries against the persistence backend """

    def __init__(self, connector: BaseResultsConnector):
        self.conn = connector

    def fetch(self, query_obj: DataReaderQuery) -> List[Result]:
        """
                Fetch records from the persistence backend using a DataReaderQuery object
                :param query_obj: object containing query parameters
                :type: query_obj: DataReaderQuery
                :return:
         """
        if not isinstance(query_obj, DataReaderQuery):
            raise TypeError('query_obj must be a DataReaderQuery object')
        return self.conn.fetch(query_obj)
        # if query_obj.id:
        #     self.conn.fetch(id=query_obj.id)
        # elif query_obj.start_time:
        #     self.conn.fetch(start_time=query_obj.start_time)

    def fetch_first(self, query_obj: DataReaderQuery) -> Result:
        items = self.fetch(query_obj)
        return items[0] if len(items) > 0 else None
