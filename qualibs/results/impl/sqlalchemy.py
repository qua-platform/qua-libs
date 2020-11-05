import enum
from typing import Optional, List, Union, Iterable, TypeVar

# import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DATETIME, Enum
from sqlalchemy.orm import sessionmaker
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.util.compat import contextmanager
from sqlalchemy.sql.expression import func

from qualibs.results.api import BaseResultsConnector, DataReaderQuery, Result, Node, Graph, Metadatum

Base = declarative_base()


class Results(Base):
    __tablename__ = 'Results'
    graph_id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey('Nodes.node_id', ondelete="CASCADE"), primary_key=True)
    # result_id = Column(Integer, primary_key=True)
    user_id = Column(String, nullable=False)
    start_time = Column(DATETIME, nullable=False)
    end_time = Column(DATETIME, nullable=False)
    name = Column(String, primary_key=True, nullable=False)
    val = Column(String, nullable=False)

    def __repr__(self):
        return "<Result(graph_id='%s', node_id='%s')>" % (
            self.graph_id, self.node_id)

    def to_model(self):
        return Result(
            graph_id=self.graph_id,
            node_id=self.node_id,
            # result_id=self.result_id,
            user_id=self.user_id,
            start_time=self.start_time,
            end_time=self.end_time,
            name=self.name,
            val=self.val,
        )


class Metadata(Base):
    __tablename__ = 'Metadata'
    graph_id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey('Nodes.node_id', ondelete="CASCADE"), primary_key=True)
    # data_id = Column(Integer, primary_key=True)
    name = Column(String, primary_key=True, nullable=False)
    val = Column(String, nullable=False)

    def __repr__(self):
        return "<Metadata(graph_id='%s', node_id='%s')>" % (
            self.graph_id, self.node_id)

    def to_model(self):
        return Metadatum(
            graph_id=self.graph_id,
            node_id=self.node_id,
            # data_id=self.data_id,
            name=self.name,
            val=self.val
        )


class NodeTypes(enum.Enum):
    Py = 1
    Qua = 2
    Cal = 3


class Nodes(Base):
    __tablename__ = 'Nodes'
    graph_id = Column(Integer, ForeignKey('Graphs.graph_id', ondelete="CASCADE"), primary_key=True)
    node_id = Column(Integer, primary_key=True)
    node_name = Column(String)
    node_type = Column(Enum(NodeTypes))
    version = Column(String)
    results = relationship("Results", cascade="all, delete-orphan")
    metadat = relationship("Metadata", cascade="all, delete-orphan")
    points_to = Column(String)
    program = Column(String)
    input_vars = Column(String)
    node_as_dict = Column(String)

    def __repr__(self):
        return "<Node(graph_id='%s', node_id='%s')>" % (
            self.graph_id, self.node_id)

    def to_model(self):
        return Node(
            graph_id=self.graph_id,
            node_id=self.node_id,
            node_name=self.node_name,
            node_type=self.node_type,
            version=self.version,
            points_to=self.points_to,
            program=self.program,
            input_vars=self.input_vars,
            node_as_dict=self.node_as_dict
        )


class Graphs(Base):
    __tablename__ = 'Graphs'
    graph_id = Column(Integer, primary_key=True)
    graph_name = Column(String)
    graph_script = Column(String)
    graph_dot_repr = Column(String)
    # results = relationship("Results", cascade="all, delete-orphan")
    # metadata = relationship("Metadata", cascade="all, delete-orphan")
    nodes = relationship("Nodes", cascade="all, delete-orphan")

    def __repr__(self):
        return "<Graph(graph_id='%s')>" % (
            self.graph_id)

    def to_model(self):
        return Graph(
            graph_id=self.graph_id,
            graph_name=self.graph_name,
            graph_script=self.graph_script,
            graph_dot_repr=self.graph_dot_repr
        )


class SqlAlchemyResultsConnector(BaseResultsConnector):
    def __init__(self, backend=':memory:', echo=False):
        backend = 'sqlite:///' + backend
        super(SqlAlchemyResultsConnector, self).__init__()
        self._engine = create_engine(backend, echo=echo)
        Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        # session = Session()

    @contextmanager
    def _session_maker(self):
        """Provide a transactional scope around a series of operations."""
        session = self._Session()
        try:
            yield session
            session.commit()
        except DBAPIError:
            session.rollback()
            raise
        finally:
            session.close()

    T = TypeVar('T', Result, Graph, Node, Metadatum)

    def table_setter(self, item):
        if isinstance(item, Node) or item == 'Nodes':
            return Nodes
        elif isinstance(item, Result) or item == 'Results':
            return Results
        elif isinstance(item, Graph) or item == 'Graphs':
            return Graphs
        elif isinstance(item, Metadatum) or item == 'Metadata':
            return Metadata
        else:
            raise TypeError('Error in item save type')

    def save(self, items: Union[T, Iterable[T]]):

        with self._session_maker() as sess:
            iter_items = [items] if type(items) in (Result, Node, Metadatum, Graph) else list(items)

            rows = [self.table_setter(it)(**it.__dict__) for it in iter_items]
            if len(rows) == 1:
                sess.add(rows[0])
            else:
                sess.add_all(rows)

    def _query(self, query_obj: DataReaderQuery):
        with self._session_maker() as sess:
            table_object = self.table_setter(query_obj.table)
            query = sess.query(table_object).order_by(table_object.graph_id)
            if query_obj.graph_id:
                query = query.filter(table_object.graph_id == query_obj.graph_id)

            if query_obj.start_time:
                query = query.filter(table_object.start_time >= query_obj.start_time)

            if query_obj.end_time:
                query = query.filter(table_object.start_time <= query_obj.end_time)

            if query_obj.user_id:
                query = query.filter(table_object.user_id == query_obj.user_id)

            if query_obj.node_name:
                query = query.filter(Nodes.node_name == query_obj.node_name)

            if query_obj.graph_name:
                query = query.join(Graphs, Graphs.graph_id == Results.graph_id).filter(
                    Graphs.graph_name == query_obj.graph_name)

            if query_obj.min_size:
                query = query.filter(func.length(Results.val) >= query_obj.min_size)

            return query

    def fetch(self, query_obj: DataReaderQuery) -> List[Result]:
        return list(row.to_model() for row in self._query(query_obj))

    def fetch_first(self, query_obj: DataReaderQuery) -> Optional[Result]:
        return self._query(query_obj).first().to_model()
