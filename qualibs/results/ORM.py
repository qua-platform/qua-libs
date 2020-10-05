import datetime

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Sequence, DATETIME
from sqlalchemy.orm import sessionmaker
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship

Base = declarative_base()


class Results(Base):
    __tablename__ = 'Results'
    graph_id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey('Nodes.node_id'),primary_key=True)
    result_id = Column(Integer, primary_key=True)
    user_id=Column(String,nullable=False)
    start_time = Column(DATETIME, nullable=False)
    end_time = Column(DATETIME, nullable=False)
    res_name = Column(String, nullable=False)
    res_val = Column(String, nullable=False)

    def __repr__(self):
        return "<Result(graph_id='%s', node_id='%s', result_id='%s')>" % (
            self.graph_id, self.node_id, self.result_id)


class Metadata(Base):
    __tablename__ = 'Metadata'
    graph_id = Column(Integer, primary_key=True)
    node_id = Column(Integer, ForeignKey('Nodes.node_id'),primary_key=True)
    data_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    val = Column(String, nullable=False)

    def __repr__(self):
        return "<Metadata(graph_id='%s', node_id='%s', data_id='%s')>" % (
            self.graph_id, self.node_id, self.data_id)


class Nodes(Base):
    __tablename__ = 'Nodes'
    graph_id = Column(Integer, ForeignKey('Graphs.graph_id'), primary_key=True)
    node_id = Column(Integer, primary_key=True)

    def __repr__(self):
        return "<Node(graph_id='%s', node_id='%s')>" % (
            self.graph_id, self.node_id)


class Graphs(Base):
    __tablename__ = 'Graphs'
    graph_id = Column(Integer, primary_key=True)
    graph_script = Column(String)

    def __repr__(self):
        return "<Graph(graph_id='%s')>" % (
            self.graph_id)


if __name__ == '__main__':
    # engine = create_engine('sqlite:///:memory:', echo=True)
    now = datetime.datetime.now
    engine = create_engine('sqlite:///my_db.db', echo=True)
    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)
    session = Session()
    a = Results(graph_id=1, node_id=1, result_id=1, user_id='Gal',start_time=now(), end_time=now(), res_name="this", res_val='that')
    b = Results(graph_id=2, node_id=2, result_id=2, user_id='Gal',start_time=now(), end_time=now(), res_name="this", res_val='that')
    session.add_all([a, b])
    # # b=Results(graph_id=2,node_id=1,result_id=1)
    # # session.add(b)
    session.commit()
