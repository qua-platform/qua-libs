import datetime
import os
from abc import ABC, abstractmethod
import sqlite3 as sl
from typing import Tuple
from time import time_ns


class GraphStore(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def write_res(self, result: Tuple[int, int, int, str, int, int, str, str]):
        raise NotImplementedError()

    @abstractmethod
    def get_res_by_id(self,id):
        raise NotImplementedError()

    def write_node(self, node: Tuple[int, int]):
        raise NotImplementedError()

    @abstractmethod
    def read_node_by_id(self,id):
        raise NotImplementedError()

    @abstractmethod
    def write_metadata(self,data:Tuple[int,int,int,str,str]):
        raise NotImplementedError()

    # @abstractmethod
    # def read_metadata(self):
    #     raise NotImplementedError()

    @abstractmethod
    def write_graph(self, graph: Tuple[int, str]):
        raise NotImplementedError()

    # @abstractmethod
    # def read_graph(self):
    #     raise NotImplementedError()


class GraphStoreSqlite(GraphStore):

    def __init__(self, db_path):
        # super.__init__(db_path)
        self.con = sl.connect(db_path)

    def init_db(self):
        with self.con:
            self.con.execute(""" 
               CREATE TABLE  IF NOT EXISTS Results 
                ( 
                    graph_id   INT NOT NULL, 
                    node_id    INTEGER NOT NULL, 
                    result_id  INTEGER NOT NULL, 
                    user_id    TEXT, 
                    start_time DATETIME, 
                    end_time   DATETIME, 
                    res_name   TEXT, 
                    res_val    TEXT, 
                    PRIMARY KEY (graph_id, node_id, result_id), 
                    CONSTRAINT node_results_FK FOREIGN KEY (node_id) REFERENCES Nodes (node_id) ON DELETE CASCADE ON UPDATE CASCADE 
                ); 

                        """)
            self.con.execute(""" 
                        CREATE TABLE  IF NOT EXISTS Nodes 
                        ( 
                            graph_id INTEGER NOT NULL, 
                            node_id  INTEGER NOT NULL, 
                            PRIMARY KEY (graph_id, node_id), 
                            CONSTRAINT graph_nodes_FK FOREIGN KEY (graph_id) REFERENCES Graphs (graph_id) ON DELETE CASCADE ON UPDATE CASCADE 
                        ); 
                        """)
            self.con.execute(""" 
                        CREATE TABLE  IF NOT EXISTS Graphs 
                        ( 
                            graph_id     INTEGER NOT NULL PRIMARY KEY, 
                            graph_script TEXT    NOT NULL 
                        ); 

                        """)

            self.con.execute(""" 
                        CREATE TABLE  IF NOT EXISTS Metadata 
                        ( 
                            graph_id INTEGER NOT NULL, 
                            node_id  INTEGER NOT NULL, 
                            data_id  INTEGER NOT NULL, 
                            name     TEXT    NOT NULL, 
                            val      TEXT    NOT NULL, 
                            PRIMARY KEY (graph_id, node_id, data_id), 
                            CONSTRAINT node_metadata_FK FOREIGN KEY (node_id) REFERENCES Nodes(node_id) ON DELETE CASCADE ON UPDATE CASCADE 
                        ); 
                        """)

    def write_res(self, result: Tuple[int, int, int, str, int, int, str, str]):
        with self.con as con:
            con.execute(""" 
                           INSERT INTO Results (graph_id,node_id,result_id,user_id,start_time,end_time,res_name,res_val) 
                           VALUES (?,?,?,?,?,?,?,?); 
                           """, result)

    def get_res_by_id(self,id):
        with self.con as con:
            cur=con.cursor()
            cur.execute(""" 
                           SELECT * FROM Results where result_id=(?)
                           """,(id,))
            rows = cur.fetchall()
            return rows[0]

    def get_res_by_start_date(self,date):
        with self.con as con:
            cur=con.cursor()
            cur.execute(""" 
                           SELECT * FROM Results where start_time=date('1004-01-01')
                           """,(date,))
            rows = cur.fetchall()
            return rows[0]


    def write_node(self, node: Tuple[int, int]):
        with self.con as con:
            con.execute(""" 
                            INSERT INTO Nodes (graph_id,node_id) 
                            VALUES (?,?); 
                        """, node)


    def read_node_by_id(self,id):
        with self.con as con:
            cur = con.cursor()
            cur.execute(""" 
                                  SELECT * FROM Nodes where node_id=(?)
                                  """, (id,))
            rows = cur.fetchall()
            return rows[0]

    def write_metadata(self,data:Tuple[int,int,int,str,str]):
        with self.con as con:
            con.execute(f""" 
                               INSERT INTO Metadata (graph_id,node_id,data_id,name,val) 
                               VALUES (?,?,?,?,?) 
                               """, data)

    def read_metadata(self):
        pass

    def write_graph(self, graph: Tuple[int, str]):
        with self.con as con:
            con.execute("""  
                            INSERT INTO Graphs (graph_id,graph_script) 
                            VALUES (?,?); 
                            """, graph)


if __name__ == '__main__':
    g_id = time_ns()
    n_id = time_ns()
    r_id = time_ns()
    try:
        os.remove('my_db.db')
    except:
        print('no file')
    a = GraphStoreSqlite('my_db.db')
    a.init_db()

    for g_id in range(10):
        a.write_graph((g_id, 'a'))
        for n_id in range(10):
            a.write_node((g_id, n_id))
            for r_id in range(2):
                now = datetime.datetime.now().isoformat()
                a.write_res((g_id, n_id, r_id, 'a', now, 1, 'a', 'a'))
                # for d_id in range(3):
                #     a.write_metadata((g_id,n_id,d_id,'a','a'))
    print(a.get_res_by_id(1))
    print(a.read_node_by_id(1))


