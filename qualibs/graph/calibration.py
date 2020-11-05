from qualibs.graph import *
from .program_node import print_yellow, print_green, print_red
from datetime import datetime, timedelta
from types import FunctionType
import asyncio


class CalibrationNode(ProgramNode):
    def __init__(self, label: str = None,
                 extract_params: FunctionType = None,
                 optimal_params: dict = None,
                 check_data_params: dict = None,
                 calibrate_params: dict = None,
                 tolerance: dict = None,
                 timeout: timedelta = None,
                 metadata_func: FunctionType = None
                 ):
        super().__init__(label, extract_params, optimal_params, set(optimal_params.keys()) if optimal_params else set(), metadata_func)
        self._type = 'Cal'
        self.optimal_params: dict = self.input_vars  # these are the calibration parameters
        self.tolerance: dict = tolerance  # these are the tolerance for the calibration parameters
        self.state: str = 'out_spec'  # one of {in_spec,out_spec,bad_data}
        self.timeout: timedelta = timeout  # this is the timeout period of the calibration in as a datetime object
        self._last_calibrated: datetime = None

        # a function that returns (dict,str) of extracted params and one of 3 states
        self.extract_params: FunctionType = extract_params
        # parameters for checking data, e.g number of samples of each variable in the experiment
        self.check_data_params: dict = check_data_params
        # parameters for calibrating, e.g number of samples of each variable in the experiment
        self.calibrate_params: dict = calibrate_params

    @property
    def last_calibrated(self):
        return self._last_calibrated

    async def check_data(self):
        self._start_time = datetime.now()
        params, state = await asyncio.get_running_loop().run_in_executor(None,
                                                                         self.extract_params,
                                                                         self.check_data_params,
                                                                         self.optimal_params,
                                                                         self.tolerance)
        self.state = state
        self._end_time = datetime.now()
        return state

    async def calibrate(self):
        self._start_time = datetime.now()
        params, state = await asyncio.get_running_loop().run_in_executor(None,
                                                                         self.extract_params,
                                                                         self.calibrate_params)
        if state == 'bad_data':
            self.state = state
            raise Exception

        self._end_time = datetime.now()
        self._last_calibrated = datetime.now()
        self.optimal_params.update(params)
        self.state = 'in_spec'
        self._fetch_result()

    async def run_async(self) -> None:
        pass

    def _fetch_result(self) -> None:
        self._result = self.optimal_params


class CalibrationGraph(ProgramGraph):

    def __init__(self, label: str = None, graph_db: GraphDB = None):
        super().__init__(label, graph_db)

    async def _graph_traversal(self, graph_db, start_nodes):
        # TODO: Figure out what to do with graph_db

        # the starting nodes of the run
        if not start_nodes:
            # start from the nodes that don't have outgoing edges
            start_nodes = list()
            for node_id in self.nodes:
                if node_id not in self.edges:
                    start_nodes.append(node_id)
        else:
            start_nodes = [n.id for n in start_nodes]

        for node_id in start_nodes:
            node = self.nodes[node_id]
            if graph_db:
                graph_db.save_node(node, self)
                if self.verbose: print_green(f"Saving metadata before running node <{node.label}>")
                graph_db.save_node_metadata(node, self)
            await self.maintain(node_id, graph_db)

    async def maintain(self, node_id, graph_db):
        # recursive maintain
        for depend_id in self.backward_edges.get(node_id, set()):
            node = self.nodes[depend_id]
            if graph_db:
                graph_db.save_node(node, self)
                if self.verbose: print_green(f"Saving metadata before running node <{node.label}>")
                graph_db.save_node_metadata(node, self)
            await self.maintain(depend_id, graph_db)

        # check_state
        if await self.check_state(node_id):
            return

        # check_data
        self._feed_input(node_id)
        state = await self.nodes[node_id].check_data()
        if state == 'in_spec':
            return
        elif state == 'bad_data':
            for depend_id in self.backward_edges[node_id]:
                await self.diagnose(depend_id)

        # calibrate
        self._feed_input(node_id)
        await self.nodes[node_id].calibrate()
        print_green(f"Calibrated node <{self.nodes[node_id].label}> as part of maintain")
        return

    async def check_state(self, node_id):
        node = self.nodes[node_id]

        # out of spec case
        if node.state != 'in_spec':
            return False

        # timeout
        if datetime.now() - node.end_time > node.timeout:
            return False

        # have dependencies recalibrated after node.end_time and pass check_state
        for depend_id in self.backward_edges.get(node_id, set()):
            if self.nodes[depend_id].last_calibrated > node.end_time:
                return False
            if not await self.check_state(depend_id):
                return False

        # success
        return True

    async def diagnose(self, node_id):
        """
        Returns: True id node or dependent recalibrated
        :param node_id:
        :return:
        """
        # check_data
        self._feed_input(node_id)
        state = await self.nodes[node_id].check_data()

        # in spec case
        if state == 'in_spec':
            return False

        # bad data case
        if state == 'bad_data':
            recalibrated = [await self.diagnose(depend_id) for depend_id in self.backward_edges[node_id]]
            if not any(recalibrated):
                return False

        # calibrate
        self._feed_input(node_id)
        await self.nodes[node_id].calibrate()
        print_green(f"Calibrated node <{self.nodes[node_id].label}> as part of diagnose")
        return True
