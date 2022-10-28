from nodeio.inputs import Inputs
from flame.workflow import Workflow

__all__=["QbSpecNode"]

class QbSpecNode(object):

    
    def __init__(self, workflow_node_unique_name,
        state=None):
        """does qubit spectroscopy
        
        :param state: (JSON - STREAM) state after res spec
        """
        self._command = "python3"
        self._bin = "node_qubit_spec.py"
        self._name = workflow_node_unique_name
        self._icon = ""
        self._inputs = _Inputs(
        state=state,)
        self._outputs = _Outputs(self._name)
        self._host = {}
        Workflow._register_node(self)  # register the node in the workflow context


    def host(self, **kwargs):
        """Sets additional options for execution on the host."""
        for key, value in kwargs.items():
            self._host[key] = value
        return self


    @property
    def i(self):
        """Node inputs"""
        return self._inputs


    @property
    def o(self):
        """Node outputs"""
        return self._outputs

    def __str__(self):
        return self._name


class _Inputs(object):

    def __init__(self,
        state=None):
        self._inputs = Inputs()
        
        self._inputs.state("state", description="state after res spec", units="JSON")
        self._inputs.set(state=state)
        
    
    
    @property
    def state(self):
        """Input: state after res spec (JSON)"""
        return self._inputs.get("state")
        
    @state.setter
    def state(self, value):
        """Input: state after res spec (JSON)"""
        self._inputs.set(state=value)
    

class _Outputs(object):

    def __init__(self, name):
        self._name = name 
        self._outputs = [
            "IQ",]

    
    @property
    def IQ(self):
        """Output: measured IQ data
        :return: (list)
        """
        return "#" + self._name + "/IQ"
    
    
