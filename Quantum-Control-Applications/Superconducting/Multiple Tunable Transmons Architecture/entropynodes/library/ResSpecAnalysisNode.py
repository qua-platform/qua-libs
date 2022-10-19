from nodeio.inputs import Inputs
from flame.workflow import Workflow

__all__=["ResSpecAnalysisNode"]

class ResSpecAnalysisNode(object):

    
    def __init__(self, workflow_node_unique_name,
        IQ=None):
        """finds resonator spectroscopy
        
        :param IQ: (list - STREAM) measurement data
        """
        self._command = "python3"
        self._bin = "node_res_spec_analysis.py"
        self._name = workflow_node_unique_name
        self._icon = ""
        self._inputs = _Inputs(
        IQ=IQ,)
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
        IQ=None):
        self._inputs = Inputs()
        
        self._inputs.state("IQ", description="measurement data", units="list")
        self._inputs.set(IQ=IQ)
        
    
    
    @property
    def IQ(self):
        """Input: measurement data (list)"""
        return self._inputs.get("IQ")
        
    @IQ.setter
    def IQ(self, value):
        """Input: measurement data (list)"""
        self._inputs.set(IQ=value)
    

class _Outputs(object):

    def __init__(self, name):
        self._name = name 
        self._outputs = [
            "state",]

    
    @property
    def state(self):
        """Output: updated state
        :return: (JSON)
        """
        return "#" + self._name + "/state"
    
    
