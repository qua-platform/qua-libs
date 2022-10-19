from nodeio.inputs import Inputs
from flame.workflow import Workflow

__all__=["ResSpecNode"]

class ResSpecNode(object):

    
    def __init__(self, workflow_node_unique_name,
        freq_init=None):
        """does resonator spec
        
        :param freq_init: (int - STREAM) initial frequency
        """
        self._command = "python3"
        self._bin = "node_res_spec.py"
        self._name = workflow_node_unique_name
        self._icon = ""
        self._inputs = _Inputs(
        freq_init=freq_init,)
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
        freq_init=None):
        self._inputs = Inputs()
        
        self._inputs.state("freq_init", description="initial frequency", units="int")
        self._inputs.set(freq_init=freq_init)
        
    
    
    @property
    def freq_init(self):
        """Input: initial frequency (int)"""
        return self._inputs.get("freq_init")
        
    @freq_init.setter
    def freq_init(self, value):
        """Input: initial frequency (int)"""
        self._inputs.set(freq_init=value)
    

class _Outputs(object):

    def __init__(self, name):
        self._name = name 
        self._outputs = []

    
    
