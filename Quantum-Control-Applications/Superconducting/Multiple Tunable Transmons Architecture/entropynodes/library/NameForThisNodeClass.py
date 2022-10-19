from nodeio.inputs import Inputs
from flame.workflow import Workflow

__all__=["NameForThisNodeClass"]

class NameForThisNodeClass(object):

    
    def __init__(self, workflow_node_unique_name):
        """what it does
        
        """
        self._command = "python3"
        self._bin = "node_res_spec.py"
        self._name = workflow_node_unique_name
        self._icon = ""
        self._inputs = _Inputs()
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

    def __init__(self):
        self._inputs = Inputs()
        
    
    

class _Outputs(object):

    def __init__(self, name):
        self._name = name 
        self._outputs = []

    
    
