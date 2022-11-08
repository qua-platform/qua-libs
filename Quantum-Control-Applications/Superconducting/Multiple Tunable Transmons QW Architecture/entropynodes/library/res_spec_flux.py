from nodeio.inputs import Inputs
from flame.workflow import Workflow

__all__=["res_spec_flux"]

class res_spec_flux(object):

    
    def __init__(self, workflow_node_unique_name,
        state=None,
        resources=None,
        debug=None):
        """flux map of resonator spec
        
        :param state: (JSON - STREAM) boostrap state
        :param resources: (list - STREAM) contains digital outputs, qubits, and resonators to be used
        :param debug: (boolean - STREAM) triggers live plot visualization for debug purposes
        """
        self._command = "python3"
        self._bin = "node_res_spec_vs_flux.py"
        self._name = workflow_node_unique_name
        self._icon = ""
        self._inputs = _Inputs(
        state=state,
        resources=resources,
        debug=debug,)
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
        state=None,
        resources=None,
        debug=None):
        self._inputs = Inputs()
        
        self._inputs.state("state", description="boostrap state", units="JSON")
        self._inputs.set(state=state)
        
        self._inputs.state("resources", description="contains digital outputs, qubits, and resonators to be used", units="list")
        self._inputs.set(resources=resources)
        
        self._inputs.state("debug", description="triggers live plot visualization for debug purposes", units="boolean")
        self._inputs.set(debug=debug)
        
    
    
    @property
    def state(self):
        """Input: boostrap state (JSON)"""
        return self._inputs.get("state")
        
    @state.setter
    def state(self, value):
        """Input: boostrap state (JSON)"""
        self._inputs.set(state=value)
    
    @property
    def resources(self):
        """Input: contains digital outputs, qubits, and resonators to be used (list)"""
        return self._inputs.get("resources")
        
    @resources.setter
    def resources(self, value):
        """Input: contains digital outputs, qubits, and resonators to be used (list)"""
        self._inputs.set(resources=value)
    
    @property
    def debug(self):
        """Input: triggers live plot visualization for debug purposes (boolean)"""
        return self._inputs.get("debug")
        
    @debug.setter
    def debug(self, value):
        """Input: triggers live plot visualization for debug purposes (boolean)"""
        self._inputs.set(debug=value)
    

class _Outputs(object):

    def __init__(self, name):
        self._name = name 
        self._outputs = [
            "state",]

    
    @property
    def state(self):
        """Output: state with updated res freqs
        :return: (JSON)
        """
        return "#" + self._name + "/state"
    
    
