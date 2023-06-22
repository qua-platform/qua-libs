# QuAM class automatically generated using QuAM SDK (ver 0.8.0)
# open source code and documentation is available at
# https://github.com/entropy-lab/quam-sdk

from typing import List, Union
import json
import sys
import os
__all__ = ["QuAM"]


class _List(object):
    """Wraps lists of simple objects to provide _record_updates capability"""

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema

    def __getitem__(self, key):
        return _List(
            self._quam,
            self._path,
            self._index + [key],
            self._schema
        )
    
    def __setitem__(self, key, newvalue):
        index = self._index + [key] 
        if (len(index) > 0):
            value_ref = self._quam._json[self._path]
            for i in range(len(index)-1):
                value_ref = value_ref[index[i]]
            value_ref[index[-1]] = newvalue
        self._quam._updates["keys"].append(self._path)
        self._quam._updates["indexes"].append(index)
        self._quam._updates["values"].append(newvalue)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self):
        value_ref = self._quam._json[self._path]
        for i in range(len(self._index)-1):
            value_ref = value_ref[self._index[i]]
        return len(value_ref)

    def _json_view(self):
        value_ref = self._quam._json[self._path]
        if (len(self._index)>0):
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            return(value_ref[self._index[-1]])
        return value_ref    

class _add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass


class Network(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def qop_ip(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "qop_ip"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @qop_ip.setter
    def qop_ip(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "qop_ip")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "qop_ip"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "qop_ip"] = value

    @property
    def qop_port(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "qop_port"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @qop_port.setter
    def qop_port(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "qop_port")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "qop_port"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "qop_port"] = value

    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Network):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Qubits(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def freq(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @freq.setter
    def freq(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "freq")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "freq"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "freq"] = value

    @property
    def power(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "power"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @power.setter
    def power(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "power")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "power"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "power"] = value

    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Qubits):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Readout(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def freq(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @freq.setter
    def freq(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "freq")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "freq"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "freq"] = value

    @property
    def power(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "power"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @power.setter
    def power(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "power")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "power"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "power"] = value

    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Readout):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class ReadoutList(object):

    def __init__(self, quam, path, index, schema):
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Readout:
        return Readout(
            self._quam,
            self._path + "[].",
            self._index + [key],
            self._schema["items"]
        )

    def __setitem__(self, key, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "", value, index=key)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self):
        length = self._quam._json[self._path + "[]_len"]
        for i in range(len(self._index)):
            length = length[self._index[i]]
        return length

    def _json_view(self):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view())
        return result

    def append(self, json_item:dict):
        """Adds a new readout by adding a JSON dictionary with following schema
{
  "freq": {
    "type": "number"
  },
  "power": {
    "type": "number"
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[].", self._index, new_item=True)
        quam_sdk.crud.bump_list_length(
            self._quam._json,
            f"{self._path}[]_len",
            self._index
        )

    def __str__(self) -> str:
        return json.dumps(self._json_view())

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)

class Local_oscillators(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def readout(self) -> ReadoutList:
        """"""
        return ReadoutList(
            self._quam, self._path + "readout", self._index,
            self._schema["properties"]["readout"]
        )

    @property
    def qubits(self) -> Qubits:
        """"""
        return Qubits(
            self._quam, self._path + "qubits.", self._index,
            self._schema["properties"]["qubits"]
        )
    @qubits.setter
    def qubits(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "qubits", value)
    
    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Local_oscillators):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Flux(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def dc(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "dc",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "dc"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @dc.setter
    def dc(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "dc")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "dc"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "dc"] = value

    @property
    def fast_flux(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "fast_flux",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "fast_flux"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @fast_flux.setter
    def fast_flux(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "fast_flux")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "fast_flux"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "fast_flux"] = value

    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Flux):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Crosstalk(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def rf(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "rf",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "rf"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rf.setter
    def rf(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "rf")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "rf"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "rf"] = value

    @property
    def flux(self) -> Flux:
        """"""
        return Flux(
            self._quam, self._path + "flux.", self._index,
            self._schema["properties"]["flux"]
        )
    @flux.setter
    def flux(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "flux", value)
    
    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Crosstalk):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Wiring(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def I(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @I.setter
    def I(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "I")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "I"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "I"] = value

    @property
    def Q(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @Q.setter
    def Q(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "Q")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "Q"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "Q"] = value

    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Wiring):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Qubit(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def f01(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "f01"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f01.setter
    def f01(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "f01")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "f01"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f01"] = value

    @property
    def T1(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "T1"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @T1.setter
    def T1(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "T1")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "T1"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "T1"] = value

    @property
    def wiring(self) -> Wiring:
        """"""
        return Wiring(
            self._quam, self._path + "wiring.", self._index,
            self._schema["properties"]["wiring"]
        )
    @wiring.setter
    def wiring(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "wiring", value)
    
    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Qubit):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class QubitsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Qubit:
        return Qubit(
            self._quam,
            self._path + "[].",
            self._index + [key],
            self._schema["items"]
        )

    def __setitem__(self, key, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "", value, index=key)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self):
        length = self._quam._json[self._path + "[]_len"]
        for i in range(len(self._index)):
            length = length[self._index[i]]
        return length

    def _json_view(self):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view())
        return result

    def append(self, json_item:dict):
        """Adds a new qubit by adding a JSON dictionary with following schema
{
  "f01": {
    "type": "number"
  },
  "T1": {
    "type": "number"
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "I": {
        "type": "integer"
      },
      "Q": {
        "type": "integer"
      }
    },
    "required": [
      "I",
      "Q"
    ]
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[].", self._index, new_item=True)
        quam_sdk.crud.bump_list_length(
            self._quam._json,
            f"{self._path}[]_len",
            self._index
        )

    def __str__(self) -> str:
        return json.dumps(self._json_view())

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)

class Wiring2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def I(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @I.setter
    def I(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "I")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "I"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "I"] = value

    @property
    def Q(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @Q.setter
    def Q(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "Q")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "Q"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "Q"] = value

    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Wiring2):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Mixer_correction(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def offset_I(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset_I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset_I.setter
    def offset_I(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset_I")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset_I"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset_I"] = value

    @property
    def offset_Q(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset_Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset_Q.setter
    def offset_Q(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset_Q")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset_Q"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset_Q"] = value

    @property
    def gain(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "gain"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gain.setter
    def gain(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "gain")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "gain"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "gain"] = value

    @property
    def phase(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "phase"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @phase.setter
    def phase(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "phase")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "phase"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "phase"] = value

    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Mixer_correction):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Resonator(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def f_res(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "f_res"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f_res.setter
    def f_res(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "f_res")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "f_res"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f_res"] = value

    @property
    def depletion_time(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "depletion_time"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @depletion_time.setter
    def depletion_time(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "depletion_time")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "depletion_time"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "depletion_time"] = value

    @property
    def time_of_flight(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "time_of_flight"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @time_of_flight.setter
    def time_of_flight(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "time_of_flight")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "time_of_flight"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "time_of_flight"] = value

    @property
    def readout_pulse_length(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "readout_pulse_length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_pulse_length.setter
    def readout_pulse_length(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "readout_pulse_length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "readout_pulse_length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "readout_pulse_length"] = value

    @property
    def readout_pulse_amp(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "readout_pulse_amp"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_pulse_amp.setter
    def readout_pulse_amp(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "readout_pulse_amp")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "readout_pulse_amp"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "readout_pulse_amp"] = value

    @property
    def rotation_angle(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "rotation_angle"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rotation_angle.setter
    def rotation_angle(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "rotation_angle")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "rotation_angle"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "rotation_angle"] = value

    @property
    def wiring(self) -> Wiring2:
        """"""
        return Wiring2(
            self._quam, self._path + "wiring.", self._index,
            self._schema["properties"]["wiring"]
        )
    @wiring.setter
    def wiring(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "wiring", value)
    
    @property
    def mixer_correction(self) -> Mixer_correction:
        """"""
        return Mixer_correction(
            self._quam, self._path + "mixer_correction.", self._index,
            self._schema["properties"]["mixer_correction"]
        )
    @mixer_correction.setter
    def mixer_correction(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "mixer_correction", value)
    
    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Resonator):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class ResonatorsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Resonator:
        return Resonator(
            self._quam,
            self._path + "[].",
            self._index + [key],
            self._schema["items"]
        )

    def __setitem__(self, key, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "", value, index=key)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __len__(self):
        length = self._quam._json[self._path + "[]_len"]
        for i in range(len(self._index)):
            length = length[self._index[i]]
        return length

    def _json_view(self):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view())
        return result

    def append(self, json_item:dict):
        """Adds a new resonator by adding a JSON dictionary with following schema
{
  "f_res": {
    "type": "number"
  },
  "depletion_time": {
    "type": "integer"
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "I": {
        "type": "integer"
      },
      "Q": {
        "type": "integer"
      }
    },
    "required": [
      "I",
      "Q"
    ]
  },
  "time_of_flight": {
    "type": "integer"
  },
  "mixer_correction": {
    "type": "object",
    "title": "mixer_correction",
    "properties": {
      "offset_I": {
        "type": "number"
      },
      "offset_Q": {
        "type": "number"
      },
      "gain": {
        "type": "number"
      },
      "phase": {
        "type": "number"
      }
    },
    "required": [
      "offset_I",
      "offset_Q",
      "gain",
      "phase"
    ]
  },
  "readout_pulse_length": {
    "type": "integer"
  },
  "readout_pulse_amp": {
    "type": "number"
  },
  "rotation_angle": {
    "type": "number"
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[].", self._index, new_item=True)
        quam_sdk.crud.bump_list_length(
            self._quam._json,
            f"{self._path}[]_len",
            self._index
        )

    def __str__(self) -> str:
        return json.dumps(self._json_view())

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)

class QuAM(object):

    def __init__(self, data: Union[None, str, dict] = None, flat_data=True):
        """
        
        Args:
            data: filename or dictionary with QuAM data
            flat_data: optional, is data stored as flat dictionary optimized
                for storage. Defaults to True.
        """
        self._quam: QuAM = self
        self._path = ""
        self._index = []
        self._record_updates = False
        self._updates = {"keys":[], "indexes":[], "values":[], "items":[]}
        self._schema_flat = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'name': 'QuAM storage format', 'description': 'optimized data structure for communication and storage', 'type': 'object', 'properties': {'local_oscillators.readout[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'resonators[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'network.qop_ip': {'type': 'string'}, 'network.qop_port': {'type': 'integer'}, 'local_oscillators.qubits.freq': {'type': 'number'}, 'local_oscillators.qubits.power': {'type': 'number'}, 'local_oscillators.readout[].freq': {'type': 'array', 'items': {'type': 'number'}}, 'local_oscillators.readout[].power': {'type': 'array', 'items': {'type': 'number'}}, 'crosstalk.flux.dc': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'crosstalk.flux.fast_flux': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'crosstalk.rf': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'qubits[].f01': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[].T1': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[].wiring.I': {'type': 'array', 'items': {'type': 'integer'}}, 'qubits[].wiring.Q': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].f_res': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].depletion_time': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].wiring.I': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].wiring.Q': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].time_of_flight': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].mixer_correction.offset_I': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].mixer_correction.offset_Q': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].mixer_correction.gain': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].mixer_correction.phase': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].readout_pulse_length': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].readout_pulse_amp': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].rotation_angle': {'type': 'array', 'items': {'type': 'number'}}}, 'additionalProperties': False}
        self._schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'title': 'QuAM', 'properties': {'network': {'type': 'object', 'title': 'network', 'properties': {'qop_ip': {'type': 'string'}, 'qop_port': {'type': 'integer'}}, 'required': ['qop_ip', 'qop_port']}, 'local_oscillators': {'type': 'object', 'title': 'local_oscillators', 'properties': {'qubits': {'type': 'object', 'title': 'qubits', 'properties': {'freq': {'type': 'number'}, 'power': {'type': 'number'}}, 'required': ['freq', 'power']}, 'readout': {'type': 'array', 'items': {'type': 'object', 'title': 'readout', 'properties': {'freq': {'type': 'number'}, 'power': {'type': 'number'}}, 'required': ['freq', 'power']}}}, 'required': ['qubits', 'readout']}, 'crosstalk': {'type': 'object', 'title': 'crosstalk', 'properties': {'flux': {'type': 'object', 'title': 'flux', 'properties': {'dc': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'fast_flux': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['dc', 'fast_flux']}, 'rf': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['flux', 'rf']}, 'qubits': {'type': 'array', 'items': {'type': 'object', 'title': 'qubit', 'properties': {'f01': {'type': 'number'}, 'T1': {'type': 'number'}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'I': {'type': 'integer'}, 'Q': {'type': 'integer'}}, 'required': ['I', 'Q']}}, 'required': ['f01', 'T1', 'wiring']}, 'description': 'list of all qubits in the experiment'}, 'resonators': {'type': 'array', 'items': {'type': 'object', 'title': 'resonator', 'properties': {'f_res': {'type': 'number'}, 'depletion_time': {'type': 'integer'}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'I': {'type': 'integer'}, 'Q': {'type': 'integer'}}, 'required': ['I', 'Q']}, 'time_of_flight': {'type': 'integer'}, 'mixer_correction': {'type': 'object', 'title': 'mixer_correction', 'properties': {'offset_I': {'type': 'number'}, 'offset_Q': {'type': 'number'}, 'gain': {'type': 'number'}, 'phase': {'type': 'number'}}, 'required': ['offset_I', 'offset_Q', 'gain', 'phase']}, 'readout_pulse_length': {'type': 'integer'}, 'readout_pulse_amp': {'type': 'number'}, 'rotation_angle': {'type': 'number'}}, 'required': ['f_res', 'depletion_time', 'wiring', 'time_of_flight', 'mixer_correction', 'readout_pulse_length', 'readout_pulse_amp', 'rotation_angle']}}}, 'required': ['network', 'local_oscillators', 'crosstalk', 'qubits', 'resonators']}
        self._runtime_var = dict()  #: scratchpad dictionary of runtime variables for user's convenience. These are not saved when exporting data.
        if data is not None:
            if type(data) is str:
                with open(data, "r") as file:
                    data = json.load(file)
            import quam_sdk.crud
            quam_sdk.crud.change_to_dot_format(data)
            if flat_data:
                self._json = data
            else:
                self._json = {}
                quam_sdk.crud.load_data_to_flat_json(
                    self, data, key_structure="", index=[]
                )
            quam_sdk.crud.validate_input(self._json, self._schema_flat)
        else:
            self._json = None
        self._freeze_attributes = True

    def _reset_update_record(self):
        """Resets self._updates record, but does not undo updates to QuAM 
        data in self._json"""
        self._updates = {"keys":[], "indexes":[], "values":[], "items":[]}

    def _add_updates(self, updates:dict):
        """Adds updates generated as another QuAM instance self._updates.
        See also `_reset_update_record` and `self._updates`
        """
        for j in range(len(updates["keys"])):
            if (len(updates["indexes"][j]) > 0):
                value_ref = self._quam._json[updates["keys"][j]]
                for i in range(len(updates["indexes"][j])-1 ):
                    value_ref = value_ref[updates["indexes"][j][i]]
                value_ref[updates["indexes"][j][-1]] = updates["values"][j]
            else:
                self._quam._json[updates["keys"][j]] = updates["values"][j]

        import quam_sdk.crud
        for item in updates["items"]:
            quam_sdk.crud.load_data_to_flat_json(self._quam, item[0],
                item[1] +"[].", item[2], new_item=True
            )
            self._quam._json[f"{item[1]}[]_len"] += 1

        if self._record_updates:
            self._updates["keys"] += updates["keys"]
            self._updates["indexes"] += updates["indexes"]
            self._updates["values"] += updates["values"]
            self._updates["items"] += updates["items"]

    def _save(self, filename:str, flat_data=True):
        """Saves QuAM data into a given filename. Supported file types: json
        
        Args:
            filename: filename where to save data
            flat_data: optional, should saved data be saved in flattened json
                optimized for storage. Defaults to True.
        """
        filetype = filename.split(".")[-1]
        if (filetype in ["json","JSON"]):
            from quam_sdk.crud import pretty_print_json
            with open(filename, "w") as file:
                if flat_data:
                    file.write(pretty_print_json(self._json))
                else:
                    # structured data output
                    file.write(pretty_print_json(self._json_view()))
        else:
            raise ValueError(f"Unsupported file type {filetype}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._json)})"

    @property
    def qubits(self) -> QubitsList:
        """list of all qubits in the experiment"""
        return QubitsList(
            self._quam, self._path + "qubits", self._index,
            self._schema["properties"]["qubits"]
        )

    @property
    def resonators(self) -> ResonatorsList:
        """"""
        return ResonatorsList(
            self._quam, self._path + "resonators", self._index,
            self._schema["properties"]["resonators"]
        )

    @property
    def network(self) -> Network:
        """"""
        return Network(
            self._quam, self._path + "network.", self._index,
            self._schema["properties"]["network"]
        )
    @network.setter
    def network(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "network", value)
    
    @property
    def local_oscillators(self) -> Local_oscillators:
        """"""
        return Local_oscillators(
            self._quam, self._path + "local_oscillators.", self._index,
            self._schema["properties"]["local_oscillators"]
        )
    @local_oscillators.setter
    def local_oscillators(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "local_oscillators", value)
    
    @property
    def crosstalk(self) -> Crosstalk:
        """"""
        return Crosstalk(
            self._quam, self._path + "crosstalk.", self._index,
            self._schema["properties"]["crosstalk"]
        )
    @crosstalk.setter
    def crosstalk(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "crosstalk", value)
    
    def _json_view(self):
        result = {}
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
            elif not callable(value):
                result[v] = value._json_view()
        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) "
            "has been loaded. Aborting printing.")
        import json
        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, QuAM):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


