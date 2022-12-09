# QuAM class automatically generated using QuAM SDK
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
        self._quam = quam
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
    def port(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "port"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @port.setter
    def port(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "port")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "port"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "port"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Digital_waveform(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def samples(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "samples",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "samples"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @samples.setter
    def samples(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "samples")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "samples"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "samples"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Digital_waveformsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Digital_waveform:
        return Digital_waveform(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new digital_waveform by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "samples": {
    "type": "array",
    "items": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "number"
        },
        {
          "type": "string"
        },
        {
          "type": "boolean"
        },
        {
          "type": "array"
        }
      ]
    }
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Common_operation(object):

    def __init__(self, quam, path, index, schema):
        """an operation which is common to all elements"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def duration(self) -> float:
        """pulse length [s]"""
        
        value = self._quam._json[self._path + "duration"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @duration.setter
    def duration(self, value: float):
        """pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "duration")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "duration"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "duration"] = value

    @property
    def amplitude(self) -> float:
        """pulse amplitude [V]"""
        
        value = self._quam._json[self._path + "amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @amplitude.setter
    def amplitude(self, value: float):
        """pulse amplitude [V]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "amplitude")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "amplitude"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "amplitude"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class I_up(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

    @property
    def offset(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset.setter
    def offset(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Q_up(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

    @property
    def offset(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset.setter
    def offset(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class I_down(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

    @property
    def offset(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset.setter
    def offset(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset"] = value

    @property
    def gain_db(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "gain_db"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gain_db.setter
    def gain_db(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "gain_db")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "gain_db"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "gain_db"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Switch(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Readout_line(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def length(self) -> float:
        """readout time on this readout line [s]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """readout time on this readout line [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def lo_freq(self) -> float:
        """LO frequency for readout line [Hz]"""
        
        value = self._quam._json[self._path + "lo_freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @lo_freq.setter
    def lo_freq(self, value: float):
        """LO frequency for readout line [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "lo_freq")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "lo_freq"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "lo_freq"] = value

    @property
    def lo_power(self) -> int:
        """LO power for readout line [dBm]"""
        
        value = self._quam._json[self._path + "lo_power"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @lo_power.setter
    def lo_power(self, value: int):
        """LO power for readout line [dBm]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "lo_power")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "lo_power"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "lo_power"] = value

    @property
    def I_up(self) -> I_up:
        """"""
        return I_up(
            self._quam, self._path + "I_up/", self._index,
            self._schema["properties"]["I_up"]
        )

    @property
    def Q_up(self) -> Q_up:
        """"""
        return Q_up(
            self._quam, self._path + "Q_up/", self._index,
            self._schema["properties"]["Q_up"]
        )

    @property
    def I_down(self) -> I_down:
        """"""
        return I_down(
            self._quam, self._path + "I_down/", self._index,
            self._schema["properties"]["I_down"]
        )

    @property
    def switch(self) -> Switch:
        """digital output declaration"""
        return Switch(
            self._quam, self._path + "switch/", self._index,
            self._schema["properties"]["switch"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Readout_linesList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Readout_line:
        return Readout_line(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new readout_line by adding a JSON dictionary with following schema
{
  "length": {
    "type": "number",
    "description": "readout time on this readout line [s]"
  },
  "lo_freq": {
    "type": "number",
    "description": "LO frequency for readout line [Hz]"
  },
  "lo_power": {
    "type": "integer",
    "description": "LO power for readout line [dBm]"
  },
  "I_up": {
    "type": "object",
    "title": "I_up",
    "properties": {
      "controller": {
        "type": "string"
      },
      "channel": {
        "type": "integer"
      },
      "offset": {
        "type": "number"
      }
    },
    "required": [
      "controller",
      "channel",
      "offset"
    ]
  },
  "Q_up": {
    "type": "object",
    "title": "Q_up",
    "properties": {
      "controller": {
        "type": "string"
      },
      "channel": {
        "type": "integer"
      },
      "offset": {
        "type": "number"
      }
    },
    "required": [
      "controller",
      "channel",
      "offset"
    ]
  },
  "I_down": {
    "type": "object",
    "title": "I_down",
    "properties": {
      "controller": {
        "type": "string"
      },
      "channel": {
        "type": "integer"
      },
      "offset": {
        "type": "number"
      },
      "gain_db": {
        "type": "integer"
      }
    },
    "required": [
      "controller",
      "channel",
      "offset",
      "gain_db"
    ]
  },
  "switch": {
    "type": "object",
    "title": "switch",
    "properties": {
      "controller": {
        "type": "string"
      },
      "channel": {
        "type": "integer"
      }
    },
    "required": [
      "controller",
      "channel"
    ]
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Integration_weight(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def cosine(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "cosine",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "cosine"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @cosine.setter
    def cosine(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "cosine")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "cosine"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "cosine"] = value

    @property
    def sine(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "sine",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "sine"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @sine.setter
    def sine(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "sine")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "sine"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "sine"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Integration_weightsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Integration_weight:
        return Integration_weight(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new integration_weight by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "cosine": {
    "type": "array",
    "items": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "number"
        },
        {
          "type": "string"
        },
        {
          "type": "boolean"
        },
        {
          "type": "array"
        }
      ]
    }
  },
  "sine": {
    "type": "array",
    "items": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "number"
        },
        {
          "type": "string"
        },
        {
          "type": "boolean"
        },
        {
          "type": "array"
        }
      ]
    }
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class F_res_vs_charge(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def a(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "a"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @a.setter
    def a(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "a")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "a"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "a"] = value

    @property
    def b(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "b"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @b.setter
    def b(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "b")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "b"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "b"] = value

    @property
    def c(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "c"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @c.setter
    def c(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "c")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "c"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "c"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Correction_matrix(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def readout_line_index(self) -> int:
        """Index of the readout line connected to this resonator."""
        
        value = self._quam._json[self._path + "readout_line_index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_line_index.setter
    def readout_line_index(self, value: int):
        """Index of the readout line connected to this resonator."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "readout_line_index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "readout_line_index"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "readout_line_index"] = value

    @property
    def time_of_flight(self) -> int:
        """Time of flight for this resonator [ns]."""
        
        value = self._quam._json[self._path + "time_of_flight"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @time_of_flight.setter
    def time_of_flight(self, value: int):
        """Time of flight for this resonator [ns]."""
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
    def maximum_amplitude(self) -> float:
        """max amplitude in volts above which the mixer will send higher harmonics."""
        
        value = self._quam._json[self._path + "maximum_amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @maximum_amplitude.setter
    def maximum_amplitude(self, value: float):
        """max amplitude in volts above which the mixer will send higher harmonics."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "maximum_amplitude")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "maximum_amplitude"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "maximum_amplitude"] = value

    @property
    def switch_delay(self) -> int:
        """delay of digital pulse"""
        
        value = self._quam._json[self._path + "switch_delay"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @switch_delay.setter
    def switch_delay(self, value: int):
        """delay of digital pulse"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "switch_delay")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "switch_delay"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "switch_delay"] = value

    @property
    def switch_buffer(self) -> int:
        """buffer of digital pulse"""
        
        value = self._quam._json[self._path + "switch_buffer"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @switch_buffer.setter
    def switch_buffer(self, value: int):
        """buffer of digital pulse"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "switch_buffer")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "switch_buffer"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "switch_buffer"] = value

    @property
    def correction_matrix(self) -> Correction_matrix:
        """"""
        return Correction_matrix(
            self._quam, self._path + "correction_matrix/", self._index,
            self._schema["properties"]["correction_matrix"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Threads(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def thread_cond(self) -> bool:
        """Boolean to decided to put manual threds or not"""
        
        value = self._quam._json[self._path + "thread_cond"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @thread_cond.setter
    def thread_cond(self, value: bool):
        """Boolean to decided to put manual threds or not"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "thread_cond")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "thread_cond"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "thread_cond"] = value

    @property
    def thread(self) -> str:
        """Manual thread being allocated"""
        
        value = self._quam._json[self._path + "thread"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @thread.setter
    def thread(self, value: str):
        """Manual thread being allocated"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "thread")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "thread"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "thread"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Readout_resonator(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def index(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @index.setter
    def index(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "index"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "index"] = value

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def f_res(self) -> float:
        """Resonator resonance frequency [Hz]."""
        
        value = self._quam._json[self._path + "f_res"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f_res.setter
    def f_res(self, value: float):
        """Resonator resonance frequency [Hz]."""
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
    def f_opt(self) -> float:
        """Resonator optimal readout frequency [Hz] (used in QUA)."""
        
        value = self._quam._json[self._path + "f_opt"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f_opt.setter
    def f_opt(self, value: float):
        """Resonator optimal readout frequency [Hz] (used in QUA)."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "f_opt")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "f_opt"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f_opt"] = value

    @property
    def readout_regime(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "readout_regime"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_regime.setter
    def readout_regime(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "readout_regime")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "readout_regime"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "readout_regime"] = value

    @property
    def readout_amplitude(self) -> float:
        """Readout amplitude for this resonator [V]. Must be within [-0.5, 0.5)."""
        
        value = self._quam._json[self._path + "readout_amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_amplitude.setter
    def readout_amplitude(self, value: float):
        """Readout amplitude for this resonator [V]. Must be within [-0.5, 0.5)."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "readout_amplitude")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "readout_amplitude"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "readout_amplitude"] = value

    @property
    def rotation_angle(self) -> float:
        """Angle by which to rotate the IQ blobs to place the separation along the 'I' quadrature [degrees]."""
        
        value = self._quam._json[self._path + "rotation_angle"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rotation_angle.setter
    def rotation_angle(self, value: float):
        """Angle by which to rotate the IQ blobs to place the separation along the 'I' quadrature [degrees]."""
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
    def ge_threshold(self) -> float:
        """Threshold (in demod unit) along the 'I' quadrature discriminating between qubit ground and excited states."""
        
        value = self._quam._json[self._path + "ge_threshold"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ge_threshold.setter
    def ge_threshold(self, value: float):
        """Threshold (in demod unit) along the 'I' quadrature discriminating between qubit ground and excited states."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ge_threshold")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ge_threshold"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ge_threshold"] = value

    @property
    def readout_fidelity(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "readout_fidelity"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_fidelity.setter
    def readout_fidelity(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "readout_fidelity")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "readout_fidelity"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "readout_fidelity"] = value

    @property
    def q_factor(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "q_factor"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @q_factor.setter
    def q_factor(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "q_factor")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "q_factor"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "q_factor"] = value

    @property
    def chi(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "chi"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @chi.setter
    def chi(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "chi")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "chi"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "chi"] = value

    @property
    def relaxation_time(self) -> float:
        """Resonator relaxation time [s]."""
        
        value = self._quam._json[self._path + "relaxation_time"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @relaxation_time.setter
    def relaxation_time(self, value: float):
        """Resonator relaxation time [s]."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "relaxation_time")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "relaxation_time"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "relaxation_time"] = value

    @property
    def integration_weights(self) -> Integration_weightsList:
        """Arbitrary integration weights defined as lists of tuples whose first element is the value of the integration weight and second element is the duration in ns for which this value should be used [(1.0, readout_len)]. The duration must be divisible by 4."""
        return Integration_weightsList(
            self._quam, self._path + "integration_weights", self._index,
            self._schema["properties"]["integration_weights"]
        )

    @property
    def f_res_vs_charge(self) -> F_res_vs_charge:
        """Vertex of the resonator frequency vs charge bias parabola as a * bias**2 + b * bias + c"""
        return F_res_vs_charge(
            self._quam, self._path + "f_res_vs_charge/", self._index,
            self._schema["properties"]["f_res_vs_charge"]
        )

    @property
    def wiring(self) -> Wiring:
        """"""
        return Wiring(
            self._quam, self._path + "wiring/", self._index,
            self._schema["properties"]["wiring"]
        )

    @property
    def threads(self) -> Threads:
        """"""
        return Threads(
            self._quam, self._path + "threads/", self._index,
            self._schema["properties"]["threads"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Readout_resonatorsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Readout_resonator:
        return Readout_resonator(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new readout_resonator by adding a JSON dictionary with following schema
{
  "index": {
    "type": "integer"
  },
  "name": {
    "type": "string"
  },
  "f_res": {
    "type": "number",
    "description": "Resonator resonance frequency [Hz]."
  },
  "f_opt": {
    "type": "number",
    "description": "Resonator optimal readout frequency [Hz] (used in QUA)."
  },
  "readout_regime": {
    "type": "string"
  },
  "readout_amplitude": {
    "type": "number",
    "description": "Readout amplitude for this resonator [V]. Must be within [-0.5, 0.5)."
  },
  "rotation_angle": {
    "type": "number",
    "description": "Angle by which to rotate the IQ blobs to place the separation along the 'I' quadrature [degrees]."
  },
  "integration_weights": {
    "type": "array",
    "items": {
      "type": "object",
      "title": "integration_weight",
      "properties": {
        "name": {
          "type": "string"
        },
        "cosine": {
          "type": "array",
          "items": {
            "anyOf": [
              {
                "type": "integer"
              },
              {
                "type": "number"
              },
              {
                "type": "string"
              },
              {
                "type": "boolean"
              },
              {
                "type": "array"
              }
            ]
          }
        },
        "sine": {
          "type": "array",
          "items": {
            "anyOf": [
              {
                "type": "integer"
              },
              {
                "type": "number"
              },
              {
                "type": "string"
              },
              {
                "type": "boolean"
              },
              {
                "type": "array"
              }
            ]
          }
        }
      },
      "required": [
        "name",
        "cosine",
        "sine"
      ]
    },
    "description": "Arbitrary integration weights defined as lists of tuples whose first element is the value of the integration weight and second element is the duration in ns for which this value should be used [(1.0, readout_len)]. The duration must be divisible by 4."
  },
  "ge_threshold": {
    "type": "number",
    "description": "Threshold (in demod unit) along the 'I' quadrature discriminating between qubit ground and excited states."
  },
  "readout_fidelity": {
    "type": "number"
  },
  "q_factor": {
    "type": "number"
  },
  "chi": {
    "type": "number"
  },
  "relaxation_time": {
    "type": "number",
    "description": "Resonator relaxation time [s]."
  },
  "f_res_vs_charge": {
    "type": "object",
    "title": "f_res_vs_charge",
    "properties": {
      "a": {
        "type": "number"
      },
      "b": {
        "type": "number"
      },
      "c": {
        "type": "number"
      }
    },
    "required": [
      "a",
      "b",
      "c"
    ]
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "readout_line_index": {
        "type": "integer",
        "description": "Index of the readout line connected to this resonator."
      },
      "time_of_flight": {
        "type": "integer",
        "description": "Time of flight for this resonator [ns]."
      },
      "correction_matrix": {
        "type": "object",
        "title": "correction_matrix",
        "properties": {
          "gain": {
            "type": "number"
          },
          "phase": {
            "type": "number"
          }
        },
        "required": [
          "gain",
          "phase"
        ]
      },
      "maximum_amplitude": {
        "type": "number",
        "description": "max amplitude in volts above which the mixer will send higher harmonics."
      },
      "switch_delay": {
        "type": "integer",
        "description": "delay of digital pulse"
      },
      "switch_buffer": {
        "type": "integer",
        "description": "buffer of digital pulse"
      }
    },
    "required": [
      "readout_line_index",
      "time_of_flight",
      "correction_matrix",
      "maximum_amplitude",
      "switch_delay",
      "switch_buffer"
    ]
  },
  "threads": {
    "type": "object",
    "title": "threads",
    "properties": {
      "thread_cond": {
        "type": "boolean",
        "description": "Boolean to decided to put manual threds or not"
      },
      "thread": {
        "type": "string",
        "description": "Manual thread being allocated"
      }
    },
    "required": [
      "thread_cond",
      "thread"
    ]
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class I(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

    @property
    def offset(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset.setter
    def offset(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Q(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

    @property
    def offset(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset.setter
    def offset(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Switch2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Drive_line(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def qubits(self) -> List[Union[str, int, float, bool, list]]:
        """LO frequency [Hz]"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "qubits",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "qubits"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @qubits.setter
    def qubits(self, value: List[Union[str, int, float, bool, list]]):
        """LO frequency [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "qubits")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "qubits"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "qubits"] = value

    @property
    def lo_freq(self) -> float:
        """LO power to drive line [dBm]"""
        
        value = self._quam._json[self._path + "lo_freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @lo_freq.setter
    def lo_freq(self, value: float):
        """LO power to drive line [dBm]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "lo_freq")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "lo_freq"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "lo_freq"] = value

    @property
    def lo_power(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "lo_power"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @lo_power.setter
    def lo_power(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "lo_power")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "lo_power"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "lo_power"] = value

    @property
    def I(self) -> I:
        """"""
        return I(
            self._quam, self._path + "I/", self._index,
            self._schema["properties"]["I"]
        )

    @property
    def Q(self) -> Q:
        """"""
        return Q(
            self._quam, self._path + "Q/", self._index,
            self._schema["properties"]["Q"]
        )

    @property
    def switch(self) -> Switch2:
        """digital output declaration"""
        return Switch2(
            self._quam, self._path + "switch/", self._index,
            self._schema["properties"]["switch"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Drive_linesList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Drive_line:
        return Drive_line(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new drive_line by adding a JSON dictionary with following schema
{
  "qubits": {
    "type": "array",
    "items": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "number"
        },
        {
          "type": "string"
        },
        {
          "type": "boolean"
        },
        {
          "type": "array"
        }
      ]
    },
    "description": "qubits associated with this drive line"
  },
  "lo_freq": {
    "type": "number",
    "description": "LO frequency [Hz]"
  },
  "lo_power": {
    "type": "integer",
    "description": "LO power to drive line [dBm]"
  },
  "I": {
    "type": "object",
    "title": "I",
    "properties": {
      "controller": {
        "type": "string"
      },
      "channel": {
        "type": "integer"
      },
      "offset": {
        "type": "number"
      }
    },
    "required": [
      "controller",
      "channel",
      "offset"
    ]
  },
  "Q": {
    "type": "object",
    "title": "Q",
    "properties": {
      "controller": {
        "type": "string"
      },
      "channel": {
        "type": "integer"
      },
      "offset": {
        "type": "number"
      }
    },
    "required": [
      "controller",
      "channel",
      "offset"
    ]
  },
  "switch": {
    "type": "object",
    "title": "switch",
    "properties": {
      "controller": {
        "type": "string"
      },
      "channel": {
        "type": "integer"
      }
    },
    "required": [
      "controller",
      "channel"
    ]
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Angle2volt(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def deg90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg90.setter
    def deg90(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg90"] = value

    @property
    def deg180(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg180"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg180.setter
    def deg180(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg180")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg180"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg180"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Drag_gaussian(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def length(self) -> float:
        """The pulse length [s]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """The pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def sigma(self) -> float:
        """The gaussian standard deviation (only for gaussian pulses) [s]"""
        
        value = self._quam._json[self._path + "sigma"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @sigma.setter
    def sigma(self, value: float):
        """The gaussian standard deviation (only for gaussian pulses) [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "sigma")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "sigma"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "sigma"] = value

    @property
    def alpha(self) -> float:
        """The DRAG coefficient alpha."""
        
        value = self._quam._json[self._path + "alpha"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @alpha.setter
    def alpha(self, value: float):
        """The DRAG coefficient alpha."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "alpha")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "alpha"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "alpha"] = value

    @property
    def detuning(self) -> int:
        """The frequency shift to correct for AC stark shift [Hz]."""
        
        value = self._quam._json[self._path + "detuning"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @detuning.setter
    def detuning(self, value: int):
        """The frequency shift to correct for AC stark shift [Hz]."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "detuning")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "detuning"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "detuning"] = value

    @property
    def shape(self) -> str:
        """Shape of the gate"""
        
        value = self._quam._json[self._path + "shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @shape.setter
    def shape(self, value: str):
        """Shape of the gate"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "shape")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "shape"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "shape"] = value

    @property
    def angle2volt(self) -> Angle2volt:
        """Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V."""
        return Angle2volt(
            self._quam, self._path + "angle2volt/", self._index,
            self._schema["properties"]["angle2volt"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Angle2volt2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def deg90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg90.setter
    def deg90(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg90"] = value

    @property
    def deg180(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg180"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg180.setter
    def deg180(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg180")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg180"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg180"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Drag_cosine(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def length(self) -> float:
        """The pulse length [s]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """The pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def alpha(self) -> float:
        """The DRAG coefficient alpha."""
        
        value = self._quam._json[self._path + "alpha"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @alpha.setter
    def alpha(self, value: float):
        """The DRAG coefficient alpha."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "alpha")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "alpha"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "alpha"] = value

    @property
    def detuning(self) -> int:
        """The frequency shift to correct for AC stark shift [Hz]."""
        
        value = self._quam._json[self._path + "detuning"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @detuning.setter
    def detuning(self, value: int):
        """The frequency shift to correct for AC stark shift [Hz]."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "detuning")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "detuning"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "detuning"] = value

    @property
    def shape(self) -> str:
        """Shape of the gate"""
        
        value = self._quam._json[self._path + "shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @shape.setter
    def shape(self, value: str):
        """Shape of the gate"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "shape")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "shape"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "shape"] = value

    @property
    def angle2volt(self) -> Angle2volt2:
        """Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V."""
        return Angle2volt2(
            self._quam, self._path + "angle2volt/", self._index,
            self._schema["properties"]["angle2volt"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Angle2volt3(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def deg90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg90.setter
    def deg90(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg90"] = value

    @property
    def deg180(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg180"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg180.setter
    def deg180(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg180")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg180"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg180"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Square(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def length(self) -> float:
        """The pulse length [s]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """The pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def shape(self) -> str:
        """Shape of the gate"""
        
        value = self._quam._json[self._path + "shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @shape.setter
    def shape(self, value: str):
        """Shape of the gate"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "shape")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "shape"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "shape"] = value

    @property
    def angle2volt(self) -> Angle2volt3:
        """Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V."""
        return Angle2volt3(
            self._quam, self._path + "angle2volt/", self._index,
            self._schema["properties"]["angle2volt"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Driving(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def drag_gaussian(self) -> Drag_gaussian:
        """"""
        return Drag_gaussian(
            self._quam, self._path + "drag_gaussian/", self._index,
            self._schema["properties"]["drag_gaussian"]
        )

    @property
    def drag_cosine(self) -> Drag_cosine:
        """"""
        return Drag_cosine(
            self._quam, self._path + "drag_cosine/", self._index,
            self._schema["properties"]["drag_cosine"]
        )

    @property
    def square(self) -> Square:
        """"""
        return Square(
            self._quam, self._path + "square/", self._index,
            self._schema["properties"]["square"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Correction_matrix2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Charge_line(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

    @property
    def offset(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset.setter
    def offset(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Charge_filter_coefficients(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def feedforward(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "feedforward",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "feedforward"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @feedforward.setter
    def feedforward(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "feedforward")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "feedforward"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "feedforward"] = value

    @property
    def feedback(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "feedback",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "feedback"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @feedback.setter
    def feedback(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "feedback")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "feedback"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "feedback"] = value

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
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def drive_line_index(self) -> int:
        """Index of the readout line connected to this qubit."""
        
        value = self._quam._json[self._path + "drive_line_index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @drive_line_index.setter
    def drive_line_index(self, value: int):
        """Index of the readout line connected to this qubit."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "drive_line_index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "drive_line_index"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "drive_line_index"] = value

    @property
    def maximum_amplitude(self) -> float:
        """max amplitude in volts above which the mixer will send higher harmonics."""
        
        value = self._quam._json[self._path + "maximum_amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @maximum_amplitude.setter
    def maximum_amplitude(self, value: float):
        """max amplitude in volts above which the mixer will send higher harmonics."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "maximum_amplitude")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "maximum_amplitude"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "maximum_amplitude"] = value

    @property
    def analog_channel_offset(self) -> float:
        """Voltage value to nullify inheret analog channel offset [V]"""
        
        value = self._quam._json[self._path + "analog_channel_offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @analog_channel_offset.setter
    def analog_channel_offset(self, value: float):
        """Voltage value to nullify inheret analog channel offset [V]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "analog_channel_offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "analog_channel_offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "analog_channel_offset"] = value

    @property
    def switch_delay(self) -> int:
        """delay of digital pulse"""
        
        value = self._quam._json[self._path + "switch_delay"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @switch_delay.setter
    def switch_delay(self, value: int):
        """delay of digital pulse"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "switch_delay")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "switch_delay"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "switch_delay"] = value

    @property
    def switch_buffer(self) -> int:
        """buffer of digital pulse"""
        
        value = self._quam._json[self._path + "switch_buffer"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @switch_buffer.setter
    def switch_buffer(self, value: int):
        """buffer of digital pulse"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "switch_buffer")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "switch_buffer"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "switch_buffer"] = value

    @property
    def correction_matrix(self) -> Correction_matrix2:
        """"""
        return Correction_matrix2(
            self._quam, self._path + "correction_matrix/", self._index,
            self._schema["properties"]["correction_matrix"]
        )

    @property
    def charge_line(self) -> Charge_line:
        """"""
        return Charge_line(
            self._quam, self._path + "charge_line/", self._index,
            self._schema["properties"]["charge_line"]
        )

    @property
    def charge_filter_coefficients(self) -> Charge_filter_coefficients:
        """"""
        return Charge_filter_coefficients(
            self._quam, self._path + "charge_filter_coefficients/", self._index,
            self._schema["properties"]["charge_filter_coefficients"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Threads2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def thread_cond(self) -> bool:
        """Boolean to decided to put manual threds or not"""
        
        value = self._quam._json[self._path + "thread_cond"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @thread_cond.setter
    def thread_cond(self, value: bool):
        """Boolean to decided to put manual threds or not"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "thread_cond")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "thread_cond"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "thread_cond"] = value

    @property
    def thread(self) -> str:
        """Manual thread being allocated"""
        
        value = self._quam._json[self._path + "thread"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @thread.setter
    def thread(self, value: str):
        """Manual thread being allocated"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "thread")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "thread"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "thread"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Charge_bias_point(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def value(self) -> float:
        """Bias voltage to set qubit to degeneracy between even and odd parity [V]"""
        
        value = self._quam._json[self._path + "value"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @value.setter
    def value(self, value: float):
        """Bias voltage to set qubit to degeneracy between even and odd parity [V]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "value")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "value"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "value"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Charge_bias_pointsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Charge_bias_point:
        return Charge_bias_point(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new charge_bias_point by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "value": {
    "type": "number",
    "description": "Bias voltage to set qubit to degeneracy between even and odd parity [V]"
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Constant(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def amplitude(self) -> float:
        """[V]"""
        
        value = self._quam._json[self._path + "amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @amplitude.setter
    def amplitude(self, value: float):
        """[V]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "amplitude")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "amplitude"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "amplitude"] = value

    @property
    def length(self) -> float:
        """[s]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """[s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class ConstantList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Constant:
        return Constant(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new constant by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "amplitude": {
    "type": "number",
    "description": "[V]"
  },
  "length": {
    "type": "number",
    "description": "[s]"
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Arbitrary(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def waveform(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "waveform",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "waveform"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @waveform.setter
    def waveform(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "waveform")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "waveform"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "waveform"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class ArbitraryList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Arbitrary:
        return Arbitrary(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new arbitrary by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "waveform": {
    "type": "array",
    "items": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "number"
        },
        {
          "type": "string"
        },
        {
          "type": "boolean"
        },
        {
          "type": "array"
        }
      ]
    },
    "description": "points describing the waveform shape"
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Sequence_states(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def constant(self) -> ConstantList:
        """"""
        return ConstantList(
            self._quam, self._path + "constant", self._index,
            self._schema["properties"]["constant"]
        )

    @property
    def arbitrary(self) -> ArbitraryList:
        """"""
        return ArbitraryList(
            self._quam, self._path + "arbitrary", self._index,
            self._schema["properties"]["arbitrary"]
        )

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
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def index(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @index.setter
    def index(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "index"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "index"] = value

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def f_01(self) -> float:
        """0-1 transition frequency [Hz]"""
        
        value = self._quam._json[self._path + "f_01"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f_01.setter
    def f_01(self, value: float):
        """0-1 transition frequency [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "f_01")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "f_01"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f_01"] = value

    @property
    def df(self) -> float:
        """Half of charge dispersion measured in spectroscopy [Hz]"""
        
        value = self._quam._json[self._path + "df"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @df.setter
    def df(self, value: float):
        """Half of charge dispersion measured in spectroscopy [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "df")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "df"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "df"] = value

    @property
    def anharmonicity(self) -> float:
        """Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]"""
        
        value = self._quam._json[self._path + "anharmonicity"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @anharmonicity.setter
    def anharmonicity(self, value: float):
        """Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "anharmonicity")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "anharmonicity"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "anharmonicity"] = value

    @property
    def rabi_freq(self) -> int:
        """Qubit Rabi frequency [Hz]"""
        
        value = self._quam._json[self._path + "rabi_freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rabi_freq.setter
    def rabi_freq(self, value: int):
        """Qubit Rabi frequency [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "rabi_freq")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "rabi_freq"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "rabi_freq"] = value

    @property
    def t1(self) -> float:
        """Relaxation time T1 [s]"""
        
        value = self._quam._json[self._path + "t1"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t1.setter
    def t1(self, value: float):
        """Relaxation time T1 [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "t1")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "t1"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "t1"] = value

    @property
    def t2(self) -> float:
        """Dephasing time T2 [s]"""
        
        value = self._quam._json[self._path + "t2"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t2.setter
    def t2(self, value: float):
        """Dephasing time T2 [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "t2")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "t2"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "t2"] = value

    @property
    def t2star(self) -> float:
        """Dephasing time T2* [s]"""
        
        value = self._quam._json[self._path + "t2star"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t2star.setter
    def t2star(self, value: float):
        """Dephasing time T2* [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "t2star")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "t2star"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "t2star"] = value

    @property
    def ramsey_det(self) -> float:
        """Detuning to observe ramsey fringes [Hz]"""
        
        value = self._quam._json[self._path + "ramsey_det"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ramsey_det.setter
    def ramsey_det(self, value: float):
        """Detuning to observe ramsey fringes [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ramsey_det")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ramsey_det"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ramsey_det"] = value

    @property
    def charge_bias_points(self) -> Charge_bias_pointsList:
        """"""
        return Charge_bias_pointsList(
            self._quam, self._path + "charge_bias_points", self._index,
            self._schema["properties"]["charge_bias_points"]
        )

    @property
    def driving(self) -> Driving:
        """"""
        return Driving(
            self._quam, self._path + "driving/", self._index,
            self._schema["properties"]["driving"]
        )

    @property
    def wiring(self) -> Wiring2:
        """"""
        return Wiring2(
            self._quam, self._path + "wiring/", self._index,
            self._schema["properties"]["wiring"]
        )

    @property
    def threads(self) -> Threads2:
        """"""
        return Threads2(
            self._quam, self._path + "threads/", self._index,
            self._schema["properties"]["threads"]
        )

    @property
    def sequence_states(self) -> Sequence_states:
        """"""
        return Sequence_states(
            self._quam, self._path + "sequence_states/", self._index,
            self._schema["properties"]["sequence_states"]
        )

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
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Qubit:
        return Qubit(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
  "index": {
    "type": "integer"
  },
  "name": {
    "type": "string"
  },
  "f_01": {
    "type": "number",
    "description": "0-1 transition frequency [Hz]"
  },
  "df": {
    "type": "number",
    "description": "Half of charge dispersion measured in spectroscopy [Hz]"
  },
  "anharmonicity": {
    "type": "number",
    "description": "Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]"
  },
  "rabi_freq": {
    "type": "integer",
    "description": "Qubit Rabi frequency [Hz]"
  },
  "t1": {
    "type": "number",
    "description": "Relaxation time T1 [s]"
  },
  "t2": {
    "type": "number",
    "description": "Dephasing time T2 [s]"
  },
  "t2star": {
    "type": "number",
    "description": "Dephasing time T2* [s]"
  },
  "ramsey_det": {
    "type": "number",
    "description": "Detuning to observe ramsey fringes [Hz]"
  },
  "driving": {
    "type": "object",
    "title": "driving",
    "properties": {
      "drag_gaussian": {
        "type": "object",
        "title": "drag_gaussian",
        "properties": {
          "length": {
            "type": "number",
            "description": "The pulse length [s]"
          },
          "sigma": {
            "type": "number",
            "description": "The gaussian standard deviation (only for gaussian pulses) [s]"
          },
          "alpha": {
            "type": "number",
            "description": "The DRAG coefficient alpha."
          },
          "detuning": {
            "type": "integer",
            "description": "The frequency shift to correct for AC stark shift [Hz]."
          },
          "shape": {
            "type": "string",
            "description": "Shape of the gate"
          },
          "angle2volt": {
            "type": "object",
            "title": "angle2volt",
            "properties": {
              "deg90": {
                "type": "number"
              },
              "deg180": {
                "type": "number"
              }
            },
            "required": [
              "deg90",
              "deg180"
            ]
          }
        },
        "required": [
          "length",
          "sigma",
          "alpha",
          "detuning",
          "shape",
          "angle2volt"
        ]
      },
      "drag_cosine": {
        "type": "object",
        "title": "drag_cosine",
        "properties": {
          "length": {
            "type": "number",
            "description": "The pulse length [s]"
          },
          "alpha": {
            "type": "number",
            "description": "The DRAG coefficient alpha."
          },
          "detuning": {
            "type": "integer",
            "description": "The frequency shift to correct for AC stark shift [Hz]."
          },
          "shape": {
            "type": "string",
            "description": "Shape of the gate"
          },
          "angle2volt": {
            "type": "object",
            "title": "angle2volt",
            "properties": {
              "deg90": {
                "type": "number"
              },
              "deg180": {
                "type": "number"
              }
            },
            "required": [
              "deg90",
              "deg180"
            ]
          }
        },
        "required": [
          "length",
          "alpha",
          "detuning",
          "shape",
          "angle2volt"
        ]
      },
      "square": {
        "type": "object",
        "title": "square",
        "properties": {
          "length": {
            "type": "number",
            "description": "The pulse length [s]"
          },
          "shape": {
            "type": "string",
            "description": "Shape of the gate"
          },
          "angle2volt": {
            "type": "object",
            "title": "angle2volt",
            "properties": {
              "deg90": {
                "type": "number"
              },
              "deg180": {
                "type": "number"
              }
            },
            "required": [
              "deg90",
              "deg180"
            ]
          }
        },
        "required": [
          "length",
          "shape",
          "angle2volt"
        ]
      }
    },
    "required": [
      "drag_gaussian",
      "drag_cosine",
      "square"
    ]
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "drive_line_index": {
        "type": "integer",
        "description": "Index of the readout line connected to this qubit."
      },
      "correction_matrix": {
        "type": "object",
        "title": "correction_matrix",
        "properties": {
          "gain": {
            "type": "number"
          },
          "phase": {
            "type": "number"
          }
        },
        "required": [
          "gain",
          "phase"
        ]
      },
      "maximum_amplitude": {
        "type": "number",
        "description": "max amplitude in volts above which the mixer will send higher harmonics."
      },
      "analog_channel_offset": {
        "type": "number",
        "description": "Voltage value to nullify inheret analog channel offset [V]"
      },
      "charge_line": {
        "type": "object",
        "title": "charge_line",
        "properties": {
          "controller": {
            "type": "string"
          },
          "channel": {
            "type": "integer"
          },
          "offset": {
            "type": "number"
          }
        },
        "required": [
          "controller",
          "channel",
          "offset"
        ]
      },
      "charge_filter_coefficients": {
        "type": "object",
        "title": "charge_filter_coefficients",
        "properties": {
          "feedforward": {
            "type": "array",
            "items": {
              "anyOf": [
                {
                  "type": "integer"
                },
                {
                  "type": "number"
                },
                {
                  "type": "string"
                },
                {
                  "type": "boolean"
                },
                {
                  "type": "array"
                }
              ]
            }
          },
          "feedback": {
            "type": "array",
            "items": {
              "anyOf": [
                {
                  "type": "integer"
                },
                {
                  "type": "number"
                },
                {
                  "type": "string"
                },
                {
                  "type": "boolean"
                },
                {
                  "type": "array"
                }
              ]
            }
          }
        },
        "required": [
          "feedforward",
          "feedback"
        ]
      },
      "switch_delay": {
        "type": "integer",
        "description": "delay of digital pulse"
      },
      "switch_buffer": {
        "type": "integer",
        "description": "buffer of digital pulse"
      }
    },
    "required": [
      "drive_line_index",
      "correction_matrix",
      "maximum_amplitude",
      "analog_channel_offset",
      "charge_line",
      "charge_filter_coefficients",
      "switch_delay",
      "switch_buffer"
    ]
  },
  "threads": {
    "type": "object",
    "title": "threads",
    "properties": {
      "thread_cond": {
        "type": "boolean",
        "description": "Boolean to decided to put manual threds or not"
      },
      "thread": {
        "type": "string",
        "description": "Manual thread being allocated"
      }
    },
    "required": [
      "thread_cond",
      "thread"
    ]
  },
  "charge_bias_points": {
    "type": "array",
    "items": {
      "type": "object",
      "title": "charge_bias_point",
      "properties": {
        "name": {
          "type": "string"
        },
        "value": {
          "type": "number",
          "description": "Bias voltage to set qubit to degeneracy between even and odd parity [V]"
        }
      },
      "required": [
        "name",
        "value"
      ]
    }
  },
  "sequence_states": {
    "type": "object",
    "title": "sequence_states",
    "properties": {
      "constant": {
        "type": "array",
        "items": {
          "type": "object",
          "title": "constant",
          "properties": {
            "name": {
              "type": "string"
            },
            "amplitude": {
              "type": "number",
              "description": "[V]"
            },
            "length": {
              "type": "number",
              "description": "[s]"
            }
          },
          "required": [
            "name",
            "amplitude",
            "length"
          ]
        }
      },
      "arbitrary": {
        "type": "array",
        "items": {
          "type": "object",
          "title": "arbitrary",
          "properties": {
            "name": {
              "type": "string"
            },
            "waveform": {
              "type": "array",
              "items": {
                "anyOf": [
                  {
                    "type": "integer"
                  },
                  {
                    "type": "number"
                  },
                  {
                    "type": "string"
                  },
                  {
                    "type": "boolean"
                  },
                  {
                    "type": "array"
                  }
                ]
              },
              "description": "points describing the waveform shape"
            }
          },
          "required": [
            "name",
            "waveform"
          ]
        }
      }
    },
    "required": [
      "constant",
      "arbitrary"
    ]
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Angle2volt4(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def deg90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg90.setter
    def deg90(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg90"] = value

    @property
    def deg180(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg180"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg180.setter
    def deg180(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg180")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg180"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg180"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Drag_gaussian2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def length(self) -> float:
        """The pulse length [s]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """The pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def sigma(self) -> float:
        """The gaussian standard deviation (only for gaussian pulses) [s]"""
        
        value = self._quam._json[self._path + "sigma"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @sigma.setter
    def sigma(self, value: float):
        """The gaussian standard deviation (only for gaussian pulses) [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "sigma")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "sigma"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "sigma"] = value

    @property
    def alpha(self) -> float:
        """The DRAG coefficient alpha."""
        
        value = self._quam._json[self._path + "alpha"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @alpha.setter
    def alpha(self, value: float):
        """The DRAG coefficient alpha."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "alpha")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "alpha"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "alpha"] = value

    @property
    def detuning(self) -> int:
        """The frequency shift to correct for AC stark shift [Hz]."""
        
        value = self._quam._json[self._path + "detuning"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @detuning.setter
    def detuning(self, value: int):
        """The frequency shift to correct for AC stark shift [Hz]."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "detuning")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "detuning"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "detuning"] = value

    @property
    def shape(self) -> str:
        """Shape of the gate"""
        
        value = self._quam._json[self._path + "shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @shape.setter
    def shape(self, value: str):
        """Shape of the gate"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "shape")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "shape"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "shape"] = value

    @property
    def angle2volt(self) -> Angle2volt4:
        """Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V."""
        return Angle2volt4(
            self._quam, self._path + "angle2volt/", self._index,
            self._schema["properties"]["angle2volt"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Angle2volt5(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def deg90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg90.setter
    def deg90(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg90"] = value

    @property
    def deg180(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg180"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg180.setter
    def deg180(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg180")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg180"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg180"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Drag_cosine2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def length(self) -> float:
        """The pulse length [s]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """The pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def alpha(self) -> float:
        """The DRAG coefficient alpha."""
        
        value = self._quam._json[self._path + "alpha"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @alpha.setter
    def alpha(self, value: float):
        """The DRAG coefficient alpha."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "alpha")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "alpha"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "alpha"] = value

    @property
    def detuning(self) -> int:
        """The frequency shift to correct for AC stark shift [Hz]."""
        
        value = self._quam._json[self._path + "detuning"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @detuning.setter
    def detuning(self, value: int):
        """The frequency shift to correct for AC stark shift [Hz]."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "detuning")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "detuning"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "detuning"] = value

    @property
    def shape(self) -> str:
        """Shape of the gate"""
        
        value = self._quam._json[self._path + "shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @shape.setter
    def shape(self, value: str):
        """Shape of the gate"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "shape")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "shape"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "shape"] = value

    @property
    def angle2volt(self) -> Angle2volt5:
        """Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V."""
        return Angle2volt5(
            self._quam, self._path + "angle2volt/", self._index,
            self._schema["properties"]["angle2volt"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Angle2volt6(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def deg90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg90.setter
    def deg90(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg90"] = value

    @property
    def deg180(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "deg180"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @deg180.setter
    def deg180(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "deg180")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "deg180"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "deg180"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Square2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def length(self) -> float:
        """The pulse length [s]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """The pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def shape(self) -> str:
        """Shape of the gate"""
        
        value = self._quam._json[self._path + "shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @shape.setter
    def shape(self, value: str):
        """Shape of the gate"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "shape")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "shape"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "shape"] = value

    @property
    def angle2volt(self) -> Angle2volt6:
        """Rotation angle (on the Bloch sphere) to voltage amplitude conversion, must be within [-0.5, 0.5) V. For instance 'deg180':0.2 will lead to a pi pulse of 0.2 V."""
        return Angle2volt6(
            self._quam, self._path + "angle2volt/", self._index,
            self._schema["properties"]["angle2volt"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Driving2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def drag_gaussian(self) -> Drag_gaussian2:
        """"""
        return Drag_gaussian2(
            self._quam, self._path + "drag_gaussian/", self._index,
            self._schema["properties"]["drag_gaussian"]
        )

    @property
    def drag_cosine(self) -> Drag_cosine2:
        """"""
        return Drag_cosine2(
            self._quam, self._path + "drag_cosine/", self._index,
            self._schema["properties"]["drag_cosine"]
        )

    @property
    def square(self) -> Square2:
        """"""
        return Square2(
            self._quam, self._path + "square/", self._index,
            self._schema["properties"]["square"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Correction_matrix3(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Wiring3(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def drive_line_index(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "drive_line_index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @drive_line_index.setter
    def drive_line_index(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "drive_line_index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "drive_line_index"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "drive_line_index"] = value

    @property
    def maximum_amplitude(self) -> float:
        """max amplitude in volts above which the mixer will send higher harmonics."""
        
        value = self._quam._json[self._path + "maximum_amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @maximum_amplitude.setter
    def maximum_amplitude(self, value: float):
        """max amplitude in volts above which the mixer will send higher harmonics."""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "maximum_amplitude")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "maximum_amplitude"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "maximum_amplitude"] = value

    @property
    def switch_delay(self) -> int:
        """delay of digital pulse"""
        
        value = self._quam._json[self._path + "switch_delay"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @switch_delay.setter
    def switch_delay(self, value: int):
        """delay of digital pulse"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "switch_delay")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "switch_delay"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "switch_delay"] = value

    @property
    def switch_buffer(self) -> int:
        """buffer of digital pulse"""
        
        value = self._quam._json[self._path + "switch_buffer"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @switch_buffer.setter
    def switch_buffer(self, value: int):
        """buffer of digital pulse"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "switch_buffer")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "switch_buffer"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "switch_buffer"] = value

    @property
    def correction_matrix(self) -> Correction_matrix3:
        """"""
        return Correction_matrix3(
            self._quam, self._path + "correction_matrix/", self._index,
            self._schema["properties"]["correction_matrix"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Threads3(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def thread_cond(self) -> bool:
        """Boolean to decided to put manual threds or not"""
        
        value = self._quam._json[self._path + "thread_cond"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @thread_cond.setter
    def thread_cond(self, value: bool):
        """Boolean to decided to put manual threds or not"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "thread_cond")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "thread_cond"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "thread_cond"] = value

    @property
    def thread(self) -> str:
        """Manual thread being allocated"""
        
        value = self._quam._json[self._path + "thread"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @thread.setter
    def thread(self, value: str):
        """Manual thread being allocated"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "thread")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "thread"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "thread"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Qubits_wo_charge(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def index(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @index.setter
    def index(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "index"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "index"] = value

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def f_01(self) -> float:
        """0-1 transition frequency [Hz]"""
        
        value = self._quam._json[self._path + "f_01"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f_01.setter
    def f_01(self, value: float):
        """0-1 transition frequency [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "f_01")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "f_01"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f_01"] = value

    @property
    def df(self) -> float:
        """Half of charge dispersion measured in spectroscopy [Hz]"""
        
        value = self._quam._json[self._path + "df"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @df.setter
    def df(self, value: float):
        """Half of charge dispersion measured in spectroscopy [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "df")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "df"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "df"] = value

    @property
    def anharmonicity(self) -> float:
        """Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]"""
        
        value = self._quam._json[self._path + "anharmonicity"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @anharmonicity.setter
    def anharmonicity(self, value: float):
        """Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "anharmonicity")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "anharmonicity"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "anharmonicity"] = value

    @property
    def rabi_freq(self) -> int:
        """Qubit Rabi frequency [Hz]"""
        
        value = self._quam._json[self._path + "rabi_freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rabi_freq.setter
    def rabi_freq(self, value: int):
        """Qubit Rabi frequency [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "rabi_freq")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "rabi_freq"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "rabi_freq"] = value

    @property
    def t1(self) -> float:
        """Relaxation time T1 [s]"""
        
        value = self._quam._json[self._path + "t1"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t1.setter
    def t1(self, value: float):
        """Relaxation time T1 [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "t1")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "t1"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "t1"] = value

    @property
    def t2(self) -> float:
        """Dephasing time T2 [s]"""
        
        value = self._quam._json[self._path + "t2"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t2.setter
    def t2(self, value: float):
        """Dephasing time T2 [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "t2")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "t2"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "t2"] = value

    @property
    def t2star(self) -> float:
        """Dephasing time T2* [s]"""
        
        value = self._quam._json[self._path + "t2star"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t2star.setter
    def t2star(self, value: float):
        """Dephasing time T2* [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "t2star")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "t2star"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "t2star"] = value

    @property
    def ramsey_det(self) -> float:
        """Detuning to observe ramsey fringes [Hz]"""
        
        value = self._quam._json[self._path + "ramsey_det"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ramsey_det.setter
    def ramsey_det(self, value: float):
        """Detuning to observe ramsey fringes [Hz]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ramsey_det")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ramsey_det"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ramsey_det"] = value

    @property
    def driving(self) -> Driving2:
        """"""
        return Driving2(
            self._quam, self._path + "driving/", self._index,
            self._schema["properties"]["driving"]
        )

    @property
    def wiring(self) -> Wiring3:
        """"""
        return Wiring3(
            self._quam, self._path + "wiring/", self._index,
            self._schema["properties"]["wiring"]
        )

    @property
    def threads(self) -> Threads3:
        """"""
        return Threads3(
            self._quam, self._path + "threads/", self._index,
            self._schema["properties"]["threads"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Qubits_wo_chargeList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Qubits_wo_charge:
        return Qubits_wo_charge(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new qubits_wo_charge by adding a JSON dictionary with following schema
{
  "index": {
    "type": "integer"
  },
  "name": {
    "type": "string"
  },
  "f_01": {
    "type": "number",
    "description": "0-1 transition frequency [Hz]"
  },
  "df": {
    "type": "number",
    "description": "Half of charge dispersion measured in spectroscopy [Hz]"
  },
  "anharmonicity": {
    "type": "number",
    "description": "Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]"
  },
  "rabi_freq": {
    "type": "integer",
    "description": "Qubit Rabi frequency [Hz]"
  },
  "t1": {
    "type": "number",
    "description": "Relaxation time T1 [s]"
  },
  "t2": {
    "type": "number",
    "description": "Dephasing time T2 [s]"
  },
  "t2star": {
    "type": "number",
    "description": "Dephasing time T2* [s]"
  },
  "ramsey_det": {
    "type": "number",
    "description": "Detuning to observe ramsey fringes [Hz]"
  },
  "driving": {
    "type": "object",
    "title": "driving",
    "properties": {
      "drag_gaussian": {
        "type": "object",
        "title": "drag_gaussian",
        "properties": {
          "length": {
            "type": "number",
            "description": "The pulse length [s]"
          },
          "sigma": {
            "type": "number",
            "description": "The gaussian standard deviation (only for gaussian pulses) [s]"
          },
          "alpha": {
            "type": "number",
            "description": "The DRAG coefficient alpha."
          },
          "detuning": {
            "type": "integer",
            "description": "The frequency shift to correct for AC stark shift [Hz]."
          },
          "shape": {
            "type": "string",
            "description": "Shape of the gate"
          },
          "angle2volt": {
            "type": "object",
            "title": "angle2volt",
            "properties": {
              "deg90": {
                "type": "number"
              },
              "deg180": {
                "type": "number"
              }
            },
            "required": [
              "deg90",
              "deg180"
            ]
          }
        },
        "required": [
          "length",
          "sigma",
          "alpha",
          "detuning",
          "shape",
          "angle2volt"
        ]
      },
      "drag_cosine": {
        "type": "object",
        "title": "drag_cosine",
        "properties": {
          "length": {
            "type": "number",
            "description": "The pulse length [s]"
          },
          "alpha": {
            "type": "number",
            "description": "The DRAG coefficient alpha."
          },
          "detuning": {
            "type": "integer",
            "description": "The frequency shift to correct for AC stark shift [Hz]."
          },
          "shape": {
            "type": "string",
            "description": "Shape of the gate"
          },
          "angle2volt": {
            "type": "object",
            "title": "angle2volt",
            "properties": {
              "deg90": {
                "type": "number"
              },
              "deg180": {
                "type": "number"
              }
            },
            "required": [
              "deg90",
              "deg180"
            ]
          }
        },
        "required": [
          "length",
          "alpha",
          "detuning",
          "shape",
          "angle2volt"
        ]
      },
      "square": {
        "type": "object",
        "title": "square",
        "properties": {
          "length": {
            "type": "number",
            "description": "The pulse length [s]"
          },
          "shape": {
            "type": "string",
            "description": "Shape of the gate"
          },
          "angle2volt": {
            "type": "object",
            "title": "angle2volt",
            "properties": {
              "deg90": {
                "type": "number"
              },
              "deg180": {
                "type": "number"
              }
            },
            "required": [
              "deg90",
              "deg180"
            ]
          }
        },
        "required": [
          "length",
          "shape",
          "angle2volt"
        ]
      }
    },
    "required": [
      "drag_gaussian",
      "drag_cosine",
      "square"
    ]
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "drive_line_index": {
        "type": "integer"
      },
      "correction_matrix": {
        "type": "object",
        "title": "correction_matrix",
        "properties": {
          "gain": {
            "type": "number"
          },
          "phase": {
            "type": "number"
          }
        },
        "required": [
          "gain",
          "phase"
        ]
      },
      "maximum_amplitude": {
        "type": "number",
        "description": "max amplitude in volts above which the mixer will send higher harmonics."
      },
      "switch_delay": {
        "type": "integer",
        "description": "delay of digital pulse"
      },
      "switch_buffer": {
        "type": "integer",
        "description": "buffer of digital pulse"
      }
    },
    "required": [
      "drive_line_index",
      "correction_matrix",
      "maximum_amplitude",
      "switch_delay",
      "switch_buffer"
    ]
  },
  "threads": {
    "type": "object",
    "title": "threads",
    "properties": {
      "thread_cond": {
        "type": "boolean",
        "description": "Boolean to decided to put manual threds or not"
      },
      "thread": {
        "type": "string",
        "description": "Manual thread being allocated"
      }
    },
    "required": [
      "thread_cond",
      "thread"
    ]
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Crosstalk_matrix(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def static(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "static",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "static"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @static.setter
    def static(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "static")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "static"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "static"] = value

    @property
    def fast(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "fast",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "fast"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @fast.setter
    def fast(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "fast")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "fast"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "fast"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Single_qubit_operation(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def direction(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "direction"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @direction.setter
    def direction(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "direction")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "direction"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "direction"] = value

    @property
    def angle(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "angle"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @angle.setter
    def angle(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "angle")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "angle"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "angle"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Single_qubit_operationsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Single_qubit_operation:
        return Single_qubit_operation(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new single_qubit_operation by adding a JSON dictionary with following schema
{
  "direction": {
    "type": "string"
  },
  "angle": {
    "type": "integer"
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Injector_line(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def controller(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controller.setter
    def controller(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def channel(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "channel"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @channel.setter
    def channel(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "channel")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "channel"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "channel"] = value

    @property
    def offset(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @offset.setter
    def offset(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "offset"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Wiring4(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def injector_line(self) -> Injector_line:
        """"""
        return Injector_line(
            self._quam, self._path + "injector_line/", self._index,
            self._schema["properties"]["injector_line"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Qp_injector(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def index(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @index.setter
    def index(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "index"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "index"] = value

    @property
    def name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @name.setter
    def name(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def energy_gap(self) -> float:
        """Superconducting energy gap [V]"""
        
        value = self._quam._json[self._path + "energy_gap"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @energy_gap.setter
    def energy_gap(self, value: float):
        """Superconducting energy gap [V]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "energy_gap")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "energy_gap"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "energy_gap"] = value

    @property
    def injection_voltage(self) -> float:
        """Injection pulse voltage for phonon injection experiment [V]"""
        
        value = self._quam._json[self._path + "injection_voltage"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @injection_voltage.setter
    def injection_voltage(self, value: float):
        """Injection pulse voltage for phonon injection experiment [V]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "injection_voltage")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "injection_voltage"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "injection_voltage"] = value

    @property
    def injection_length(self) -> float:
        """Injection pulse lenght for phonon injection experiment [s]"""
        
        value = self._quam._json[self._path + "injection_length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @injection_length.setter
    def injection_length(self, value: float):
        """Injection pulse lenght for phonon injection experiment [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "injection_length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "injection_length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "injection_length"] = value

    @property
    def analog_channel_offset(self) -> float:
        """Voltage value to nullify inheret analog channel offset [V]"""
        
        value = self._quam._json[self._path + "analog_channel_offset"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @analog_channel_offset.setter
    def analog_channel_offset(self, value: float):
        """Voltage value to nullify inheret analog channel offset [V]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "analog_channel_offset")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "analog_channel_offset"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "analog_channel_offset"] = value

    @property
    def wiring(self) -> Wiring4:
        """"""
        return Wiring4(
            self._quam, self._path + "wiring/", self._index,
            self._schema["properties"]["wiring"]
        )

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Qp_injectorsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Qp_injector:
        return Qp_injector(
            self._quam,
            self._path + "[]/",
            self._index + [key],
            self._schema["items"]
        )

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
        """Adds a new qp_injector by adding a JSON dictionary with following schema
{
  "index": {
    "type": "integer"
  },
  "name": {
    "type": "string"
  },
  "energy_gap": {
    "type": "number",
    "description": "Superconducting energy gap [V]"
  },
  "injection_voltage": {
    "type": "number",
    "description": "Injection pulse voltage for phonon injection experiment [V]"
  },
  "injection_length": {
    "type": "number",
    "description": "Injection pulse lenght for phonon injection experiment [s]"
  },
  "analog_channel_offset": {
    "type": "number",
    "description": "Voltage value to nullify inheret analog channel offset [V]"
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "injector_line": {
        "type": "object",
        "title": "injector_line",
        "properties": {
          "controller": {
            "type": "string"
          },
          "channel": {
            "type": "integer"
          },
          "offset": {
            "type": "number"
          }
        },
        "required": [
          "controller",
          "channel",
          "offset"
        ]
      }
    },
    "required": [
      "injector_line"
    ]
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
        self._quam._json[f"{self._path}[]_len"] += 1

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

class Results(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def directory(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "directory"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @directory.setter
    def directory(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "directory")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "directory"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "directory"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Running_strategy(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def running(self) -> bool:
        """"""
        
        value = self._quam._json[self._path + "running"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @running.setter
    def running(self, value: bool):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "running")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "running"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "running"] = value

    @property
    def start(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "start",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "start"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @start.setter
    def start(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "start")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "start"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "start"] = value

    @property
    def end(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "end",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "end"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @end.setter
    def end(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "end")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "end"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "end"] = value

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class QuAM(object):

    def __init__(self, flat_json: Union[None, str, dict] = None):
        """"""
        self._quam = self
        self._path = ""
        self._index = []
        self._record_updates = False
        if flat_json is not None:
            if type(flat_json) is str:
                with open(flat_json, "r") as file:
                    flat_json = json.load(file)
            self._json = flat_json      # initial json
        else:
            self._json = None
        self._updates = {"keys":[], "indexes":[], "values":[], "items":[]}
        self._schema_flat = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'name': 'QuAM storage format', 'description': 'optimized data structure for communication and storage', 'type': 'object', 'properties': {'digital_waveforms[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'readout_lines[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'readout_resonators[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'readout_resonators[]/integration_weights[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'drive_lines[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]/charge_bias_points[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]/sequence_states/constant[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]/sequence_states/arbitrary[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits_wo_charge[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'single_qubit_operations[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qp_injectors[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}}, 'additionalProperties': False}
        self._schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'title': 'QuAM', 'properties': {'network': {'type': 'object', 'title': 'network', 'properties': {'qop_ip': {'type': 'string'}, 'port': {'type': 'integer'}}, 'required': ['qop_ip', 'port']}, 'controllers': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'digital_waveforms': {'type': 'array', 'items': {'type': 'object', 'title': 'digital_waveform', 'properties': {'name': {'type': 'string'}, 'samples': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['name', 'samples']}}, 'common_operation': {'type': 'object', 'title': 'common_operation', 'description': 'an operation which is common to all elements', 'properties': {'name': {'type': 'string'}, 'duration': {'type': 'number', 'description': 'pulse length [s]'}, 'amplitude': {'type': 'number', 'description': 'pulse amplitude [V]'}}, 'required': ['name', 'duration', 'amplitude']}, 'readout_lines': {'type': 'array', 'items': {'type': 'object', 'title': 'readout_line', 'properties': {'length': {'type': 'number', 'description': 'readout time on this readout line [s]'}, 'lo_freq': {'type': 'number', 'description': 'LO frequency for readout line [Hz]'}, 'lo_power': {'type': 'integer', 'description': 'LO power for readout line [dBm]'}, 'I_up': {'type': 'object', 'title': 'I_up', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'Q_up': {'type': 'object', 'title': 'Q_up', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'I_down': {'type': 'object', 'title': 'I_down', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}, 'gain_db': {'type': 'integer'}}, 'required': ['controller', 'channel', 'offset', 'gain_db']}, 'switch': {'type': 'object', 'title': 'switch', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}}, 'required': ['controller', 'channel']}}, 'required': ['length', 'lo_freq', 'lo_power', 'I_up', 'Q_up', 'I_down', 'switch']}}, 'readout_resonators': {'type': 'array', 'items': {'type': 'object', 'title': 'readout_resonator', 'properties': {'index': {'type': 'integer'}, 'name': {'type': 'string'}, 'f_res': {'type': 'number', 'description': 'Resonator resonance frequency [Hz].'}, 'f_opt': {'type': 'number', 'description': 'Resonator optimal readout frequency [Hz] (used in QUA).'}, 'readout_regime': {'type': 'string'}, 'readout_amplitude': {'type': 'number', 'description': 'Readout amplitude for this resonator [V]. Must be within [-0.5, 0.5).'}, 'rotation_angle': {'type': 'number', 'description': "Angle by which to rotate the IQ blobs to place the separation along the 'I' quadrature [degrees]."}, 'integration_weights': {'type': 'array', 'items': {'type': 'object', 'title': 'integration_weight', 'properties': {'name': {'type': 'string'}, 'cosine': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'sine': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['name', 'cosine', 'sine']}, 'description': 'Arbitrary integration weights defined as lists of tuples whose first element is the value of the integration weight and second element is the duration in ns for which this value should be used [(1.0, readout_len)]. The duration must be divisible by 4.'}, 'ge_threshold': {'type': 'number', 'description': "Threshold (in demod unit) along the 'I' quadrature discriminating between qubit ground and excited states."}, 'readout_fidelity': {'type': 'number'}, 'q_factor': {'type': 'number'}, 'chi': {'type': 'number'}, 'relaxation_time': {'type': 'number', 'description': 'Resonator relaxation time [s].'}, 'f_res_vs_charge': {'type': 'object', 'title': 'f_res_vs_charge', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}, 'c': {'type': 'number'}}, 'required': ['a', 'b', 'c']}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'readout_line_index': {'type': 'integer', 'description': 'Index of the readout line connected to this resonator.'}, 'time_of_flight': {'type': 'integer', 'description': 'Time of flight for this resonator [ns].'}, 'correction_matrix': {'type': 'object', 'title': 'correction_matrix', 'properties': {'gain': {'type': 'number'}, 'phase': {'type': 'number'}}, 'required': ['gain', 'phase']}, 'maximum_amplitude': {'type': 'number', 'description': 'max amplitude in volts above which the mixer will send higher harmonics.'}, 'switch_delay': {'type': 'integer', 'description': 'delay of digital pulse'}, 'switch_buffer': {'type': 'integer', 'description': 'buffer of digital pulse'}}, 'required': ['readout_line_index', 'time_of_flight', 'correction_matrix', 'maximum_amplitude', 'switch_delay', 'switch_buffer']}, 'threads': {'type': 'object', 'title': 'threads', 'properties': {'thread_cond': {'type': 'boolean', 'description': 'Boolean to decided to put manual threds or not'}, 'thread': {'type': 'string', 'description': 'Manual thread being allocated'}}, 'required': ['thread_cond', 'thread']}}, 'required': ['index', 'name', 'f_res', 'f_opt', 'readout_regime', 'readout_amplitude', 'rotation_angle', 'integration_weights', 'ge_threshold', 'readout_fidelity', 'q_factor', 'chi', 'relaxation_time', 'f_res_vs_charge', 'wiring', 'threads']}}, 'drive_lines': {'type': 'array', 'items': {'type': 'object', 'title': 'drive_line', 'properties': {'qubits': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}, 'description': 'qubits associated with this drive line'}, 'lo_freq': {'type': 'number', 'description': 'LO frequency [Hz]'}, 'lo_power': {'type': 'integer', 'description': 'LO power to drive line [dBm]'}, 'I': {'type': 'object', 'title': 'I', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'Q': {'type': 'object', 'title': 'Q', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'switch': {'type': 'object', 'title': 'switch', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}}, 'required': ['controller', 'channel']}}, 'required': ['qubits', 'lo_freq', 'lo_power', 'I', 'Q', 'switch']}}, 'qubits': {'type': 'array', 'items': {'type': 'object', 'title': 'qubit', 'properties': {'index': {'type': 'integer'}, 'name': {'type': 'string'}, 'f_01': {'type': 'number', 'description': '0-1 transition frequency [Hz]'}, 'df': {'type': 'number', 'description': 'Half of charge dispersion measured in spectroscopy [Hz]'}, 'anharmonicity': {'type': 'number', 'description': 'Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]'}, 'rabi_freq': {'type': 'integer', 'description': 'Qubit Rabi frequency [Hz]'}, 't1': {'type': 'number', 'description': 'Relaxation time T1 [s]'}, 't2': {'type': 'number', 'description': 'Dephasing time T2 [s]'}, 't2star': {'type': 'number', 'description': 'Dephasing time T2* [s]'}, 'ramsey_det': {'type': 'number', 'description': 'Detuning to observe ramsey fringes [Hz]'}, 'driving': {'type': 'object', 'title': 'driving', 'properties': {'drag_gaussian': {'type': 'object', 'title': 'drag_gaussian', 'properties': {'length': {'type': 'number', 'description': 'The pulse length [s]'}, 'sigma': {'type': 'number', 'description': 'The gaussian standard deviation (only for gaussian pulses) [s]'}, 'alpha': {'type': 'number', 'description': 'The DRAG coefficient alpha.'}, 'detuning': {'type': 'integer', 'description': 'The frequency shift to correct for AC stark shift [Hz].'}, 'shape': {'type': 'string', 'description': 'Shape of the gate'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['length', 'sigma', 'alpha', 'detuning', 'shape', 'angle2volt']}, 'drag_cosine': {'type': 'object', 'title': 'drag_cosine', 'properties': {'length': {'type': 'number', 'description': 'The pulse length [s]'}, 'alpha': {'type': 'number', 'description': 'The DRAG coefficient alpha.'}, 'detuning': {'type': 'integer', 'description': 'The frequency shift to correct for AC stark shift [Hz].'}, 'shape': {'type': 'string', 'description': 'Shape of the gate'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['length', 'alpha', 'detuning', 'shape', 'angle2volt']}, 'square': {'type': 'object', 'title': 'square', 'properties': {'length': {'type': 'number', 'description': 'The pulse length [s]'}, 'shape': {'type': 'string', 'description': 'Shape of the gate'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['length', 'shape', 'angle2volt']}}, 'required': ['drag_gaussian', 'drag_cosine', 'square']}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'drive_line_index': {'type': 'integer', 'description': 'Index of the readout line connected to this qubit.'}, 'correction_matrix': {'type': 'object', 'title': 'correction_matrix', 'properties': {'gain': {'type': 'number'}, 'phase': {'type': 'number'}}, 'required': ['gain', 'phase']}, 'maximum_amplitude': {'type': 'number', 'description': 'max amplitude in volts above which the mixer will send higher harmonics.'}, 'analog_channel_offset': {'type': 'number', 'description': 'Voltage value to nullify inheret analog channel offset [V]'}, 'charge_line': {'type': 'object', 'title': 'charge_line', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'charge_filter_coefficients': {'type': 'object', 'title': 'charge_filter_coefficients', 'properties': {'feedforward': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'feedback': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['feedforward', 'feedback']}, 'switch_delay': {'type': 'integer', 'description': 'delay of digital pulse'}, 'switch_buffer': {'type': 'integer', 'description': 'buffer of digital pulse'}}, 'required': ['drive_line_index', 'correction_matrix', 'maximum_amplitude', 'analog_channel_offset', 'charge_line', 'charge_filter_coefficients', 'switch_delay', 'switch_buffer']}, 'threads': {'type': 'object', 'title': 'threads', 'properties': {'thread_cond': {'type': 'boolean', 'description': 'Boolean to decided to put manual threds or not'}, 'thread': {'type': 'string', 'description': 'Manual thread being allocated'}}, 'required': ['thread_cond', 'thread']}, 'charge_bias_points': {'type': 'array', 'items': {'type': 'object', 'title': 'charge_bias_point', 'properties': {'name': {'type': 'string'}, 'value': {'type': 'number', 'description': 'Bias voltage to set qubit to degeneracy between even and odd parity [V]'}}, 'required': ['name', 'value']}}, 'sequence_states': {'type': 'object', 'title': 'sequence_states', 'properties': {'constant': {'type': 'array', 'items': {'type': 'object', 'title': 'constant', 'properties': {'name': {'type': 'string'}, 'amplitude': {'type': 'number', 'description': '[V]'}, 'length': {'type': 'number', 'description': '[s]'}}, 'required': ['name', 'amplitude', 'length']}}, 'arbitrary': {'type': 'array', 'items': {'type': 'object', 'title': 'arbitrary', 'properties': {'name': {'type': 'string'}, 'waveform': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}, 'description': 'points describing the waveform shape'}}, 'required': ['name', 'waveform']}}}, 'required': ['constant', 'arbitrary']}}, 'required': ['index', 'name', 'f_01', 'df', 'anharmonicity', 'rabi_freq', 't1', 't2', 't2star', 'ramsey_det', 'driving', 'wiring', 'threads', 'charge_bias_points', 'sequence_states']}}, 'qubits_wo_charge': {'type': 'array', 'items': {'type': 'object', 'title': 'qubits_wo_charge', 'properties': {'index': {'type': 'integer'}, 'name': {'type': 'string'}, 'f_01': {'type': 'number', 'description': '0-1 transition frequency [Hz]'}, 'df': {'type': 'number', 'description': 'Half of charge dispersion measured in spectroscopy [Hz]'}, 'anharmonicity': {'type': 'number', 'description': 'Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]'}, 'rabi_freq': {'type': 'integer', 'description': 'Qubit Rabi frequency [Hz]'}, 't1': {'type': 'number', 'description': 'Relaxation time T1 [s]'}, 't2': {'type': 'number', 'description': 'Dephasing time T2 [s]'}, 't2star': {'type': 'number', 'description': 'Dephasing time T2* [s]'}, 'ramsey_det': {'type': 'number', 'description': 'Detuning to observe ramsey fringes [Hz]'}, 'driving': {'type': 'object', 'title': 'driving', 'properties': {'drag_gaussian': {'type': 'object', 'title': 'drag_gaussian', 'properties': {'length': {'type': 'number', 'description': 'The pulse length [s]'}, 'sigma': {'type': 'number', 'description': 'The gaussian standard deviation (only for gaussian pulses) [s]'}, 'alpha': {'type': 'number', 'description': 'The DRAG coefficient alpha.'}, 'detuning': {'type': 'integer', 'description': 'The frequency shift to correct for AC stark shift [Hz].'}, 'shape': {'type': 'string', 'description': 'Shape of the gate'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['length', 'sigma', 'alpha', 'detuning', 'shape', 'angle2volt']}, 'drag_cosine': {'type': 'object', 'title': 'drag_cosine', 'properties': {'length': {'type': 'number', 'description': 'The pulse length [s]'}, 'alpha': {'type': 'number', 'description': 'The DRAG coefficient alpha.'}, 'detuning': {'type': 'integer', 'description': 'The frequency shift to correct for AC stark shift [Hz].'}, 'shape': {'type': 'string', 'description': 'Shape of the gate'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['length', 'alpha', 'detuning', 'shape', 'angle2volt']}, 'square': {'type': 'object', 'title': 'square', 'properties': {'length': {'type': 'number', 'description': 'The pulse length [s]'}, 'shape': {'type': 'string', 'description': 'Shape of the gate'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['length', 'shape', 'angle2volt']}}, 'required': ['drag_gaussian', 'drag_cosine', 'square']}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'drive_line_index': {'type': 'integer'}, 'correction_matrix': {'type': 'object', 'title': 'correction_matrix', 'properties': {'gain': {'type': 'number'}, 'phase': {'type': 'number'}}, 'required': ['gain', 'phase']}, 'maximum_amplitude': {'type': 'number', 'description': 'max amplitude in volts above which the mixer will send higher harmonics.'}, 'switch_delay': {'type': 'integer', 'description': 'delay of digital pulse'}, 'switch_buffer': {'type': 'integer', 'description': 'buffer of digital pulse'}}, 'required': ['drive_line_index', 'correction_matrix', 'maximum_amplitude', 'switch_delay', 'switch_buffer']}, 'threads': {'type': 'object', 'title': 'threads', 'properties': {'thread_cond': {'type': 'boolean', 'description': 'Boolean to decided to put manual threds or not'}, 'thread': {'type': 'string', 'description': 'Manual thread being allocated'}}, 'required': ['thread_cond', 'thread']}}, 'required': ['index', 'name', 'f_01', 'df', 'anharmonicity', 'rabi_freq', 't1', 't2', 't2star', 'ramsey_det', 'driving', 'wiring', 'threads']}}, 'crosstalk_matrix': {'type': 'object', 'title': 'crosstalk_matrix', 'properties': {'static': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'fast': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['static', 'fast']}, 'single_qubit_operations': {'type': 'array', 'items': {'type': 'object', 'title': 'single_qubit_operation', 'properties': {'direction': {'type': 'string'}, 'angle': {'type': 'integer'}}, 'required': ['direction', 'angle']}}, 'qp_injectors': {'type': 'array', 'items': {'type': 'object', 'title': 'qp_injector', 'properties': {'index': {'type': 'integer'}, 'name': {'type': 'string'}, 'energy_gap': {'type': 'number', 'description': 'Superconducting energy gap [V]'}, 'injection_voltage': {'type': 'number', 'description': 'Injection pulse voltage for phonon injection experiment [V]'}, 'injection_length': {'type': 'number', 'description': 'Injection pulse lenght for phonon injection experiment [s]'}, 'analog_channel_offset': {'type': 'number', 'description': 'Voltage value to nullify inheret analog channel offset [V]'}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'injector_line': {'type': 'object', 'title': 'injector_line', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}}, 'required': ['injector_line']}}, 'required': ['index', 'name', 'energy_gap', 'injection_voltage', 'injection_length', 'analog_channel_offset', 'wiring']}}, 'results': {'type': 'object', 'title': 'results', 'properties': {'directory': {'type': 'string'}}, 'required': ['directory']}, 'running_strategy': {'type': 'object', 'title': 'running_strategy', 'properties': {'running': {'type': 'boolean'}, 'start': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'end': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['running', 'start', 'end']}}, 'required': ['network', 'controllers', 'digital_waveforms', 'common_operation', 'readout_lines', 'readout_resonators', 'drive_lines', 'qubits', 'qubits_wo_charge', 'crosstalk_matrix', 'single_qubit_operations', 'qp_injectors', 'results', 'running_strategy']}
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
                item[1] +"[]/", item[2], new_item=True
            )
            self._quam._json[f"{item[1]}[]_len"] += 1

        if self._record_updates:
            self._updates["keys"] += updates["keys"]
            self._updates["indexes"] += updates["indexes"]
            self._updates["values"] += updates["values"]
            self._updates["items"] += updates["items"]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._json)})"

    @property
    def controllers(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "controllers",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "controllers"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @controllers.setter
    def controllers(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controllers")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controllers"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controllers"] = value

    @property
    def digital_waveforms(self) -> Digital_waveformsList:
        """"""
        return Digital_waveformsList(
            self._quam, self._path + "digital_waveforms", self._index,
            self._schema["properties"]["digital_waveforms"]
        )

    @property
    def readout_lines(self) -> Readout_linesList:
        """"""
        return Readout_linesList(
            self._quam, self._path + "readout_lines", self._index,
            self._schema["properties"]["readout_lines"]
        )

    @property
    def readout_resonators(self) -> Readout_resonatorsList:
        """"""
        return Readout_resonatorsList(
            self._quam, self._path + "readout_resonators", self._index,
            self._schema["properties"]["readout_resonators"]
        )

    @property
    def drive_lines(self) -> Drive_linesList:
        """"""
        return Drive_linesList(
            self._quam, self._path + "drive_lines", self._index,
            self._schema["properties"]["drive_lines"]
        )

    @property
    def qubits(self) -> QubitsList:
        """"""
        return QubitsList(
            self._quam, self._path + "qubits", self._index,
            self._schema["properties"]["qubits"]
        )

    @property
    def qubits_wo_charge(self) -> Qubits_wo_chargeList:
        """"""
        return Qubits_wo_chargeList(
            self._quam, self._path + "qubits_wo_charge", self._index,
            self._schema["properties"]["qubits_wo_charge"]
        )

    @property
    def single_qubit_operations(self) -> Single_qubit_operationsList:
        """"""
        return Single_qubit_operationsList(
            self._quam, self._path + "single_qubit_operations", self._index,
            self._schema["properties"]["single_qubit_operations"]
        )

    @property
    def qp_injectors(self) -> Qp_injectorsList:
        """"""
        return Qp_injectorsList(
            self._quam, self._path + "qp_injectors", self._index,
            self._schema["properties"]["qp_injectors"]
        )

    @property
    def network(self) -> Network:
        """"""
        return Network(
            self._quam, self._path + "network/", self._index,
            self._schema["properties"]["network"]
        )

    @property
    def common_operation(self) -> Common_operation:
        """"""
        return Common_operation(
            self._quam, self._path + "common_operation/", self._index,
            self._schema["properties"]["common_operation"]
        )

    @property
    def crosstalk_matrix(self) -> Crosstalk_matrix:
        """"""
        return Crosstalk_matrix(
            self._quam, self._path + "crosstalk_matrix/", self._index,
            self._schema["properties"]["crosstalk_matrix"]
        )

    @property
    def results(self) -> Results:
        """"""
        return Results(
            self._quam, self._path + "results/", self._index,
            self._schema["properties"]["results"]
        )

    @property
    def running_strategy(self) -> Running_strategy:
        """"""
        return Running_strategy(
            self._quam, self._path + "running_strategy/", self._index,
            self._schema["properties"]["running_strategy"]
        )

    def build_config(self, digital_out: list, qubits: list, qubits_wo_charge: list, injector_list: list, shape: str):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.build_config(self, digital_out, qubits, qubits_wo_charge, injector_list, shape)

    def save(self, filename: str, reuse_existing_values: bool = False):
        """Saves quam data to file

    Args:
        filename (str): destination file name
        reuse_existing_values (bool, optional): if destination file exists, it will try
        to reuse key values from that file. Defaults to False.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.save(self, filename, reuse_existing_values)

    def save_results(self, filename, figures: List = ()):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.save_results(self, filename, figures)

    def get_wiring(self):
        """
    Print the state connectivity.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_wiring(self)

    def get_sequence_state(self, index: int, sequence_state: str = None):
        """
    Get the sequence state object of a given qubit.

    :param index: index of the qubit to be retrieved.
    :param sequence_state: name of the sequence.
    :return: the sequence state object. Print the list of sequence states if 'sequence_state' is None.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_sequence_state(self, index, sequence_state)

    def get_qubit(self, qubit_name: str):
        """
    Get the qubit object corresponding to the specified qubit name.

    :param qubit_name: name of the qubit to get.
    :return: the qubit object.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_qubit(self, qubit_name)

    def get_resonator(self, resonator_name: str):
        """
    Get the readout resonator object corresponding to the specified resonator name.

    :param resonator_name: name of the qubit to get.
    :return: the qubit object.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_resonator(self, resonator_name)

    def get_qubit_gate(self, index: int, shape: str):
        """
    Get the gate of a given qubit from its shape.

    :param index: index of the qubit to be retrieved.
    :param shape: name of the gate as defined under the qubit driving section.
    :return: the qubit gate object.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_qubit_gate(self, index, shape)

    def get_charge_bias_point(self, index: int, charge_bias_point: str = None):
        """
    Get the charge bias point for a given qubit.

    :param index: index of the qubit to be retrieved.
    :param charge_bias_point: name of the charge bias point.
    :return: charge bias point object. Print the list of charge bias point if 'charge_bias_point' is None.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_charge_bias_point(self, index, charge_bias_point)

    def get_readout_IF(self, index: int) -> float:
        """
    Get the intermediate frequency of the readout resonator specified by its index.

    :param index: index of the readout resonator to be retrieved.
    :return: the intermediate frequency in Hz.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_readout_IF(self, index)

    def get_qubit_IF(self, index: int) -> float:
        """
    Get the intermediate frequency of the qubit specified by its index.

    :param index: index of the qubit to be retrieved.
    :return: the intermediate frequency in Hz.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_qubit_IF(self, index)

    def set_f_res_vs_charge_vertex(self, index: int, three_points: List[tuple]):
        """
    Set the vertex corresponding to the resonator frequency vs charge parabola from three points.

    :param index: index of the readout resonator to be retrieved.
    :param three_points: list of three tuples corresponding to the three points located on the parabola to fit defined as [(x1, y1), (x2, y2), (x3, y3)].
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.set_f_res_vs_charge_vertex(self, index, three_points)

    def get_f_res_from_charge(self, index: int, charge_bias: float) -> float:
        """
    Get the resonance frequency of the specified readout resonator for a given charge bias.
    The vertex of the resonator frequency vs charge parabola must be set beforehand.

    :param index: index of the readout resonator to retrieve.
    :param charge_bias: value of the charge bias at which the resonance frequency will be derived.
    :return: the readout resonator resonance frequency corresponding to the specified charge bias.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_f_res_from_charge(self, index, charge_bias)

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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


