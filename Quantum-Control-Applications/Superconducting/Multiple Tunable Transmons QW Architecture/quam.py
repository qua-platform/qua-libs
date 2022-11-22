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


class Q_down(object):

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
    def Q_down(self) -> Q_down:
        """"""
        return Q_down(
            self._quam, self._path + "Q_down/", self._index,
            self._schema["properties"]["Q_down"]
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
  "Q_down": {
    "type": "object",
    "title": "Q_down",
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
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
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
    def opt_readout_frequency(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "opt_readout_frequency"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @opt_readout_frequency.setter
    def opt_readout_frequency(self, value: float):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "opt_readout_frequency")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "opt_readout_frequency"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "opt_readout_frequency"] = value

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
    def wiring(self) -> Wiring:
        """"""
        return Wiring(
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
  "ge_threshold": {
    "type": "number",
    "description": "Threshold (in demod unit) along the 'I' quadrature discriminating between qubit ground and excited states."
  },
  "opt_readout_frequency": {
    "type": "number"
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
      }
    },
    "required": [
      "readout_line_index",
      "time_of_flight",
      "correction_matrix"
    ]
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
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
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
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
    def gate_len(self) -> float:
        """The pulse length [s]"""
        
        value = self._quam._json[self._path + "gate_len"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gate_len.setter
    def gate_len(self, value: float):
        """The pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "gate_len")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "gate_len"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "gate_len"] = value

    @property
    def gate_sigma(self) -> float:
        """The gaussian standard deviation (only for gaussian pulses) [s]"""
        
        value = self._quam._json[self._path + "gate_sigma"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gate_sigma.setter
    def gate_sigma(self, value: float):
        """The gaussian standard deviation (only for gaussian pulses) [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "gate_sigma")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "gate_sigma"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "gate_sigma"] = value

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
    def gate_shape(self) -> str:
        """Shape of the gate"""
        
        value = self._quam._json[self._path + "gate_shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gate_shape.setter
    def gate_shape(self, value: str):
        """Shape of the gate"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "gate_shape")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "gate_shape"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "gate_shape"] = value

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
    def gate_len(self) -> float:
        """The pulse length [s]"""
        
        value = self._quam._json[self._path + "gate_len"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gate_len.setter
    def gate_len(self, value: float):
        """The pulse length [s]"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "gate_len")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "gate_len"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "gate_len"] = value

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
    def gate_shape(self) -> str:
        """Shape of the gate"""
        
        value = self._quam._json[self._path + "gate_shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gate_shape.setter
    def gate_shape(self, value: str):
        """Shape of the gate"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "gate_shape")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "gate_shape"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "gate_shape"] = value

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


class Flux_line(object):

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


class Flux_filter_coefficients(object):

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
    def correction_matrix(self) -> Correction_matrix2:
        """"""
        return Correction_matrix2(
            self._quam, self._path + "correction_matrix/", self._index,
            self._schema["properties"]["correction_matrix"]
        )

    @property
    def flux_line(self) -> Flux_line:
        """"""
        return Flux_line(
            self._quam, self._path + "flux_line/", self._index,
            self._schema["properties"]["flux_line"]
        )

    @property
    def flux_filter_coefficients(self) -> Flux_filter_coefficients:
        """"""
        return Flux_filter_coefficients(
            self._quam, self._path + "flux_filter_coefficients/", self._index,
            self._schema["properties"]["flux_filter_coefficients"]
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


class Flux_bias_point(object):

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
        """Bias voltage to set qubit to maximal frequency [V]"""
        
        value = self._quam._json[self._path + "value"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @value.setter
    def value(self, value: float):
        """Bias voltage to set qubit to maximal frequency [V]"""
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


class Flux_bias_pointsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Flux_bias_point:
        return Flux_bias_point(
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
        """Adds a new flux_bias_point by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "value": {
    "type": "number",
    "description": "Bias voltage to set qubit to maximal frequency [V]"
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
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
    def length(self) -> int:
        """[ns]"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: int):
        """[ns]"""
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
    "type": "integer",
    "description": "[ns]"
  }
}"""
        import quam_sdk.crud
        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path +"[]/", self._index, new_item=True)
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
    def amplitude(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @amplitude.setter
    def amplitude(self, value: float):
        """"""
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
    def length(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: int):
        """"""
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
  "amplitude": {
    "type": "number"
  },
  "length": {
    "type": "integer"
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
    def flux_bias_points(self) -> Flux_bias_pointsList:
        """"""
        return Flux_bias_pointsList(
            self._quam, self._path + "flux_bias_points", self._index,
            self._schema["properties"]["flux_bias_points"]
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
  "driving": {
    "type": "object",
    "title": "driving",
    "properties": {
      "drag_gaussian": {
        "type": "object",
        "title": "drag_gaussian",
        "properties": {
          "gate_len": {
            "type": "number",
            "description": "The pulse length [s]"
          },
          "gate_sigma": {
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
          "gate_shape": {
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
          "gate_len",
          "gate_sigma",
          "alpha",
          "detuning",
          "gate_shape",
          "angle2volt"
        ]
      },
      "drag_cosine": {
        "type": "object",
        "title": "drag_cosine",
        "properties": {
          "gate_len": {
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
          "gate_shape": {
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
          "gate_len",
          "alpha",
          "detuning",
          "gate_shape",
          "angle2volt"
        ]
      }
    },
    "required": [
      "drag_gaussian",
      "drag_cosine"
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
      "flux_line": {
        "type": "object",
        "title": "flux_line",
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
      "flux_filter_coefficients": {
        "type": "object",
        "title": "flux_filter_coefficients",
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
      }
    },
    "required": [
      "drive_line_index",
      "correction_matrix",
      "flux_line",
      "flux_filter_coefficients"
    ]
  },
  "flux_bias_points": {
    "type": "array",
    "items": {
      "type": "object",
      "title": "flux_bias_point",
      "properties": {
        "name": {
          "type": "string"
        },
        "value": {
          "type": "number",
          "description": "Bias voltage to set qubit to maximal frequency [V]"
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
              "type": "integer",
              "description": "[ns]"
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
            "amplitude": {
              "type": "number"
            },
            "length": {
              "type": "integer"
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
            "amplitude",
            "length",
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

class Constant2(object):

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
        """"""
        
        value = self._quam._json[self._path + "amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @amplitude.setter
    def amplitude(self, value: float):
        """"""
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
    def length(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: int):
        """"""
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


class Arbitrary2(object):

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
        """"""
        
        value = self._quam._json[self._path + "amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @amplitude.setter
    def amplitude(self, value: float):
        """"""
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
    def length(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: int):
        """"""
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


class Flux_pulse(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def constant(self) -> Constant2:
        """"""
        return Constant2(
            self._quam, self._path + "constant/", self._index,
            self._schema["properties"]["constant"]
        )

    @property
    def arbitrary(self) -> Arbitrary2:
        """"""
        return Arbitrary2(
            self._quam, self._path + "arbitrary/", self._index,
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


class CZ(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def conditional_qubit(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "conditional_qubit"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @conditional_qubit.setter
    def conditional_qubit(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "conditional_qubit")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "conditional_qubit"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "conditional_qubit"] = value

    @property
    def target_qubit(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "target_qubit"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @target_qubit.setter
    def target_qubit(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "target_qubit")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "target_qubit"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "target_qubit"] = value

    @property
    def flux_pulse(self) -> Flux_pulse:
        """"""
        return Flux_pulse(
            self._quam, self._path + "flux_pulse/", self._index,
            self._schema["properties"]["flux_pulse"]
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


class CZList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> CZ:
        return CZ(
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
        """Adds a new CZ by adding a JSON dictionary with following schema
{
  "conditional_qubit": {
    "type": "integer"
  },
  "target_qubit": {
    "type": "integer"
  },
  "flux_pulse": {
    "type": "object",
    "title": "flux_pulse",
    "properties": {
      "constant": {
        "type": "object",
        "title": "constant",
        "properties": {
          "name": {
            "type": "string"
          },
          "amplitude": {
            "type": "number"
          },
          "length": {
            "type": "integer"
          }
        },
        "required": [
          "name",
          "amplitude",
          "length"
        ]
      },
      "arbitrary": {
        "type": "object",
        "title": "arbitrary",
        "properties": {
          "name": {
            "type": "string"
          },
          "amplitude": {
            "type": "number"
          },
          "length": {
            "type": "integer"
          }
        },
        "required": [
          "name",
          "amplitude",
          "length"
        ]
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

class Two_qubit_gates(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def CZ(self) -> CZList:
        """"""
        return CZList(
            self._quam, self._path + "CZ", self._index,
            self._schema["properties"]["CZ"]
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
        self._schema_flat = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'name': 'QuAM storage format', 'description': 'optimized data structure for communication and storage', 'type': 'object', 'properties': {'digital_waveforms[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'readout_lines[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'readout_resonators[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'drive_lines[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]/flux_bias_points[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]/sequence_states/constant[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]/sequence_states/arbitrary[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'single_qubit_operations[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'two_qubit_gates/CZ[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}}, 'additionalProperties': False}
        self._schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'title': 'QuAM', 'properties': {'network': {'type': 'object', 'title': 'network', 'properties': {'qop_ip': {'type': 'string'}, 'port': {'type': 'integer'}}, 'required': ['qop_ip', 'port']}, 'controllers': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'digital_waveforms': {'type': 'array', 'items': {'type': 'object', 'title': 'digital_waveform', 'properties': {'name': {'type': 'string'}, 'samples': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['name', 'samples']}}, 'common_operation': {'type': 'object', 'title': 'common_operation', 'description': 'an operation which is common to all elements', 'properties': {'name': {'type': 'string'}, 'duration': {'type': 'number', 'description': 'pulse length [s]'}, 'amplitude': {'type': 'number', 'description': 'pulse amplitude [V]'}}, 'required': ['name', 'duration', 'amplitude']}, 'readout_lines': {'type': 'array', 'items': {'type': 'object', 'title': 'readout_line', 'properties': {'length': {'type': 'number', 'description': 'readout time on this readout line [s]'}, 'lo_freq': {'type': 'number', 'description': 'LO frequency for readout line [Hz]'}, 'lo_power': {'type': 'integer', 'description': 'LO power for readout line [dBm]'}, 'I_up': {'type': 'object', 'title': 'I_up', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'Q_up': {'type': 'object', 'title': 'Q_up', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'I_down': {'type': 'object', 'title': 'I_down', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}, 'gain_db': {'type': 'integer'}}, 'required': ['controller', 'channel', 'offset', 'gain_db']}, 'Q_down': {'type': 'object', 'title': 'Q_down', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}, 'gain_db': {'type': 'integer'}}, 'required': ['controller', 'channel', 'offset', 'gain_db']}}, 'required': ['length', 'lo_freq', 'lo_power', 'I_up', 'Q_up', 'I_down', 'Q_down']}}, 'readout_resonators': {'type': 'array', 'items': {'type': 'object', 'title': 'readout_resonator', 'properties': {'index': {'type': 'integer'}, 'name': {'type': 'string'}, 'f_res': {'type': 'number', 'description': 'Resonator resonance frequency [Hz].'}, 'f_opt': {'type': 'number', 'description': 'Resonator optimal readout frequency [Hz] (used in QUA).'}, 'readout_regime': {'type': 'string'}, 'readout_amplitude': {'type': 'number', 'description': 'Readout amplitude for this resonator [V]. Must be within [-0.5, 0.5).'}, 'rotation_angle': {'type': 'number', 'description': "Angle by which to rotate the IQ blobs to place the separation along the 'I' quadrature [degrees]."}, 'ge_threshold': {'type': 'number', 'description': "Threshold (in demod unit) along the 'I' quadrature discriminating between qubit ground and excited states."}, 'opt_readout_frequency': {'type': 'number'}, 'readout_fidelity': {'type': 'number'}, 'q_factor': {'type': 'number'}, 'chi': {'type': 'number'}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'readout_line_index': {'type': 'integer', 'description': 'Index of the readout line connected to this resonator.'}, 'time_of_flight': {'type': 'integer', 'description': 'Time of flight for this resonator [ns].'}, 'correction_matrix': {'type': 'object', 'title': 'correction_matrix', 'properties': {'gain': {'type': 'number'}, 'phase': {'type': 'number'}}, 'required': ['gain', 'phase']}}, 'required': ['readout_line_index', 'time_of_flight', 'correction_matrix']}}, 'required': ['index', 'name', 'f_res', 'f_opt', 'readout_regime', 'readout_amplitude', 'rotation_angle', 'ge_threshold', 'opt_readout_frequency', 'readout_fidelity', 'q_factor', 'chi', 'wiring']}}, 'drive_lines': {'type': 'array', 'items': {'type': 'object', 'title': 'drive_line', 'properties': {'qubits': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}, 'description': 'qubits associated with this drive line'}, 'lo_freq': {'type': 'number', 'description': 'LO frequency [Hz]'}, 'lo_power': {'type': 'integer', 'description': 'LO power to drive line [dBm]'}, 'I': {'type': 'object', 'title': 'I', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'Q': {'type': 'object', 'title': 'Q', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}}, 'required': ['qubits', 'lo_freq', 'lo_power', 'I', 'Q']}}, 'qubits': {'type': 'array', 'items': {'type': 'object', 'title': 'qubit', 'properties': {'index': {'type': 'integer'}, 'name': {'type': 'string'}, 'f_01': {'type': 'number', 'description': '0-1 transition frequency [Hz]'}, 'anharmonicity': {'type': 'number', 'description': 'Qubit anharmonicity: difference in energy between the 2-1 and the 1-0 energy levels [Hz]'}, 'rabi_freq': {'type': 'integer', 'description': 'Qubit Rabi frequency [Hz]'}, 't1': {'type': 'number', 'description': 'Relaxation time T1 [s]'}, 't2': {'type': 'number', 'description': 'Dephasing time T2 [s]'}, 't2star': {'type': 'number', 'description': 'Dephasing time T2* [s]'}, 'driving': {'type': 'object', 'title': 'driving', 'properties': {'drag_gaussian': {'type': 'object', 'title': 'drag_gaussian', 'properties': {'gate_len': {'type': 'number', 'description': 'The pulse length [s]'}, 'gate_sigma': {'type': 'number', 'description': 'The gaussian standard deviation (only for gaussian pulses) [s]'}, 'alpha': {'type': 'number', 'description': 'The DRAG coefficient alpha.'}, 'detuning': {'type': 'integer', 'description': 'The frequency shift to correct for AC stark shift [Hz].'}, 'gate_shape': {'type': 'string', 'description': 'Shape of the gate'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['gate_len', 'gate_sigma', 'alpha', 'detuning', 'gate_shape', 'angle2volt']}, 'drag_cosine': {'type': 'object', 'title': 'drag_cosine', 'properties': {'gate_len': {'type': 'number', 'description': 'The pulse length [s]'}, 'alpha': {'type': 'number', 'description': 'The DRAG coefficient alpha.'}, 'detuning': {'type': 'integer', 'description': 'The frequency shift to correct for AC stark shift [Hz].'}, 'gate_shape': {'type': 'string', 'description': 'Shape of the gate'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['gate_len', 'alpha', 'detuning', 'gate_shape', 'angle2volt']}}, 'required': ['drag_gaussian', 'drag_cosine']}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'drive_line_index': {'type': 'integer'}, 'correction_matrix': {'type': 'object', 'title': 'correction_matrix', 'properties': {'gain': {'type': 'number'}, 'phase': {'type': 'number'}}, 'required': ['gain', 'phase']}, 'flux_line': {'type': 'object', 'title': 'flux_line', 'properties': {'controller': {'type': 'string'}, 'channel': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'channel', 'offset']}, 'flux_filter_coefficients': {'type': 'object', 'title': 'flux_filter_coefficients', 'properties': {'feedforward': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'feedback': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['feedforward', 'feedback']}}, 'required': ['drive_line_index', 'correction_matrix', 'flux_line', 'flux_filter_coefficients']}, 'flux_bias_points': {'type': 'array', 'items': {'type': 'object', 'title': 'flux_bias_point', 'properties': {'name': {'type': 'string'}, 'value': {'type': 'number', 'description': 'Bias voltage to set qubit to maximal frequency [V]'}}, 'required': ['name', 'value']}}, 'sequence_states': {'type': 'object', 'title': 'sequence_states', 'properties': {'constant': {'type': 'array', 'items': {'type': 'object', 'title': 'constant', 'properties': {'name': {'type': 'string'}, 'amplitude': {'type': 'number', 'description': '[V]'}, 'length': {'type': 'integer', 'description': '[ns]'}}, 'required': ['name', 'amplitude', 'length']}}, 'arbitrary': {'type': 'array', 'items': {'type': 'object', 'title': 'arbitrary', 'properties': {'name': {'type': 'string'}, 'amplitude': {'type': 'number'}, 'length': {'type': 'integer'}, 'waveform': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}, 'description': 'points describing the waveform shape'}}, 'required': ['name', 'amplitude', 'length', 'waveform']}}}, 'required': ['constant', 'arbitrary']}}, 'required': ['index', 'name', 'f_01', 'anharmonicity', 'rabi_freq', 't1', 't2', 't2star', 'driving', 'wiring', 'flux_bias_points', 'sequence_states']}}, 'crosstalk_matrix': {'type': 'object', 'title': 'crosstalk_matrix', 'properties': {'static': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'fast': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['static', 'fast']}, 'single_qubit_operations': {'type': 'array', 'items': {'type': 'object', 'title': 'single_qubit_operation', 'properties': {'direction': {'type': 'string'}, 'angle': {'type': 'integer'}}, 'required': ['direction', 'angle']}}, 'two_qubit_gates': {'type': 'object', 'title': 'two_qubit_gates', 'properties': {'CZ': {'type': 'array', 'items': {'type': 'object', 'title': 'CZ', 'properties': {'conditional_qubit': {'type': 'integer'}, 'target_qubit': {'type': 'integer'}, 'flux_pulse': {'type': 'object', 'title': 'flux_pulse', 'properties': {'constant': {'type': 'object', 'title': 'constant', 'properties': {'name': {'type': 'string'}, 'amplitude': {'type': 'number'}, 'length': {'type': 'integer'}}, 'required': ['name', 'amplitude', 'length']}, 'arbitrary': {'type': 'object', 'title': 'arbitrary', 'properties': {'name': {'type': 'string'}, 'amplitude': {'type': 'number'}, 'length': {'type': 'integer'}}, 'required': ['name', 'amplitude', 'length']}}, 'required': ['constant', 'arbitrary']}}, 'required': ['conditional_qubit', 'target_qubit', 'flux_pulse']}}}, 'required': ['CZ']}, 'running_strategy': {'type': 'object', 'title': 'running_strategy', 'properties': {'running': {'type': 'boolean'}, 'start': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'end': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['running', 'start', 'end']}}, 'required': ['network', 'controllers', 'digital_waveforms', 'common_operation', 'readout_lines', 'readout_resonators', 'drive_lines', 'qubits', 'crosstalk_matrix', 'single_qubit_operations', 'two_qubit_gates', 'running_strategy']}
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
    def single_qubit_operations(self) -> Single_qubit_operationsList:
        """"""
        return Single_qubit_operationsList(
            self._quam, self._path + "single_qubit_operations", self._index,
            self._schema["properties"]["single_qubit_operations"]
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
    def two_qubit_gates(self) -> Two_qubit_gates:
        """"""
        return Two_qubit_gates(
            self._quam, self._path + "two_qubit_gates/", self._index,
            self._schema["properties"]["two_qubit_gates"]
        )

    @property
    def running_strategy(self) -> Running_strategy:
        """"""
        return Running_strategy(
            self._quam, self._path + "running_strategy/", self._index,
            self._schema["properties"]["running_strategy"]
        )

    def build_config(self, digital_out: list, qubits: list, gate_shape: str):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.build_config(self, digital_out, qubits, gate_shape)

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

    def get_wiring(self):
        """
    Print the state connectivity.
    """
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_wiring(self)

    def get_sequence_state(self, index: int, sequence_state: str):
        """
    Get the sequence state object.

    :param index: index of the qubit to be retrieved.
    :param sequence_state: name of the sequence.
    :return: the sequence state object.
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

    def get_driving(self, pulse, q: int):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_driving(self, pulse, q)

    def get_qubit_gate(self, index, gate_shape):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_qubit_gate(self, index, gate_shape)

    def get_length(self, index, operation):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_length(self, index, operation)

    def set_length(self, index, operation, length):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.set_length(self, index, operation, length)

    def get_flux_bias_point(self, index: int, flux_bias_point):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_flux_bias_point(self, index, flux_bias_point)

    def get_readout_IF(self, index):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_readout_IF(self, index)

    def get_qubit_IF(self, index):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.get_qubit_IF(self, index)

    def nullify_qubits(self, cond: bool, q_list: list, indx: int):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.nullify_qubits(self, cond, q_list, indx)

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


