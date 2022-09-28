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


class Analog_output(object):

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
    def output(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "output"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @output.setter
    def output(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "output")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "output"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "output"] = value

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


class Analog_outputsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Analog_output:
        return Analog_output(
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
        """Adds a new analog_output by adding a JSON dictionary with following schema
{
  "controller": {
    "type": "string"
  },
  "output": {
    "type": "integer"
  },
  "offset": {
    "type": "number"
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

class Analog_input(object):

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
    def input(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "input"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @input.setter
    def input(self, value: int):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "input")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "input"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "input"] = value

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


class Analog_inputsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Analog_input:
        return Analog_input(
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
        """Adds a new analog_input by adding a JSON dictionary with following schema
{
  "controller": {
    "type": "string"
  },
  "input": {
    "type": "integer"
  },
  "offset": {
    "type": "number"
  },
  "gain_db": {
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

class Analog_waveform(object):

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
    def type(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "type"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @type.setter
    def type(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "type")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "type"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "type"] = value

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


class Analog_waveformsList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Analog_waveform:
        return Analog_waveform(
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
        """Adds a new analog_waveform by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "type": {
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

class Waveforms(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def I(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @I.setter
    def I(self, value: str):
        """"""
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
    def Q(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @Q.setter
    def Q(self, value: str):
        """"""
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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Pulse(object):

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
    def operation(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "operation"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @operation.setter
    def operation(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "operation")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "operation"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "operation"] = value

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
    def waveforms(self) -> Waveforms:
        """"""
        return Waveforms(
            self._quam, self._path + "waveforms/", self._index,
            self._schema["properties"]["waveforms"]
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


class PulsesList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Pulse:
        return Pulse(
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
        """Adds a new pulse by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "operation": {
    "type": "string"
  },
  "length": {
    "type": "integer"
  },
  "waveforms": {
    "type": "object",
    "title": "waveforms",
    "properties": {
      "I": {
        "type": "string"
      },
      "Q": {
        "type": "string"
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

class Waveforms2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def single(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "single"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @single.setter
    def single(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "single")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "single"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "single"] = value

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


class Pulses_single(object):

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
    def operation(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "operation"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @operation.setter
    def operation(self, value: str):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "operation")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "operation"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "operation"] = value

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
    def waveforms(self) -> Waveforms2:
        """"""
        return Waveforms2(
            self._quam, self._path + "waveforms/", self._index,
            self._schema["properties"]["waveforms"]
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


class Pulses_singleList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Pulses_single:
        return Pulses_single(
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
        """Adds a new pulses_single by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "operation": {
    "type": "string"
  },
  "length": {
    "type": "integer"
  },
  "waveforms": {
    "type": "object",
    "title": "waveforms",
    "properties": {
      "single": {
        "type": "string"
      }
    },
    "required": [
      "single"
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
        """readout time (seconds) on this drive line"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """readout time (seconds) on this drive line"""
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
        """LO frequency for readout line"""
        
        value = self._quam._json[self._path + "lo_freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @lo_freq.setter
    def lo_freq(self, value: float):
        """LO frequency for readout line"""
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
    "description": "readout time (seconds) on this drive line"
  },
  "lo_freq": {
    "type": "number",
    "description": "LO frequency for readout line"
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
        """"""
        
        value = self._quam._json[self._path + "readout_line_index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_line_index.setter
    def readout_line_index(self, value: int):
        """"""
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
        """"""
        
        value = self._quam._json[self._path + "time_of_flight"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @time_of_flight.setter
    def time_of_flight(self, value: int):
        """"""
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
    def I(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "I",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @I.setter
    def I(self, value: List[Union[str, int, float, bool, list]]):
        """"""
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
    def Q(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "Q",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @Q.setter
    def Q(self, value: List[Union[str, int, float, bool, list]]):
        """"""
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

    @property
    def correction_matrix(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "correction_matrix",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "correction_matrix"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @correction_matrix.setter
    def correction_matrix(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "correction_matrix")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "correction_matrix"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "correction_matrix"] = value

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
    def f_res(self) -> float:
        """resonator frequency (Hz)"""
        
        value = self._quam._json[self._path + "f_res"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f_res.setter
    def f_res(self, value: float):
        """resonator frequency (Hz)"""
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
        """"""
        
        value = self._quam._json[self._path + "readout_amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_amplitude.setter
    def readout_amplitude(self, value: float):
        """"""
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
    def rotation_angle(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "rotation_angle"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rotation_angle.setter
    def rotation_angle(self, value: float):
        """"""
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
  "f_res": {
    "type": "number",
    "description": "resonator frequency (Hz)"
  },
  "q_factor": {
    "type": "number"
  },
  "readout_regime": {
    "type": "string"
  },
  "readout_amplitude": {
    "type": "number"
  },
  "opt_readout_frequency": {
    "type": "number"
  },
  "rotation_angle": {
    "type": "number"
  },
  "readout_fidelity": {
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
        "type": "integer"
      },
      "time_of_flight": {
        "type": "integer"
      },
      "I": {
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
      "Q": {
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
      "correction_matrix": {
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
      "readout_line_index",
      "time_of_flight",
      "I",
      "Q",
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
        """LO frequency"""
        
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
        """LO frequency"""
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
    def freq(self) -> float:
        """LO power to mixer"""
        
        value = self._quam._json[self._path + "freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @freq.setter
    def freq(self, value: float):
        """LO power to mixer"""
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
    def power(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "power"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @power.setter
    def power(self, value: int):
        """"""
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

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


class Drive_lineList(object):

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
  "freq": {
    "type": "number",
    "description": "LO frequency"
  },
  "power": {
    "type": "integer",
    "description": "LO power to mixer"
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


class Driving(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def gate_len(self) -> float:
        """(seconds)"""
        
        value = self._quam._json[self._path + "gate_len"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gate_len.setter
    def gate_len(self, value: float):
        """(seconds)"""
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
        """"""
        
        value = self._quam._json[self._path + "gate_sigma"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gate_sigma.setter
    def gate_sigma(self, value: float):
        """"""
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
    def gate_shape(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "gate_shape"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gate_shape.setter
    def gate_shape(self, value: str):
        """"""
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
        """"""
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


class Flux_filter_coef(object):

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
    def I(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "I",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @I.setter
    def I(self, value: List[Union[str, int, float, bool, list]]):
        """"""
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
    def Q(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "Q",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @Q.setter
    def Q(self, value: List[Union[str, int, float, bool, list]]):
        """"""
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

    @property
    def correction_matrix(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "correction_matrix",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "correction_matrix"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @correction_matrix.setter
    def correction_matrix(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "correction_matrix")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "correction_matrix"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "correction_matrix"] = value

    @property
    def flux_line(self) -> List[Union[str, int, float, bool, list]]:
        """"""
        
        if self._quam._record_updates:
            return _List(
                self._quam,
                self._path + "flux_line",
                self._index,
                self._schema
            )
        
        value = self._quam._json[self._path + "flux_line"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @flux_line.setter
    def flux_line(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "flux_line")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "flux_line"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "flux_line"] = value

    @property
    def flux_filter_coef(self) -> Flux_filter_coef:
        """filter taps IIR and FIR to fast flux line"""
        return Flux_filter_coef(
            self._quam, self._path + "flux_filter_coef/", self._index,
            self._schema["properties"]["flux_filter_coef"]
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


class Sequence_state(object):

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


class Sequence_statesList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Sequence_state:
        return Sequence_state(
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
        """Adds a new sequence_state by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "amplitude": {
    "type": "number"
  },
  "length": {
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

class Qubit(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def f_01(self) -> float:
        """01 transition frequency (Hz)"""
        
        value = self._quam._json[self._path + "f_01"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f_01.setter
    def f_01(self, value: float):
        """01 transition frequency (Hz)"""
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
        """"""
        
        value = self._quam._json[self._path + "anharmonicity"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @anharmonicity.setter
    def anharmonicity(self, value: float):
        """"""
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
    def rabi_freq(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "rabi_freq"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rabi_freq.setter
    def rabi_freq(self, value: float):
        """"""
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
        """"""
        
        value = self._quam._json[self._path + "t1"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t1.setter
    def t1(self, value: float):
        """"""
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
        """"""
        
        value = self._quam._json[self._path + "t2"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t2.setter
    def t2(self, value: float):
        """"""
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
        """"""
        
        value = self._quam._json[self._path + "t2star"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @t2star.setter
    def t2star(self, value: float):
        """"""
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
    def sequence_states(self) -> Sequence_statesList:
        """"""
        return Sequence_statesList(
            self._quam, self._path + "sequence_states", self._index,
            self._schema["properties"]["sequence_states"]
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
  "f_01": {
    "type": "number",
    "description": "01 transition frequency (Hz)"
  },
  "anharmonicity": {
    "type": "number"
  },
  "rabi_freq": {
    "type": "number"
  },
  "t1": {
    "type": "number"
  },
  "t2": {
    "type": "number"
  },
  "t2star": {
    "type": "number"
  },
  "driving": {
    "type": "object",
    "title": "driving",
    "properties": {
      "gate_len": {
        "type": "number",
        "description": "(seconds)"
      },
      "gate_sigma": {
        "type": "number"
      },
      "gate_shape": {
        "type": "string"
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
      "gate_shape",
      "angle2volt"
    ]
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "I": {
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
      "Q": {
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
      "correction_matrix": {
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
      "flux_line": {
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
        "description": "controller port associated with fast flux line"
      },
      "flux_filter_coef": {
        "type": "object",
        "title": "flux_filter_coef",
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
      "I",
      "Q",
      "correction_matrix",
      "flux_line",
      "flux_filter_coef"
    ]
  },
  "sequence_states": {
    "type": "array",
    "items": {
      "type": "object",
      "title": "sequence_state",
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
        self._schema_flat = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'name': 'QuAM storage format', 'description': 'optimized data structure for communication and storage', 'type': 'object', 'properties': {'analog_outputs[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'analog_inputs[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'analog_waveforms[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'digital_waveforms[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'pulses[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'pulses_single[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'readout_lines[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'readout_resonators[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'drive_line[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'qubits[]/sequence_states[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'single_qubit_operations[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}}, 'additionalProperties': False}
        self._schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'title': 'QuAM', 'properties': {'analog_outputs': {'type': 'array', 'items': {'type': 'object', 'title': 'analog_output', 'properties': {'controller': {'type': 'string'}, 'output': {'type': 'integer'}, 'offset': {'type': 'number'}}, 'required': ['controller', 'output', 'offset']}}, 'analog_inputs': {'type': 'array', 'items': {'type': 'object', 'title': 'analog_input', 'properties': {'controller': {'type': 'string'}, 'input': {'type': 'integer'}, 'offset': {'type': 'number'}, 'gain_db': {'type': 'integer'}}, 'required': ['controller', 'input', 'offset', 'gain_db']}}, 'analog_waveforms': {'type': 'array', 'items': {'type': 'object', 'title': 'analog_waveform', 'properties': {'name': {'type': 'string'}, 'type': {'type': 'string'}, 'samples': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['name', 'type', 'samples']}}, 'digital_waveforms': {'type': 'array', 'items': {'type': 'object', 'title': 'digital_waveform', 'properties': {'name': {'type': 'string'}, 'samples': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['name', 'samples']}}, 'pulses': {'type': 'array', 'items': {'type': 'object', 'title': 'pulse', 'properties': {'name': {'type': 'string'}, 'operation': {'type': 'string'}, 'length': {'type': 'integer'}, 'waveforms': {'type': 'object', 'title': 'waveforms', 'properties': {'I': {'type': 'string'}, 'Q': {'type': 'string'}}, 'required': ['I', 'Q']}}, 'required': ['name', 'operation', 'length', 'waveforms']}}, 'pulses_single': {'type': 'array', 'items': {'type': 'object', 'title': 'pulses_single', 'properties': {'name': {'type': 'string'}, 'operation': {'type': 'string'}, 'length': {'type': 'integer'}, 'waveforms': {'type': 'object', 'title': 'waveforms', 'properties': {'single': {'type': 'string'}}, 'required': ['single']}}, 'required': ['name', 'operation', 'length', 'waveforms']}}, 'readout_lines': {'type': 'array', 'items': {'type': 'object', 'title': 'readout_line', 'properties': {'length': {'type': 'number', 'description': 'readout time (seconds) on this drive line'}, 'lo_freq': {'type': 'number', 'description': 'LO frequency for readout line'}}, 'required': ['length', 'lo_freq']}}, 'readout_resonators': {'type': 'array', 'items': {'type': 'object', 'title': 'readout_resonator', 'properties': {'f_res': {'type': 'number', 'description': 'resonator frequency (Hz)'}, 'q_factor': {'type': 'number'}, 'readout_regime': {'type': 'string'}, 'readout_amplitude': {'type': 'number'}, 'opt_readout_frequency': {'type': 'number'}, 'rotation_angle': {'type': 'number'}, 'readout_fidelity': {'type': 'number'}, 'chi': {'type': 'number'}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'readout_line_index': {'type': 'integer'}, 'time_of_flight': {'type': 'integer'}, 'I': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'Q': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'correction_matrix': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['readout_line_index', 'time_of_flight', 'I', 'Q', 'correction_matrix']}}, 'required': ['f_res', 'q_factor', 'readout_regime', 'readout_amplitude', 'opt_readout_frequency', 'rotation_angle', 'readout_fidelity', 'chi', 'wiring']}}, 'crosstalk_matrix': {'type': 'object', 'title': 'crosstalk_matrix', 'properties': {'static': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}, 'description': 'crosstalk matrix for slow flux lines'}, 'fast': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}, 'description': 'crosstalk matrix for fast flux lines'}}, 'required': ['static', 'fast']}, 'drive_line': {'type': 'array', 'items': {'type': 'object', 'title': 'drive_line', 'properties': {'qubits': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}, 'description': 'qubits associated with this drive line'}, 'freq': {'type': 'number', 'description': 'LO frequency'}, 'power': {'type': 'integer', 'description': 'LO power to mixer'}}, 'required': ['qubits', 'freq', 'power']}}, 'qubits': {'type': 'array', 'items': {'type': 'object', 'title': 'qubit', 'properties': {'f_01': {'type': 'number', 'description': '01 transition frequency (Hz)'}, 'anharmonicity': {'type': 'number'}, 'rabi_freq': {'type': 'number'}, 't1': {'type': 'number'}, 't2': {'type': 'number'}, 't2star': {'type': 'number'}, 'driving': {'type': 'object', 'title': 'driving', 'properties': {'gate_len': {'type': 'number', 'description': '(seconds)'}, 'gate_sigma': {'type': 'number'}, 'gate_shape': {'type': 'string'}, 'angle2volt': {'type': 'object', 'title': 'angle2volt', 'properties': {'deg90': {'type': 'number'}, 'deg180': {'type': 'number'}}, 'required': ['deg90', 'deg180']}}, 'required': ['gate_len', 'gate_sigma', 'gate_shape', 'angle2volt']}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'I': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'Q': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'correction_matrix': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'flux_line': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}, 'description': 'controller port associated with fast flux line'}, 'flux_filter_coef': {'type': 'object', 'title': 'flux_filter_coef', 'properties': {'feedforward': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'feedback': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['feedforward', 'feedback']}}, 'required': ['I', 'Q', 'correction_matrix', 'flux_line', 'flux_filter_coef']}, 'sequence_states': {'type': 'array', 'items': {'type': 'object', 'title': 'sequence_state', 'properties': {'name': {'type': 'string'}, 'amplitude': {'type': 'number'}, 'length': {'type': 'integer'}}, 'required': ['name', 'amplitude', 'length']}}}, 'required': ['f_01', 'anharmonicity', 'rabi_freq', 't1', 't2', 't2star', 'driving', 'wiring', 'sequence_states']}}, 'single_qubit_operations': {'type': 'array', 'items': {'type': 'object', 'title': 'single_qubit_operation', 'properties': {'direction': {'type': 'string'}, 'angle': {'type': 'integer'}}, 'required': ['direction', 'angle']}}, 'running_strategy': {'type': 'object', 'title': 'running_strategy', 'properties': {'running': {'type': 'boolean'}, 'start': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}, 'end': {'type': 'array', 'items': {'anyOf': [{'type': 'integer'}, {'type': 'number'}, {'type': 'string'}, {'type': 'boolean'}, {'type': 'array'}]}}}, 'required': ['running', 'start', 'end']}}, 'required': ['analog_outputs', 'analog_inputs', 'analog_waveforms', 'digital_waveforms', 'pulses', 'pulses_single', 'readout_lines', 'readout_resonators', 'crosstalk_matrix', 'drive_line', 'qubits', 'single_qubit_operations', 'running_strategy']}
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
    def analog_outputs(self) -> Analog_outputsList:
        """"""
        return Analog_outputsList(
            self._quam, self._path + "analog_outputs", self._index,
            self._schema["properties"]["analog_outputs"]
        )

    @property
    def analog_inputs(self) -> Analog_inputsList:
        """"""
        return Analog_inputsList(
            self._quam, self._path + "analog_inputs", self._index,
            self._schema["properties"]["analog_inputs"]
        )

    @property
    def analog_waveforms(self) -> Analog_waveformsList:
        """"""
        return Analog_waveformsList(
            self._quam, self._path + "analog_waveforms", self._index,
            self._schema["properties"]["analog_waveforms"]
        )

    @property
    def digital_waveforms(self) -> Digital_waveformsList:
        """"""
        return Digital_waveformsList(
            self._quam, self._path + "digital_waveforms", self._index,
            self._schema["properties"]["digital_waveforms"]
        )

    @property
    def pulses(self) -> PulsesList:
        """"""
        return PulsesList(
            self._quam, self._path + "pulses", self._index,
            self._schema["properties"]["pulses"]
        )

    @property
    def pulses_single(self) -> Pulses_singleList:
        """"""
        return Pulses_singleList(
            self._quam, self._path + "pulses_single", self._index,
            self._schema["properties"]["pulses_single"]
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
    def drive_line(self) -> Drive_lineList:
        """"""
        return Drive_lineList(
            self._quam, self._path + "drive_line", self._index,
            self._schema["properties"]["drive_line"]
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
    def crosstalk_matrix(self) -> Crosstalk_matrix:
        """"""
        return Crosstalk_matrix(
            self._quam, self._path + "crosstalk_matrix/", self._index,
            self._schema["properties"]["crosstalk_matrix"]
        )

    @property
    def running_strategy(self) -> Running_strategy:
        """"""
        return Running_strategy(
            self._quam, self._path + "running_strategy/", self._index,
            self._schema["properties"]["running_strategy"]
        )

    def build_config(self):
        """"""
        with _add_path(os.path.dirname(os.path.abspath(__file__))):
            import config
        return config.build_config(self)

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


