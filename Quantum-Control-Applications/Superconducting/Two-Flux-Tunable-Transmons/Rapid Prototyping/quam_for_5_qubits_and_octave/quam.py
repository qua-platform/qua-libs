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
        return _List(self._quam, self._path, self._index + [key], self._schema)

    def __setitem__(self, key, newvalue):
        index = self._index + [key]
        if len(index) > 0:
            value_ref = self._quam._json[self._path]
            for i in range(len(index) - 1):
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
        for i in range(len(self._index) - 1):
            value_ref = value_ref[self._index[i]]
        return len(value_ref)

    def _json_view(self, metadata: bool = False):
        value_ref = self._quam._json[self._path]
        if len(self._index) > 0:
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            return value_ref[self._index[-1]]
        return value_ref


class _add_path:
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "qop_ip"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "qop_ip"] = value

    @property
    def cluster_name(self) -> str:
        """"""

        value = self._quam._json[self._path + "cluster_name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @cluster_name.setter
    def cluster_name(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "cluster_name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "cluster_name"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "cluster_name"] = value

    @property
    def save_dir(self) -> str:
        """"""

        value = self._quam._json[self._path + "save_dir"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @save_dir.setter
    def save_dir(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "save_dir")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "save_dir"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "save_dir"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Network2(object):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "qop_ip"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "qop_port"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "qop_port"] = value

    @property
    def save_dir(self) -> str:
        """"""

        value = self._quam._json[self._path + "save_dir"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @save_dir.setter
    def save_dir(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "save_dir")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "save_dir"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "save_dir"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Network2):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "freq"]
            for i in range(len(self._index) - 1):
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
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "power")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "power"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "power"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class QubitsList(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Qubit:
        return Qubit(self._quam, self._path + "[].", self._index + [key], self._schema["items"])

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

    def _json_view(self, metadata=False):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view(metadata=metadata and i == 0))
        return result

    def append(self, json_item: dict):
        """Adds a new qubit by adding a JSON dictionary with following schema
        {
          "freq": {
            "type": "number"
          },
          "power": {
            "type": "integer"
          }
        }"""
        import quam_sdk.crud

        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path + "[].", self._index, new_item=True)
        quam_sdk.crud.bump_list_length(self._quam._json, f"{self._path}[]_len", self._index)

    def __str__(self) -> str:
        return json.dumps(self._json_view())

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "freq"]
            for i in range(len(self._index) - 1):
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
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "power")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "power"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "power"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class ReadoutList(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Readout:
        return Readout(self._quam, self._path + "[].", self._index + [key], self._schema["items"])

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

    def _json_view(self, metadata=False):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view(metadata=metadata and i == 0))
        return result

    def append(self, json_item: dict):
        """Adds a new readout by adding a JSON dictionary with following schema
        {
          "freq": {
            "type": "number"
          },
          "power": {
            "type": "integer"
          }
        }"""
        import quam_sdk.crud

        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path + "[].", self._index, new_item=True)
        quam_sdk.crud.bump_list_length(self._quam._json, f"{self._path}[]_len", self._index)

    def __str__(self) -> str:
        return json.dumps(self._json_view())

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
    def qubits(self) -> QubitsList:
        """"""
        return QubitsList(self._quam, self._path + "qubits", self._index, self._schema["properties"]["qubits"])

    @property
    def readout(self) -> ReadoutList:
        """"""
        return ReadoutList(self._quam, self._path + "readout", self._index, self._schema["properties"]["readout"])

    @property
    def network(self) -> Network2:
        """"""
        return Network2(self._quam, self._path + "network.", self._index, self._schema["properties"]["network"])

    @network.setter
    def network(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "network", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "offset_I"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "offset_Q"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "gain"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "phase"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "phase"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
    def controller(self) -> str:
        """"""

        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @controller.setter
    def controller(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def I(self) -> List[Union[str, int, float, bool, list]]:
        """"""

        if self._quam._record_updates:
            return _List(self._quam, self._path + "I", self._index, self._schema)

        value = self._quam._json[self._path + "I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @I.setter
    def I(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "I")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "I"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "I"] = value

    @property
    def Q(self) -> List[Union[str, int, float, bool, list]]:
        """"""

        if self._quam._record_updates:
            return _List(self._quam, self._path + "Q", self._index, self._schema)

        value = self._quam._json[self._path + "Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @Q.setter
    def Q(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "Q")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "Q"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "Q"] = value

    @property
    def mixer_correction(self) -> Mixer_correction:
        """"""
        return Mixer_correction(
            self._quam, self._path + "mixer_correction.", self._index, self._schema["properties"]["mixer_correction"]
        )

    @mixer_correction.setter
    def mixer_correction(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "mixer_correction", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Xy(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def LO_index(self) -> int:
        """"""

        value = self._quam._json[self._path + "LO_index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @LO_index.setter
    def LO_index(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "LO_index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "LO_index"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "LO_index"] = value

    @property
    def f_01(self) -> float:
        """"""

        value = self._quam._json[self._path + "f_01"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @f_01.setter
    def f_01(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "f_01")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "f_01"]
            for i in range(len(self._index) - 1):
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
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "anharmonicity")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "anharmonicity"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "anharmonicity"] = value

    @property
    def drag_coefficient(self) -> float:
        """"""

        value = self._quam._json[self._path + "drag_coefficient"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @drag_coefficient.setter
    def drag_coefficient(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "drag_coefficient")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "drag_coefficient"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "drag_coefficient"] = value

    @property
    def ac_stark_detuning(self) -> float:
        """"""

        value = self._quam._json[self._path + "ac_stark_detuning"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @ac_stark_detuning.setter
    def ac_stark_detuning(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ac_stark_detuning")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "ac_stark_detuning"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ac_stark_detuning"] = value

    @property
    def pi_length(self) -> int:
        """"""

        value = self._quam._json[self._path + "pi_length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @pi_length.setter
    def pi_length(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "pi_length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "pi_length"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "pi_length"] = value

    @property
    def pi_amp(self) -> float:
        """"""

        value = self._quam._json[self._path + "pi_amp"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @pi_amp.setter
    def pi_amp(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "pi_amp")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "pi_amp"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "pi_amp"] = value

    @property
    def wiring(self) -> Wiring:
        """"""
        return Wiring(self._quam, self._path + "wiring.", self._index, self._schema["properties"]["wiring"])

    @wiring.setter
    def wiring(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "wiring", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Xy):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Filter(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def iir_taps(self) -> List[Union[str, int, float, bool, list]]:
        """"""

        if self._quam._record_updates:
            return _List(self._quam, self._path + "iir_taps", self._index, self._schema)

        value = self._quam._json[self._path + "iir_taps"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @iir_taps.setter
    def iir_taps(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "iir_taps")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "iir_taps"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "iir_taps"] = value

    @property
    def fir_taps(self) -> List[Union[str, int, float, bool, list]]:
        """"""

        if self._quam._record_updates:
            return _List(self._quam, self._path + "fir_taps", self._index, self._schema)

        value = self._quam._json[self._path + "fir_taps"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @fir_taps.setter
    def fir_taps(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "fir_taps")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "fir_taps"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "fir_taps"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Filter):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
    def controller(self) -> str:
        """"""

        value = self._quam._json[self._path + "controller"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @controller.setter
    def controller(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

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
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "port")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "port"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "port"] = value

    @property
    def filter(self) -> Filter:
        """"""
        return Filter(self._quam, self._path + "filter.", self._index, self._schema["properties"]["filter"])

    @filter.setter
    def filter(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "filter", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Iswap(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def level(self) -> float:
        """"""

        value = self._quam._json[self._path + "level"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @level.setter
    def level(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "level")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "level"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "level"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Iswap):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Cz(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "length"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "length"] = value

    @property
    def level(self) -> float:
        """"""

        value = self._quam._json[self._path + "level"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @level.setter
    def level(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "level")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "level"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "level"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Cz):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Z(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def flux_pulse_length(self) -> int:
        """"""

        value = self._quam._json[self._path + "flux_pulse_length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @flux_pulse_length.setter
    def flux_pulse_length(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "flux_pulse_length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "flux_pulse_length"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "flux_pulse_length"] = value

    @property
    def flux_pulse_amp(self) -> float:
        """"""

        value = self._quam._json[self._path + "flux_pulse_amp"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @flux_pulse_amp.setter
    def flux_pulse_amp(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "flux_pulse_amp")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "flux_pulse_amp"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "flux_pulse_amp"] = value

    @property
    def max_frequency_point(self) -> float:
        """"""

        value = self._quam._json[self._path + "max_frequency_point"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @max_frequency_point.setter
    def max_frequency_point(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "max_frequency_point")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "max_frequency_point"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "max_frequency_point"] = value

    @property
    def min_frequency_point(self) -> float:
        """"""

        value = self._quam._json[self._path + "min_frequency_point"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @min_frequency_point.setter
    def min_frequency_point(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "min_frequency_point")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "min_frequency_point"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "min_frequency_point"] = value

    @property
    def wiring(self) -> Wiring2:
        """"""
        return Wiring2(self._quam, self._path + "wiring.", self._index, self._schema["properties"]["wiring"])

    @wiring.setter
    def wiring(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "wiring", value)

    @property
    def iswap(self) -> Iswap:
        """"""
        return Iswap(self._quam, self._path + "iswap.", self._index, self._schema["properties"]["iswap"])

    @iswap.setter
    def iswap(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "iswap", value)

    @property
    def cz(self) -> Cz:
        """"""
        return Cz(self._quam, self._path + "cz.", self._index, self._schema["properties"]["cz"])

    @cz.setter
    def cz(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "cz", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Z):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Qubit2(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
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
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def ge_threshold(self) -> float:
        """"""

        value = self._quam._json[self._path + "ge_threshold"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @ge_threshold.setter
    def ge_threshold(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ge_threshold")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "ge_threshold"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ge_threshold"] = value

    @property
    def T1(self) -> int:
        """"""

        value = self._quam._json[self._path + "T1"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @T1.setter
    def T1(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "T1")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "T1"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "T1"] = value

    @property
    def T2(self) -> int:
        """"""

        value = self._quam._json[self._path + "T2"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @T2.setter
    def T2(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "T2")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "T2"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "T2"] = value

    @property
    def T2echo(self) -> int:
        """"""

        value = self._quam._json[self._path + "T2echo"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @T2echo.setter
    def T2echo(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "T2echo")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "T2echo"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "T2echo"] = value

    @property
    def xy(self) -> Xy:
        """"""
        return Xy(self._quam, self._path + "xy.", self._index, self._schema["properties"]["xy"])

    @xy.setter
    def xy(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "xy", value)

    @property
    def z(self) -> Z:
        """"""
        return Z(self._quam, self._path + "z.", self._index, self._schema["properties"]["z"])

    @z.setter
    def z(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "z", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Qubit2):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class QubitsList2(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Qubit2:
        return Qubit2(self._quam, self._path + "[].", self._index + [key], self._schema["items"])

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

    def _json_view(self, metadata=False):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view(metadata=metadata and i == 0))
        return result

    def append(self, json_item: dict):
        """Adds a new qubit by adding a JSON dictionary with following schema
        {
          "name": {
            "type": "string"
          },
          "xy": {
            "type": "object",
            "title": "xy",
            "properties": {
              "LO_index": {
                "type": "integer"
              },
              "f_01": {
                "type": "number"
              },
              "anharmonicity": {
                "type": "number"
              },
              "drag_coefficient": {
                "type": "number"
              },
              "ac_stark_detuning": {
                "type": "number"
              },
              "pi_length": {
                "type": "integer"
              },
              "pi_amp": {
                "type": "number"
              },
              "wiring": {
                "type": "object",
                "title": "wiring",
                "properties": {
                  "controller": {
                    "type": "string"
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
                  }
                },
                "required": [
                  "controller",
                  "I",
                  "Q",
                  "mixer_correction"
                ]
              }
            },
            "required": [
              "LO_index",
              "f_01",
              "anharmonicity",
              "drag_coefficient",
              "ac_stark_detuning",
              "pi_length",
              "pi_amp",
              "wiring"
            ]
          },
          "z": {
            "type": "object",
            "title": "z",
            "properties": {
              "wiring": {
                "type": "object",
                "title": "wiring",
                "properties": {
                  "controller": {
                    "type": "string"
                  },
                  "port": {
                    "type": "integer"
                  },
                  "filter": {
                    "type": "object",
                    "title": "filter",
                    "properties": {
                      "iir_taps": {
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
                      "fir_taps": {
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
                      "iir_taps",
                      "fir_taps"
                    ]
                  }
                },
                "required": [
                  "controller",
                  "port",
                  "filter"
                ]
              },
              "flux_pulse_length": {
                "type": "integer"
              },
              "flux_pulse_amp": {
                "type": "number"
              },
              "max_frequency_point": {
                "type": "number"
              },
              "min_frequency_point": {
                "type": "number"
              },
              "iswap": {
                "type": "object",
                "title": "iswap",
                "properties": {
                  "length": {
                    "type": "integer"
                  },
                  "level": {
                    "type": "number"
                  }
                },
                "required": [
                  "length",
                  "level"
                ]
              },
              "cz": {
                "type": "object",
                "title": "cz",
                "properties": {
                  "length": {
                    "type": "integer"
                  },
                  "level": {
                    "type": "number"
                  }
                },
                "required": [
                  "length",
                  "level"
                ]
              }
            },
            "required": [
              "wiring",
              "flux_pulse_length",
              "flux_pulse_amp",
              "max_frequency_point",
              "min_frequency_point",
              "iswap",
              "cz"
            ]
          },
          "ge_threshold": {
            "type": "number"
          },
          "T1": {
            "type": "integer"
          },
          "T2": {
            "type": "integer"
          },
          "T2echo": {
            "type": "integer"
          }
        }"""
        import quam_sdk.crud

        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path + "[].", self._index, new_item=True)
        quam_sdk.crud.bump_list_length(self._quam._json, f"{self._path}[]_len", self._index)

    def __str__(self) -> str:
        return json.dumps(self._json_view())

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Mixer_correction2(object):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "offset_I"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "offset_Q"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "gain"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "phase"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "phase"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Mixer_correction2):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Wiring3(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
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
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "controller")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

    @property
    def I(self) -> List[Union[str, int, float, bool, list]]:
        """"""

        if self._quam._record_updates:
            return _List(self._quam, self._path + "I", self._index, self._schema)

        value = self._quam._json[self._path + "I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @I.setter
    def I(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "I")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "I"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "I"] = value

    @property
    def Q(self) -> List[Union[str, int, float, bool, list]]:
        """"""

        if self._quam._record_updates:
            return _List(self._quam, self._path + "Q", self._index, self._schema)

        value = self._quam._json[self._path + "Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @Q.setter
    def Q(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "Q")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "Q"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "Q"] = value

    @property
    def mixer_correction(self) -> Mixer_correction2:
        """"""
        return Mixer_correction2(
            self._quam, self._path + "mixer_correction.", self._index, self._schema["properties"]["mixer_correction"]
        )

    @mixer_correction.setter
    def mixer_correction(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "mixer_correction", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Wiring3):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
    def name(self) -> str:
        """"""

        value = self._quam._json[self._path + "name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @name.setter
    def name(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def LO_index(self) -> int:
        """"""

        value = self._quam._json[self._path + "LO_index"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @LO_index.setter
    def LO_index(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "LO_index")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "LO_index"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "LO_index"] = value

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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "f_res"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f_res"] = value

    @property
    def f_opt(self) -> float:
        """"""

        value = self._quam._json[self._path + "f_opt"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @f_opt.setter
    def f_opt(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "f_opt")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "f_opt"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f_opt"] = value

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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "depletion_time"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "depletion_time"] = value

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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "readout_pulse_length"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "readout_pulse_amp"]
            for i in range(len(self._index) - 1):
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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "rotation_angle"]
            for i in range(len(self._index) - 1):
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
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "readout_fidelity")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "readout_fidelity"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "readout_fidelity"] = value

    @property
    def wiring(self) -> Wiring3:
        """"""
        return Wiring3(self._quam, self._path + "wiring.", self._index, self._schema["properties"]["wiring"])

    @wiring.setter
    def wiring(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "wiring", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class ResonatorsList(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Resonator:
        return Resonator(self._quam, self._path + "[].", self._index + [key], self._schema["items"])

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

    def _json_view(self, metadata=False):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view(metadata=metadata and i == 0))
        return result

    def append(self, json_item: dict):
        """Adds a new resonator by adding a JSON dictionary with following schema
        {
          "name": {
            "type": "string"
          },
          "LO_index": {
            "type": "integer"
          },
          "f_res": {
            "type": "number"
          },
          "f_opt": {
            "type": "number"
          },
          "depletion_time": {
            "type": "integer"
          },
          "readout_pulse_length": {
            "type": "integer"
          },
          "readout_pulse_amp": {
            "type": "number"
          },
          "rotation_angle": {
            "type": "number"
          },
          "readout_fidelity": {
            "type": "number"
          },
          "wiring": {
            "type": "object",
            "title": "wiring",
            "properties": {
              "controller": {
                "type": "string"
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
              }
            },
            "required": [
              "controller",
              "I",
              "Q",
              "mixer_correction"
            ]
          }
        }"""
        import quam_sdk.crud

        self._schema["items"]["additionalProperties"] = False
        quam_sdk.crud.validate_input(json_item, self._schema["items"])
        if self._quam._record_updates:
            self._quam._updates["items"].append([json_item, self._path, self._index])
        quam_sdk.crud.load_data_to_flat_json(self._quam, json_item, self._path + "[].", self._index, new_item=True)
        quam_sdk.crud.bump_list_length(self._quam._json, f"{self._path}[]_len", self._index)

    def __str__(self) -> str:
        return json.dumps(self._json_view())

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
    def z(self) -> List[Union[str, int, float, bool, list]]:
        """"""

        if self._quam._record_updates:
            return _List(self._quam, self._path + "z", self._index, self._schema)

        value = self._quam._json[self._path + "z"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @z.setter
    def z(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "z")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "z"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "z"] = value

    @property
    def xy(self) -> List[Union[str, int, float, bool, list]]:
        """"""

        if self._quam._record_updates:
            return _List(self._quam, self._path + "xy", self._index, self._schema)

        value = self._quam._json[self._path + "xy"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @xy.setter
    def xy(self, value: List[Union[str, int, float, bool, list]]):
        """"""
        if not isinstance(value, List):
            raise TypeError(f"Expected List[Union[str, int, float, bool, list]] but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "xy")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "xy"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "xy"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)


class Global_parameters(object):
    def __init__(self, quam, path, index, schema):
        """"""
        self._quam: QuAM = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "time_of_flight"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "time_of_flight"] = value

    @property
    def downconversion_offset_I(self) -> float:
        """"""

        value = self._quam._json[self._path + "downconversion_offset_I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @downconversion_offset_I.setter
    def downconversion_offset_I(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "downconversion_offset_I")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "downconversion_offset_I"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "downconversion_offset_I"] = value

    @property
    def downconversion_offset_Q(self) -> float:
        """"""

        value = self._quam._json[self._path + "downconversion_offset_Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value

    @downconversion_offset_Q.setter
    def downconversion_offset_Q(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "downconversion_offset_Q")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if len(self._index) > 0:
            value_ref = self._quam._json[self._path + "downconversion_offset_Q"]
            for i in range(len(self._index) - 1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "downconversion_offset_Q"] = value

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
        import json

        return json.dumps(self._json_view())

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Global_parameters):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
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
        self._updates = {"keys": [], "indexes": [], "values": [], "items": []}
        self._schema_flat = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "name": "QuAM storage format",
            "description": "optimized data structure for communication and storage",
            "type": "object",
            "properties": {
                "local_oscillators.qubits[]_len": {"anyof": [{"type": "array"}, {"type": "integer"}]},
                "local_oscillators.readout[]_len": {"anyof": [{"type": "array"}, {"type": "integer"}]},
                "qubits[]_len": {"anyof": [{"type": "array"}, {"type": "integer"}]},
                "resonators[]_len": {"anyof": [{"type": "array"}, {"type": "integer"}]},
                "network.qop_ip": {"type": "string"},
                "network.cluster_name": {"type": "string"},
                "network.save_dir": {"type": "string"},
                "local_oscillators.network.qop_ip": {"type": "string"},
                "local_oscillators.network.qop_port": {"type": "integer"},
                "local_oscillators.network.save_dir": {"type": "string"},
                "local_oscillators.qubits[].freq": {"type": "array", "items": {"type": "number"}},
                "local_oscillators.qubits[].power": {"type": "array", "items": {"type": "integer"}},
                "local_oscillators.readout[].freq": {"type": "array", "items": {"type": "number"}},
                "local_oscillators.readout[].power": {"type": "array", "items": {"type": "integer"}},
                "qubits[].name": {"type": "array", "items": {"type": "string"}},
                "qubits[].xy.LO_index": {"type": "array", "items": {"type": "integer"}},
                "qubits[].xy.f_01": {"type": "array", "items": {"type": "number"}},
                "qubits[].xy.anharmonicity": {"type": "array", "items": {"type": "number"}},
                "qubits[].xy.drag_coefficient": {"type": "array", "items": {"type": "number"}},
                "qubits[].xy.ac_stark_detuning": {"type": "array", "items": {"type": "number"}},
                "qubits[].xy.pi_length": {"type": "array", "items": {"type": "integer"}},
                "qubits[].xy.pi_amp": {"type": "array", "items": {"type": "number"}},
                "qubits[].xy.wiring.controller": {"type": "array", "items": {"type": "string"}},
                "qubits[].xy.wiring.I": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"},
                                {"type": "array"},
                            ]
                        },
                    },
                },
                "qubits[].xy.wiring.Q": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"},
                                {"type": "array"},
                            ]
                        },
                    },
                },
                "qubits[].xy.wiring.mixer_correction.offset_I": {"type": "array", "items": {"type": "number"}},
                "qubits[].xy.wiring.mixer_correction.offset_Q": {"type": "array", "items": {"type": "number"}},
                "qubits[].xy.wiring.mixer_correction.gain": {"type": "array", "items": {"type": "number"}},
                "qubits[].xy.wiring.mixer_correction.phase": {"type": "array", "items": {"type": "number"}},
                "qubits[].z.wiring.controller": {"type": "array", "items": {"type": "string"}},
                "qubits[].z.wiring.port": {"type": "array", "items": {"type": "integer"}},
                "qubits[].z.wiring.filter.iir_taps": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"},
                                {"type": "array"},
                            ]
                        },
                    },
                },
                "qubits[].z.wiring.filter.fir_taps": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"},
                                {"type": "array"},
                            ]
                        },
                    },
                },
                "qubits[].z.flux_pulse_length": {"type": "array", "items": {"type": "integer"}},
                "qubits[].z.flux_pulse_amp": {"type": "array", "items": {"type": "number"}},
                "qubits[].z.max_frequency_point": {"type": "array", "items": {"type": "number"}},
                "qubits[].z.min_frequency_point": {"type": "array", "items": {"type": "number"}},
                "qubits[].z.iswap.length": {"type": "array", "items": {"type": "integer"}},
                "qubits[].z.iswap.level": {"type": "array", "items": {"type": "number"}},
                "qubits[].z.cz.length": {"type": "array", "items": {"type": "integer"}},
                "qubits[].z.cz.level": {"type": "array", "items": {"type": "number"}},
                "qubits[].ge_threshold": {"type": "array", "items": {"type": "number"}},
                "qubits[].T1": {"type": "array", "items": {"type": "integer"}},
                "qubits[].T2": {"type": "array", "items": {"type": "integer"}},
                "qubits[].T2echo": {"type": "array", "items": {"type": "integer"}},
                "resonators[].name": {"type": "array", "items": {"type": "string"}},
                "resonators[].LO_index": {"type": "array", "items": {"type": "integer"}},
                "resonators[].f_res": {"type": "array", "items": {"type": "number"}},
                "resonators[].f_opt": {"type": "array", "items": {"type": "number"}},
                "resonators[].depletion_time": {"type": "array", "items": {"type": "integer"}},
                "resonators[].readout_pulse_length": {"type": "array", "items": {"type": "integer"}},
                "resonators[].readout_pulse_amp": {"type": "array", "items": {"type": "number"}},
                "resonators[].rotation_angle": {"type": "array", "items": {"type": "number"}},
                "resonators[].readout_fidelity": {"type": "array", "items": {"type": "number"}},
                "resonators[].wiring.controller": {"type": "array", "items": {"type": "string"}},
                "resonators[].wiring.I": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"},
                                {"type": "array"},
                            ]
                        },
                    },
                },
                "resonators[].wiring.Q": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "integer"},
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"},
                                {"type": "array"},
                            ]
                        },
                    },
                },
                "resonators[].wiring.mixer_correction.offset_I": {"type": "array", "items": {"type": "number"}},
                "resonators[].wiring.mixer_correction.offset_Q": {"type": "array", "items": {"type": "number"}},
                "resonators[].wiring.mixer_correction.gain": {"type": "array", "items": {"type": "number"}},
                "resonators[].wiring.mixer_correction.phase": {"type": "array", "items": {"type": "number"}},
                "crosstalk.z": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "number"},
                            {"type": "string"},
                            {"type": "boolean"},
                            {"type": "array"},
                        ]
                    },
                },
                "crosstalk.xy": {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "number"},
                            {"type": "string"},
                            {"type": "boolean"},
                            {"type": "array"},
                        ]
                    },
                },
                "global_parameters.time_of_flight": {"type": "integer"},
                "global_parameters.downconversion_offset_I": {"type": "number"},
                "global_parameters.downconversion_offset_Q": {"type": "number"},
            },
            "additionalProperties": False,
        }
        self._schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "title": "QuAM",
            "properties": {
                "network": {
                    "type": "object",
                    "title": "network",
                    "properties": {
                        "qop_ip": {"type": "string"},
                        "cluster_name": {"type": "string"},
                        "save_dir": {"type": "string"},
                    },
                    "required": ["qop_ip", "cluster_name", "save_dir"],
                },
                "local_oscillators": {
                    "type": "object",
                    "title": "local_oscillators",
                    "properties": {
                        "network": {
                            "type": "object",
                            "title": "network",
                            "properties": {
                                "qop_ip": {"type": "string"},
                                "qop_port": {"type": "integer"},
                                "save_dir": {"type": "string"},
                            },
                            "required": ["qop_ip", "qop_port", "save_dir"],
                        },
                        "qubits": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "title": "qubit",
                                "properties": {"freq": {"type": "number"}, "power": {"type": "integer"}},
                                "required": ["freq", "power"],
                            },
                        },
                        "readout": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "title": "readout",
                                "properties": {"freq": {"type": "number"}, "power": {"type": "integer"}},
                                "required": ["freq", "power"],
                            },
                        },
                    },
                    "required": ["network", "qubits", "readout"],
                },
                "qubits": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "title": "qubit",
                        "properties": {
                            "name": {"type": "string"},
                            "xy": {
                                "type": "object",
                                "title": "xy",
                                "properties": {
                                    "LO_index": {"type": "integer"},
                                    "f_01": {"type": "number"},
                                    "anharmonicity": {"type": "number"},
                                    "drag_coefficient": {"type": "number"},
                                    "ac_stark_detuning": {"type": "number"},
                                    "pi_length": {"type": "integer"},
                                    "pi_amp": {"type": "number"},
                                    "wiring": {
                                        "type": "object",
                                        "title": "wiring",
                                        "properties": {
                                            "controller": {"type": "string"},
                                            "I": {
                                                "type": "array",
                                                "items": {
                                                    "anyOf": [
                                                        {"type": "integer"},
                                                        {"type": "number"},
                                                        {"type": "string"},
                                                        {"type": "boolean"},
                                                        {"type": "array"},
                                                    ]
                                                },
                                            },
                                            "Q": {
                                                "type": "array",
                                                "items": {
                                                    "anyOf": [
                                                        {"type": "integer"},
                                                        {"type": "number"},
                                                        {"type": "string"},
                                                        {"type": "boolean"},
                                                        {"type": "array"},
                                                    ]
                                                },
                                            },
                                            "mixer_correction": {
                                                "type": "object",
                                                "title": "mixer_correction",
                                                "properties": {
                                                    "offset_I": {"type": "number"},
                                                    "offset_Q": {"type": "number"},
                                                    "gain": {"type": "number"},
                                                    "phase": {"type": "number"},
                                                },
                                                "required": ["offset_I", "offset_Q", "gain", "phase"],
                                            },
                                        },
                                        "required": ["controller", "I", "Q", "mixer_correction"],
                                    },
                                },
                                "required": [
                                    "LO_index",
                                    "f_01",
                                    "anharmonicity",
                                    "drag_coefficient",
                                    "ac_stark_detuning",
                                    "pi_length",
                                    "pi_amp",
                                    "wiring",
                                ],
                            },
                            "z": {
                                "type": "object",
                                "title": "z",
                                "properties": {
                                    "wiring": {
                                        "type": "object",
                                        "title": "wiring",
                                        "properties": {
                                            "controller": {"type": "string"},
                                            "port": {"type": "integer"},
                                            "filter": {
                                                "type": "object",
                                                "title": "filter",
                                                "properties": {
                                                    "iir_taps": {
                                                        "type": "array",
                                                        "items": {
                                                            "anyOf": [
                                                                {"type": "integer"},
                                                                {"type": "number"},
                                                                {"type": "string"},
                                                                {"type": "boolean"},
                                                                {"type": "array"},
                                                            ]
                                                        },
                                                    },
                                                    "fir_taps": {
                                                        "type": "array",
                                                        "items": {
                                                            "anyOf": [
                                                                {"type": "integer"},
                                                                {"type": "number"},
                                                                {"type": "string"},
                                                                {"type": "boolean"},
                                                                {"type": "array"},
                                                            ]
                                                        },
                                                    },
                                                },
                                                "required": ["iir_taps", "fir_taps"],
                                            },
                                        },
                                        "required": ["controller", "port", "filter"],
                                    },
                                    "flux_pulse_length": {"type": "integer"},
                                    "flux_pulse_amp": {"type": "number"},
                                    "max_frequency_point": {"type": "number"},
                                    "min_frequency_point": {"type": "number"},
                                    "iswap": {
                                        "type": "object",
                                        "title": "iswap",
                                        "properties": {"length": {"type": "integer"}, "level": {"type": "number"}},
                                        "required": ["length", "level"],
                                    },
                                    "cz": {
                                        "type": "object",
                                        "title": "cz",
                                        "properties": {"length": {"type": "integer"}, "level": {"type": "number"}},
                                        "required": ["length", "level"],
                                    },
                                },
                                "required": [
                                    "wiring",
                                    "flux_pulse_length",
                                    "flux_pulse_amp",
                                    "max_frequency_point",
                                    "min_frequency_point",
                                    "iswap",
                                    "cz",
                                ],
                            },
                            "ge_threshold": {"type": "number"},
                            "T1": {"type": "integer"},
                            "T2": {"type": "integer"},
                            "T2echo": {"type": "integer"},
                        },
                        "required": ["name", "xy", "z", "ge_threshold", "T1", "T2", "T2echo"],
                    },
                },
                "resonators": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "title": "resonator",
                        "properties": {
                            "name": {"type": "string"},
                            "LO_index": {"type": "integer"},
                            "f_res": {"type": "number"},
                            "f_opt": {"type": "number"},
                            "depletion_time": {"type": "integer"},
                            "readout_pulse_length": {"type": "integer"},
                            "readout_pulse_amp": {"type": "number"},
                            "rotation_angle": {"type": "number"},
                            "readout_fidelity": {"type": "number"},
                            "wiring": {
                                "type": "object",
                                "title": "wiring",
                                "properties": {
                                    "controller": {"type": "string"},
                                    "I": {
                                        "type": "array",
                                        "items": {
                                            "anyOf": [
                                                {"type": "integer"},
                                                {"type": "number"},
                                                {"type": "string"},
                                                {"type": "boolean"},
                                                {"type": "array"},
                                            ]
                                        },
                                    },
                                    "Q": {
                                        "type": "array",
                                        "items": {
                                            "anyOf": [
                                                {"type": "integer"},
                                                {"type": "number"},
                                                {"type": "string"},
                                                {"type": "boolean"},
                                                {"type": "array"},
                                            ]
                                        },
                                    },
                                    "mixer_correction": {
                                        "type": "object",
                                        "title": "mixer_correction",
                                        "properties": {
                                            "offset_I": {"type": "number"},
                                            "offset_Q": {"type": "number"},
                                            "gain": {"type": "number"},
                                            "phase": {"type": "number"},
                                        },
                                        "required": ["offset_I", "offset_Q", "gain", "phase"],
                                    },
                                },
                                "required": ["controller", "I", "Q", "mixer_correction"],
                            },
                        },
                        "required": [
                            "name",
                            "LO_index",
                            "f_res",
                            "f_opt",
                            "depletion_time",
                            "readout_pulse_length",
                            "readout_pulse_amp",
                            "rotation_angle",
                            "readout_fidelity",
                            "wiring",
                        ],
                    },
                },
                "crosstalk": {
                    "type": "object",
                    "title": "crosstalk",
                    "properties": {
                        "z": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "integer"},
                                    {"type": "number"},
                                    {"type": "string"},
                                    {"type": "boolean"},
                                    {"type": "array"},
                                ]
                            },
                        },
                        "xy": {
                            "type": "array",
                            "items": {
                                "anyOf": [
                                    {"type": "integer"},
                                    {"type": "number"},
                                    {"type": "string"},
                                    {"type": "boolean"},
                                    {"type": "array"},
                                ]
                            },
                        },
                    },
                    "required": ["z", "xy"],
                },
                "global_parameters": {
                    "type": "object",
                    "title": "global_parameters",
                    "properties": {
                        "time_of_flight": {"type": "integer"},
                        "downconversion_offset_I": {"type": "number"},
                        "downconversion_offset_Q": {"type": "number"},
                    },
                    "required": ["time_of_flight", "downconversion_offset_I", "downconversion_offset_Q"],
                },
            },
            "required": ["network", "local_oscillators", "qubits", "resonators", "crosstalk", "global_parameters"],
        }
        self._runtime_var = (
            dict()
        )  #: scratchpad dictionary of runtime variables for user's convenience. These are not saved when exporting data.
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
                quam_sdk.crud.load_data_to_flat_json(self, data, key_structure="", index=[])
            quam_sdk.crud.validate_input(self._json, self._schema_flat)
        else:
            self._json = None
        self._freeze_attributes = True

    def _reset_update_record(self):
        """Resets self._updates record, but does not undo updates to QuAM
        data in self._json"""
        self._updates = {"keys": [], "indexes": [], "values": [], "items": []}

    def _add_updates(self, updates: dict):
        """Adds updates generated as another QuAM instance self._updates.
        See also `_reset_update_record` and `self._updates`
        """
        for j in range(len(updates["keys"])):
            if len(updates["indexes"][j]) > 0:
                value_ref = self._quam._json[updates["keys"][j]]
                for i in range(len(updates["indexes"][j]) - 1):
                    value_ref = value_ref[updates["indexes"][j][i]]
                value_ref[updates["indexes"][j][-1]] = updates["values"][j]
            else:
                self._quam._json[updates["keys"][j]] = updates["values"][j]

        import quam_sdk.crud

        for item in updates["items"]:
            quam_sdk.crud.load_data_to_flat_json(self._quam, item[0], item[1] + "[].", item[2], new_item=True)
            self._quam._json[f"{item[1]}[]_len"] += 1

        if self._record_updates:
            self._updates["keys"] += updates["keys"]
            self._updates["indexes"] += updates["indexes"]
            self._updates["values"] += updates["values"]
            self._updates["items"] += updates["items"]

    def _save(self, filename: str, flat_data=True):
        """Saves QuAM data into a given filename. Supported file types: json

        Args:
            filename: filename where to save data
            flat_data: optional, should saved data be saved in flattened json
                optimized for storage. Defaults to True.
        """
        filetype = filename.split(".")[-1]
        if filetype in ["json", "JSON"]:
            from quam_sdk.crud import pretty_print_json

            with open(filename, "w") as file:
                if flat_data:
                    file.write(pretty_print_json(self._json))
                else:
                    # structured data output
                    file.write(pretty_print_json(self._json_view(metadata=True)))
        else:
            raise ValueError(f"Unsupported file type {filetype}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._json)})"

    @property
    def qubits(self) -> QubitsList2:
        """"""
        return QubitsList2(self._quam, self._path + "qubits", self._index, self._schema["properties"]["qubits"])

    @property
    def resonators(self) -> ResonatorsList:
        """"""
        return ResonatorsList(
            self._quam, self._path + "resonators", self._index, self._schema["properties"]["resonators"]
        )

    @property
    def network(self) -> Network:
        """"""
        return Network(self._quam, self._path + "network.", self._index, self._schema["properties"]["network"])

    @network.setter
    def network(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "network", value)

    @property
    def local_oscillators(self) -> Local_oscillators:
        """"""
        return Local_oscillators(
            self._quam, self._path + "local_oscillators.", self._index, self._schema["properties"]["local_oscillators"]
        )

    @local_oscillators.setter
    def local_oscillators(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "local_oscillators", value)

    @property
    def crosstalk(self) -> Crosstalk:
        """"""
        return Crosstalk(self._quam, self._path + "crosstalk.", self._index, self._schema["properties"]["crosstalk"])

    @crosstalk.setter
    def crosstalk(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "crosstalk", value)

    @property
    def global_parameters(self) -> Global_parameters:
        """"""
        return Global_parameters(
            self._quam, self._path + "global_parameters.", self._index, self._schema["properties"]["global_parameters"]
        )

    @global_parameters.setter
    def global_parameters(self, value: dict):
        import quam_sdk.crud

        quam_sdk.crud.replace_layer_in_quam_data(self, "global_parameters", value)

    def _repr_pretty_(self, p, cycle) -> str:
        st = ""
        from quam_sdk.crud import _get_units

        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata: bool = False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self, v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""):
                        result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[") + 1 : -1])

        if metadata:
            if len(functions) > 0:
                result["_func"] = functions

            if self.__init__.__doc__ not in (None, ""):
                result["_docs"] = self.__init__.__doc__[:-212]

        return result

    def __str__(self) -> str:
        if self._quam._json is None:
            raise ValueError("No data about Quantum Abstract Machine (QuAM) " "has been loaded. Aborting printing.")
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
            raise TypeError(
                f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system stete used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor"
            )
        object.__setattr__(self, key, value)
