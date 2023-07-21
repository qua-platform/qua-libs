# QuAM class automatically generated using QuAM SDK (ver 0.9.1)
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

    def _json_view(self, metadata:bool=False):
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
    def octave1_ip(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "octave1_ip"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @octave1_ip.setter
    def octave1_ip(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "octave1_ip")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "octave1_ip"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "octave1_ip"] = value

    @property
    def octave2_ip(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "octave2_ip"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @octave2_ip.setter
    def octave2_ip(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "octave2_ip")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "octave2_ip"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "octave2_ip"] = value

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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "cluster_name"]
            for i in range(len(self._index)-1):
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "save_dir"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "save_dir"] = value

    def _repr_pretty_(self, p, cycle)->str:
        st = ""
        from quam_sdk.crud import _get_units
        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata:bool=False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[")+1: -1])

        if metadata:
            if len(functions)>0:
                result["_func"] = functions

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
                " attributes, please update system state used for automatic\n"
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

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

    def _repr_pretty_(self, p, cycle)->str:
        st = ""
        from quam_sdk.crud import _get_units
        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata:bool=False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[")+1: -1])

        if metadata:
            if len(functions)>0:
                result["_func"] = functions

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
                " attributes, please update system state used for automatic\n"
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "f_01"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f_01"] = value

    @property
    def lo(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "lo"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @lo.setter
    def lo(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "lo")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "lo"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "lo"] = value

    @property
    def rf_gain(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "rf_gain"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rf_gain.setter
    def rf_gain(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "rf_gain")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "rf_gain"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "rf_gain"] = value

    @property
    def rf_switch_mode(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "rf_switch_mode"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rf_switch_mode.setter
    def rf_switch_mode(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "rf_switch_mode")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "rf_switch_mode"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "rf_switch_mode"] = value

    @property
    def mixer_name(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "mixer_name"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @mixer_name.setter
    def mixer_name(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "mixer_name")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "mixer_name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "mixer_name"] = value

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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "anharmonicity"]
            for i in range(len(self._index)-1):
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "drag_coefficient"]
            for i in range(len(self._index)-1):
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ac_stark_detuning"]
            for i in range(len(self._index)-1):
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "pi_length"]
            for i in range(len(self._index)-1):
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "pi_amp"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "pi_amp"] = value

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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "T1"]
            for i in range(len(self._index)-1):
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "T2"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "T2"] = value

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
    
    def _repr_pretty_(self, p, cycle)->str:
        st = ""
        from quam_sdk.crud import _get_units
        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata:bool=False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[")+1: -1])

        if metadata:
            if len(functions)>0:
                result["_func"] = functions

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
                " attributes, please update system state used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
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

    def _json_view(self, metadata=False):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view(metadata=metadata and i==0))
        return result

    def append(self, json_item:dict):
        """Adds a new qubit by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "f_01": {
    "type": "number"
  },
  "lo": {
    "type": "number"
  },
  "rf_gain": {
    "type": "integer"
  },
  "rf_switch_mode": {
    "type": "string"
  },
  "mixer_name": {
    "type": "string"
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
        "type": "integer"
      },
      "Q": {
        "type": "integer"
      }
    },
    "required": [
      "controller",
      "I",
      "Q"
    ]
  },
  "T1": {
    "type": "integer"
  },
  "T2": {
    "type": "integer"
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "controller"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "controller"] = value

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

    def _repr_pretty_(self, p, cycle)->str:
        st = ""
        from quam_sdk.crud import _get_units
        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata:bool=False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[")+1: -1])

        if metadata:
            if len(functions)>0:
                result["_func"] = functions

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
                " attributes, please update system state used for automatic\n"
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "name"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "name"] = value

    @property
    def f_readout(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "f_readout"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @f_readout.setter
    def f_readout(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "f_readout")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "f_readout"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "f_readout"] = value

    @property
    def lo(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "lo"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @lo.setter
    def lo(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "lo")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "lo"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "lo"] = value

    @property
    def rf_gain(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "rf_gain"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rf_gain.setter
    def rf_gain(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "rf_gain")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "rf_gain"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "rf_gain"] = value

    @property
    def rf_switch_mode(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "rf_switch_mode"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @rf_switch_mode.setter
    def rf_switch_mode(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "rf_switch_mode")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "rf_switch_mode"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "rf_switch_mode"] = value

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
    def optimal_pulse_length(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "optimal_pulse_length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @optimal_pulse_length.setter
    def optimal_pulse_length(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "optimal_pulse_length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "optimal_pulse_length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "optimal_pulse_length"] = value

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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ge_threshold"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ge_threshold"] = value

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
    
    def _repr_pretty_(self, p, cycle)->str:
        st = ""
        from quam_sdk.crud import _get_units
        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata:bool=False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[")+1: -1])

        if metadata:
            if len(functions)>0:
                result["_func"] = functions

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
                " attributes, please update system state used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
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

    def _json_view(self, metadata=False):
        result = []
        for i in range(self.__len__()):
            result.append(self.__getitem__(i)._json_view(metadata=metadata and i==0))
        return result

    def append(self, json_item:dict):
        """Adds a new resonator by adding a JSON dictionary with following schema
{
  "name": {
    "type": "string"
  },
  "f_readout": {
    "type": "number"
  },
  "lo": {
    "type": "number"
  },
  "rf_gain": {
    "type": "integer"
  },
  "rf_switch_mode": {
    "type": "string"
  },
  "depletion_time": {
    "type": "integer"
  },
  "readout_pulse_length": {
    "type": "integer"
  },
  "optimal_pulse_length": {
    "type": "integer"
  },
  "readout_pulse_amp": {
    "type": "number"
  },
  "rotation_angle": {
    "type": "number"
  },
  "ge_threshold": {
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
        "type": "integer"
      },
      "Q": {
        "type": "integer"
      }
    },
    "required": [
      "controller",
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
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "time_of_flight"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "time_of_flight"] = value

    @property
    def saturation_amp(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "saturation_amp"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @saturation_amp.setter
    def saturation_amp(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "saturation_amp")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "saturation_amp"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "saturation_amp"] = value

    @property
    def saturation_len(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "saturation_len"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @saturation_len.setter
    def saturation_len(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "saturation_len")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "saturation_len"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "saturation_len"] = value

    @property
    def con1_downconversion_offset_I(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "con1_downconversion_offset_I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con1_downconversion_offset_I.setter
    def con1_downconversion_offset_I(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con1_downconversion_offset_I")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con1_downconversion_offset_I"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con1_downconversion_offset_I"] = value

    @property
    def con1_downconversion_offset_Q(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "con1_downconversion_offset_Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con1_downconversion_offset_Q.setter
    def con1_downconversion_offset_Q(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con1_downconversion_offset_Q")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con1_downconversion_offset_Q"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con1_downconversion_offset_Q"] = value

    @property
    def con1_downconversion_gain(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "con1_downconversion_gain"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con1_downconversion_gain.setter
    def con1_downconversion_gain(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con1_downconversion_gain")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con1_downconversion_gain"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con1_downconversion_gain"] = value

    @property
    def con2_downconversion_offset_I(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "con2_downconversion_offset_I"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con2_downconversion_offset_I.setter
    def con2_downconversion_offset_I(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con2_downconversion_offset_I")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con2_downconversion_offset_I"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con2_downconversion_offset_I"] = value

    @property
    def con2_downconversion_offset_Q(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "con2_downconversion_offset_Q"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con2_downconversion_offset_Q.setter
    def con2_downconversion_offset_Q(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con2_downconversion_offset_Q")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con2_downconversion_offset_Q"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con2_downconversion_offset_Q"] = value

    @property
    def con2_downconversion_gain(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "con2_downconversion_gain"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con2_downconversion_gain.setter
    def con2_downconversion_gain(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con2_downconversion_gain")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con2_downconversion_gain"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con2_downconversion_gain"] = value

    def _repr_pretty_(self, p, cycle)->str:
        st = ""
        from quam_sdk.crud import _get_units
        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata:bool=False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[")+1: -1])

        if metadata:
            if len(functions)>0:
                result["_func"] = functions

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
        if isinstance(other, Global_parameters):
            return f"{self._path}{self._index}" == f"{other._path}{other._index}"
        return NotImplemented

    def __setattr__(self, key, value):
        if hasattr(self, "_freeze_attributes") and not hasattr(self, key):
            raise TypeError(f"One cannot add non-existing attribute '{key}'"
                " to Quantum Abstract Machine (QuAM).\n"
                " If you want to change available"
                " attributes, please update system state used for automatic\n"
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
        self._schema_flat = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'name': 'QuAM storage format', 'description': 'optimized data structure for communication and storage', 'type': 'object', 'properties': {'qubits[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'resonators[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'network.qop_ip': {'type': 'string'}, 'network.octave1_ip': {'type': 'string'}, 'network.octave2_ip': {'type': 'string'}, 'network.qop_port': {'type': 'integer'}, 'network.cluster_name': {'type': 'string'}, 'network.save_dir': {'type': 'string'}, 'qubits[].name': {'type': 'array', 'items': {'type': 'string'}}, 'qubits[].f_01': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[].lo': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[].rf_gain': {'type': 'array', 'items': {'type': 'integer'}}, 'qubits[].rf_switch_mode': {'type': 'array', 'items': {'type': 'string'}}, 'qubits[].mixer_name': {'type': 'array', 'items': {'type': 'string'}}, 'qubits[].anharmonicity': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[].drag_coefficient': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[].ac_stark_detuning': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[].pi_length': {'type': 'array', 'items': {'type': 'integer'}}, 'qubits[].pi_amp': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[].wiring.controller': {'type': 'array', 'items': {'type': 'string'}}, 'qubits[].wiring.I': {'type': 'array', 'items': {'type': 'integer'}}, 'qubits[].wiring.Q': {'type': 'array', 'items': {'type': 'integer'}}, 'qubits[].T1': {'type': 'array', 'items': {'type': 'integer'}}, 'qubits[].T2': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].name': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[].f_readout': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].lo': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].rf_gain': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].rf_switch_mode': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[].depletion_time': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].readout_pulse_length': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].optimal_pulse_length': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].readout_pulse_amp': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].rotation_angle': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].ge_threshold': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[].wiring.controller': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[].wiring.I': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[].wiring.Q': {'type': 'array', 'items': {'type': 'integer'}}, 'global_parameters.time_of_flight': {'type': 'integer'}, 'global_parameters.saturation_amp': {'type': 'number'}, 'global_parameters.saturation_len': {'type': 'integer'}, 'global_parameters.con1_downconversion_offset_I': {'type': 'number'}, 'global_parameters.con1_downconversion_offset_Q': {'type': 'number'}, 'global_parameters.con1_downconversion_gain': {'type': 'integer'}, 'global_parameters.con2_downconversion_offset_I': {'type': 'number'}, 'global_parameters.con2_downconversion_offset_Q': {'type': 'number'}, 'global_parameters.con2_downconversion_gain': {'type': 'integer'}}, 'additionalProperties': False}
        self._schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'title': 'QuAM', 'properties': {'network': {'type': 'object', 'title': 'network', 'properties': {'qop_ip': {'type': 'string'}, 'octave1_ip': {'type': 'string'}, 'octave2_ip': {'type': 'string'}, 'qop_port': {'type': 'integer'}, 'cluster_name': {'type': 'string'}, 'save_dir': {'type': 'string'}}, 'required': ['qop_ip', 'octave1_ip', 'octave2_ip', 'qop_port', 'cluster_name', 'save_dir']}, 'qubits': {'type': 'array', 'items': {'type': 'object', 'title': 'qubit', 'properties': {'name': {'type': 'string'}, 'f_01': {'type': 'number'}, 'lo': {'type': 'number'}, 'rf_gain': {'type': 'integer'}, 'rf_switch_mode': {'type': 'string'}, 'mixer_name': {'type': 'string'}, 'anharmonicity': {'type': 'number'}, 'drag_coefficient': {'type': 'number'}, 'ac_stark_detuning': {'type': 'number'}, 'pi_length': {'type': 'integer'}, 'pi_amp': {'type': 'number'}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'controller': {'type': 'string'}, 'I': {'type': 'integer'}, 'Q': {'type': 'integer'}}, 'required': ['controller', 'I', 'Q']}, 'T1': {'type': 'integer'}, 'T2': {'type': 'integer'}}, 'required': ['name', 'f_01', 'lo', 'rf_gain', 'rf_switch_mode', 'mixer_name', 'anharmonicity', 'drag_coefficient', 'ac_stark_detuning', 'pi_length', 'pi_amp', 'wiring', 'T1', 'T2']}}, 'resonators': {'type': 'array', 'items': {'type': 'object', 'title': 'resonator', 'properties': {'name': {'type': 'string'}, 'f_readout': {'type': 'number'}, 'lo': {'type': 'number'}, 'rf_gain': {'type': 'integer'}, 'rf_switch_mode': {'type': 'string'}, 'depletion_time': {'type': 'integer'}, 'readout_pulse_length': {'type': 'integer'}, 'optimal_pulse_length': {'type': 'integer'}, 'readout_pulse_amp': {'type': 'number'}, 'rotation_angle': {'type': 'number'}, 'ge_threshold': {'type': 'number'}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'controller': {'type': 'string'}, 'I': {'type': 'integer'}, 'Q': {'type': 'integer'}}, 'required': ['controller', 'I', 'Q']}}, 'required': ['name', 'f_readout', 'lo', 'rf_gain', 'rf_switch_mode', 'depletion_time', 'readout_pulse_length', 'optimal_pulse_length', 'readout_pulse_amp', 'rotation_angle', 'ge_threshold', 'wiring']}}, 'global_parameters': {'type': 'object', 'title': 'global_parameters', 'properties': {'time_of_flight': {'type': 'integer'}, 'saturation_amp': {'type': 'number'}, 'saturation_len': {'type': 'integer'}, 'con1_downconversion_offset_I': {'type': 'number'}, 'con1_downconversion_offset_Q': {'type': 'number'}, 'con1_downconversion_gain': {'type': 'integer'}, 'con2_downconversion_offset_I': {'type': 'number'}, 'con2_downconversion_offset_Q': {'type': 'number'}, 'con2_downconversion_gain': {'type': 'integer'}}, 'required': ['time_of_flight', 'saturation_amp', 'saturation_len', 'con1_downconversion_offset_I', 'con1_downconversion_offset_Q', 'con1_downconversion_gain', 'con2_downconversion_offset_I', 'con2_downconversion_offset_Q', 'con2_downconversion_gain']}}, 'required': ['network', 'qubits', 'resonators', 'global_parameters']}
        self._runtime_var = dict()  #: scratchpad dictionary of runtime variables for user's convenience. These are not saved when exporting data.
        if data is not None:
            if type(data) is str:
                if len(data)<=5:
                    raise ValueError(f"Unsupported data format {data}")
                if (data[-5:].lower()==".json"):
                    with open(data, "r") as file:
                        data = json.load(file)
                elif (data[-5:].lower()==".yaml"):
                    import yaml
                    from yaml.loader import SafeLoader
                    with open(data, "r") as file:
                        data = yaml.load(file, Loader=SafeLoader)
                else:
                    raise ValueError(f"Unsupported data format {data}.\n"
                        "Supported formats .json and .yaml")
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
        if (filetype.lower() == "json"):
            from quam_sdk.crud import pretty_print_json
            with open(filename, "w") as file:
                if flat_data:
                    file.write(pretty_print_json(self._json))
                else:
                    # structured data output
                    file.write(pretty_print_json(self._json_view(metadata=True)))
        elif (filetype.lower() == "yaml"):
            import yaml
            with open(filename, "w") as file:
                if flat_data:
                    yaml.dump(self._json, file)
                else:
                    yaml.dump(self._json_view(metadata=True), file)
        else:
            raise ValueError(f"Unsupported file type {filetype}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self._json)})"

    @property
    def qubits(self) -> QubitsList:
        """"""
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
    def global_parameters(self) -> Global_parameters:
        """"""
        return Global_parameters(
            self._quam, self._path + "global_parameters.", self._index,
            self._schema["properties"]["global_parameters"]
        )
    @global_parameters.setter
    def global_parameters(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "global_parameters", value)
    
    def _repr_pretty_(self, p, cycle)->str:
        st = ""
        from quam_sdk.crud import _get_units
        for k in self._schema["required"]:
            units = _get_units(self._schema["properties"], k)
            st += f"{k}: {getattr(self, k)} {units}\n"
        return p.text(st)

    def _json_view(self, metadata:bool=False):
        result = {}
        if metadata:
            functions = []
        for v in [func for func in dir(self) if not func.startswith("_")]:
            value = getattr(self,v)
            if type(value) in [str, int, float, None, list, bool]:
                result[v] = value
                if metadata:
                    doc = getattr(self.__class__, v).__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif not callable(value):
                result[v] = value._json_view(metadata=metadata)
                if metadata:
                    doc = getattr(self, v).__init__.__doc__
                    if doc not in (None, ""): result[f"{v}_docs"] = doc
            elif metadata:
                doc = getattr(self.__class__, v).__doc__
                if doc[-1] == "]":
                    functions.append(doc[doc.rfind("[")+1: -1])

        if metadata:
            if len(functions)>0:
                result["_func"] = functions

            if self.__init__.__doc__ not in (None, ""):
                if len(self.__init__.__doc__) > 212:
                    result["_docs"] = self.__init__.__doc__[:-212]

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
                " attributes, please update system state used for automatic\n"
                " generation of QuAM class via quam_sdk.quamConstructor")
        object.__setattr__(self, key, value)


