# QuAM class automatically generated using QuAM SDK (ver 0.7.2)
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


class Amplitude(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def minus_x90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "minus_x90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @minus_x90.setter
    def minus_x90(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "minus_x90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "minus_x90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "minus_x90"] = value

    @property
    def minus_y90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "minus_y90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @minus_y90.setter
    def minus_y90(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "minus_y90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "minus_y90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "minus_y90"] = value

    @property
    def x180(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "x180"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @x180.setter
    def x180(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "x180")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "x180"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "x180"] = value

    @property
    def x90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "x90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @x90.setter
    def x90(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "x90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "x90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "x90"] = value

    @property
    def y180(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "y180"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @y180.setter
    def y180(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "y180")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "y180"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "y180"] = value

    @property
    def y90(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "y90"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @y90.setter
    def y90(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "y90")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "y90"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "y90"] = value

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
        if isinstance(other, Amplitude):
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


class Pulse_params(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ac_stark_shift(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "ac_stark_shift"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ac_stark_shift.setter
    def ac_stark_shift(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ac_stark_shift")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ac_stark_shift"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ac_stark_shift"] = value

    @property
    def drag_coef(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "drag_coef"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @drag_coef.setter
    def drag_coef(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "drag_coef")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "drag_coef"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "drag_coef"] = value

    @property
    def length(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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
    def saturation_len(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "saturation_len"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @saturation_len.setter
    def saturation_len(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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
    def short_pi_amp(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "short_pi_amp"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @short_pi_amp.setter
    def short_pi_amp(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "short_pi_amp")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "short_pi_amp"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "short_pi_amp"] = value

    @property
    def short_pi_len(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "short_pi_len"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @short_pi_len.setter
    def short_pi_len(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "short_pi_len")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "short_pi_len"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "short_pi_len"] = value

    @property
    def amplitude(self) -> Amplitude:
        """"""
        return Amplitude(
            self._quam, self._path + "amplitude/", self._index,
            self._schema["properties"]["amplitude"]
        )
    @amplitude.setter
    def amplitude(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "amplitude", value)
    
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
        if isinstance(other, Pulse_params):
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


class I(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ao(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "ao"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ao.setter
    def ao(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ao")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ao"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ao"] = value

    @property
    def con(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "con"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con.setter
    def con(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con"] = value

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
        if isinstance(other, I):
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


class Q(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ao(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "ao"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ao.setter
    def ao(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ao")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ao"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ao"] = value

    @property
    def con(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "con"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con.setter
    def con(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con"] = value

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
        if isinstance(other, Q):
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
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def I(self) -> I:
        """"""
        return I(
            self._quam, self._path + "I/", self._index,
            self._schema["properties"]["I"]
        )
    @I.setter
    def I(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "I", value)
    
    @property
    def Q(self) -> Q:
        """"""
        return Q(
            self._quam, self._path + "Q/", self._index,
            self._schema["properties"]["Q"]
        )
    @Q.setter
    def Q(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "Q", value)
    
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
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
    def T2(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "T2"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @T2.setter
    def T2(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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
    def storage_chi(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "storage_chi"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @storage_chi.setter
    def storage_chi(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "storage_chi")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "storage_chi"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "storage_chi"] = value

    @property
    def pulse_params(self) -> Pulse_params:
        """"""
        return Pulse_params(
            self._quam, self._path + "pulse_params/", self._index,
            self._schema["properties"]["pulse_params"]
        )
    @pulse_params.setter
    def pulse_params(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "pulse_params", value)
    
    @property
    def wiring(self) -> Wiring:
        """"""
        return Wiring(
            self._quam, self._path + "wiring/", self._index,
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
  "T1": {
    "type": "number"
  },
  "T2": {
    "type": "number"
  },
  "anharmonicity": {
    "type": "number"
  },
  "f_01": {
    "type": "number"
  },
  "lo": {
    "type": "number"
  },
  "name": {
    "type": "string"
  },
  "pulse_params": {
    "type": "object",
    "title": "pulse_params",
    "properties": {
      "ac_stark_shift": {
        "type": "number"
      },
      "amplitude": {
        "type": "object",
        "title": "amplitude",
        "properties": {
          "minus_x90": {
            "type": "number"
          },
          "minus_y90": {
            "type": "number"
          },
          "x180": {
            "type": "number"
          },
          "x90": {
            "type": "number"
          },
          "y180": {
            "type": "number"
          },
          "y90": {
            "type": "number"
          }
        },
        "required": [
          "minus_x90",
          "minus_y90",
          "x180",
          "x90",
          "y180",
          "y90"
        ]
      },
      "drag_coef": {
        "type": "number"
      },
      "length": {
        "type": "number"
      },
      "saturation_amp": {
        "type": "number"
      },
      "saturation_len": {
        "type": "number"
      },
      "short_pi_amp": {
        "type": "number"
      },
      "short_pi_len": {
        "type": "number"
      }
    },
    "required": [
      "ac_stark_shift",
      "amplitude",
      "drag_coef",
      "length",
      "saturation_amp",
      "saturation_len",
      "short_pi_amp",
      "short_pi_len"
    ]
  },
  "storage_chi": {
    "type": "number"
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "I": {
        "type": "object",
        "title": "I",
        "properties": {
          "ao": {
            "type": "integer"
          },
          "con": {
            "type": "string"
          }
        },
        "required": [
          "ao",
          "con"
        ]
      },
      "Q": {
        "type": "object",
        "title": "Q",
        "properties": {
          "ao": {
            "type": "integer"
          },
          "con": {
            "type": "string"
          }
        },
        "required": [
          "ao",
          "con"
        ]
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

class Readout_params(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
    def readout_amplitude(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "readout_amplitude"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_amplitude.setter
    def readout_amplitude(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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
    def readout_length(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "readout_length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @readout_length.setter
    def readout_length(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "readout_length")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "readout_length"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "readout_length"] = value

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
        if isinstance(other, Readout_params):
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


class I2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ao(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "ao"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ao.setter
    def ao(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ao")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ao"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ao"] = value

    @property
    def con(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "con"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con.setter
    def con(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con"] = value

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
        if isinstance(other, I2):
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


class Q2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ao(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "ao"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ao.setter
    def ao(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ao")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ao"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ao"] = value

    @property
    def con(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "con"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con.setter
    def con(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con"] = value

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
        if isinstance(other, Q2):
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


class Wiring_input(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def I(self) -> I2:
        """"""
        return I2(
            self._quam, self._path + "I/", self._index,
            self._schema["properties"]["I"]
        )
    @I.setter
    def I(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "I", value)
    
    @property
    def Q(self) -> Q2:
        """"""
        return Q2(
            self._quam, self._path + "Q/", self._index,
            self._schema["properties"]["Q"]
        )
    @Q.setter
    def Q(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "Q", value)
    
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
        if isinstance(other, Wiring_input):
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


class I3(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ai(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "ai"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ai.setter
    def ai(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ai")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ai"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ai"] = value

    @property
    def con(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "con"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con.setter
    def con(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con"] = value

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
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, I3):
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


class Q3(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ai(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "ai"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ai.setter
    def ai(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ai")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ai"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ai"] = value

    @property
    def con(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "con"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con.setter
    def con(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con"] = value

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
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Q3):
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


class Wiring_output(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def gain(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "gain"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @gain.setter
    def gain(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
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
    def I(self) -> I3:
        """"""
        return I3(
            self._quam, self._path + "I/", self._index,
            self._schema["properties"]["I"]
        )
    @I.setter
    def I(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "I", value)
    
    @property
    def Q(self) -> Q3:
        """"""
        return Q3(
            self._quam, self._path + "Q/", self._index,
            self._schema["properties"]["Q"]
        )
    @Q.setter
    def Q(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "Q", value)
    
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
        if isinstance(other, Wiring_output):
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
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def frequency(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "frequency"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @frequency.setter
    def frequency(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "frequency")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "frequency"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "frequency"] = value

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
    def time_constant(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "time_constant"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @time_constant.setter
    def time_constant(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "time_constant")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "time_constant"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "time_constant"] = value

    @property
    def tof(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "tof"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @tof.setter
    def tof(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "tof")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "tof"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "tof"] = value

    @property
    def readout_params(self) -> Readout_params:
        """"""
        return Readout_params(
            self._quam, self._path + "readout_params/", self._index,
            self._schema["properties"]["readout_params"]
        )
    @readout_params.setter
    def readout_params(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "readout_params", value)
    
    @property
    def wiring_input(self) -> Wiring_input:
        """"""
        return Wiring_input(
            self._quam, self._path + "wiring_input/", self._index,
            self._schema["properties"]["wiring_input"]
        )
    @wiring_input.setter
    def wiring_input(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "wiring_input", value)
    
    @property
    def wiring_output(self) -> Wiring_output:
        """"""
        return Wiring_output(
            self._quam, self._path + "wiring_output/", self._index,
            self._schema["properties"]["wiring_output"]
        )
    @wiring_output.setter
    def wiring_output(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "wiring_output", value)
    
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
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Resonator:
        return Resonator(
            self._quam,
            self._path + "[]/",
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
  "frequency": {
    "type": "number"
  },
  "lo": {
    "type": "number"
  },
  "name": {
    "type": "string"
  },
  "time_constant": {
    "type": "number"
  },
  "readout_params": {
    "type": "object",
    "title": "readout_params",
    "properties": {
      "ge_threshold": {
        "type": "number"
      },
      "readout_amplitude": {
        "type": "number"
      },
      "readout_length": {
        "type": "number"
      },
      "rotation_angle": {
        "type": "number"
      }
    },
    "required": [
      "ge_threshold",
      "readout_amplitude",
      "readout_length",
      "rotation_angle"
    ]
  },
  "tof": {
    "type": "integer"
  },
  "wiring_input": {
    "type": "object",
    "title": "wiring_input",
    "properties": {
      "I": {
        "type": "object",
        "title": "I",
        "properties": {
          "ao": {
            "type": "integer"
          },
          "con": {
            "type": "string"
          }
        },
        "required": [
          "ao",
          "con"
        ]
      },
      "Q": {
        "type": "object",
        "title": "Q",
        "properties": {
          "ao": {
            "type": "integer"
          },
          "con": {
            "type": "string"
          }
        },
        "required": [
          "ao",
          "con"
        ]
      }
    },
    "required": [
      "I",
      "Q"
    ]
  },
  "wiring_output": {
    "type": "object",
    "title": "wiring_output",
    "properties": {
      "I": {
        "type": "object",
        "title": "I",
        "properties": {
          "ai": {
            "type": "integer"
          },
          "con": {
            "type": "string"
          },
          "offset": {
            "type": "number"
          }
        },
        "required": [
          "ai",
          "con",
          "offset"
        ]
      },
      "Q": {
        "type": "object",
        "title": "Q",
        "properties": {
          "ai": {
            "type": "integer"
          },
          "con": {
            "type": "string"
          },
          "offset": {
            "type": "number"
          }
        },
        "required": [
          "ai",
          "con",
          "offset"
        ]
      },
      "gain": {
        "type": "integer"
      }
    },
    "required": [
      "I",
      "Q",
      "gain"
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

class Displacement_params(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
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

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Displacement_params):
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


class Saturation_params(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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
        """"""
        
        value = self._quam._json[self._path + "length"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @length.setter
    def length(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
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

    def __hash__(self):
        return hash(f"{self._path}{self._index}")

    def __eq__(self, other):
        if isinstance(other, Saturation_params):
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


class I4(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ao(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "ao"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ao.setter
    def ao(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ao")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ao"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ao"] = value

    @property
    def con(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "con"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con.setter
    def con(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con"] = value

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
        if isinstance(other, I4):
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


class Q4(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def ao(self) -> int:
        """"""
        
        value = self._quam._json[self._path + "ao"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @ao.setter
    def ao(self, value: int):
        """"""
        if not isinstance(value, int):
            raise TypeError(f"Expected int but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "ao")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "ao"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "ao"] = value

    @property
    def con(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "con"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @con.setter
    def con(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "con")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "con"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "con"] = value

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
        if isinstance(other, Q4):
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


class Wiring2(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    @property
    def I(self) -> I4:
        """"""
        return I4(
            self._quam, self._path + "I/", self._index,
            self._schema["properties"]["I"]
        )
    @I.setter
    def I(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "I", value)
    
    @property
    def Q(self) -> Q4:
        """"""
        return Q4(
            self._quam, self._path + "Q/", self._index,
            self._schema["properties"]["Q"]
        )
    @Q.setter
    def Q(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "Q", value)
    
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


class Storage(object):

    def __init__(self, quam, path, index, schema):
        """"""
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

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
    def decay_rate(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "decay_rate"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @decay_rate.setter
    def decay_rate(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "decay_rate")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "decay_rate"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "decay_rate"] = value

    @property
    def frequency(self) -> float:
        """"""
        
        value = self._quam._json[self._path + "frequency"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @frequency.setter
    def frequency(self, value: float):
        """"""
        if not isinstance(value, float):
            raise TypeError(f"Expected float but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "frequency")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "frequency"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "frequency"] = value

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
    def displacement_params(self) -> Displacement_params:
        """"""
        return Displacement_params(
            self._quam, self._path + "displacement_params/", self._index,
            self._schema["properties"]["displacement_params"]
        )
    @displacement_params.setter
    def displacement_params(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "displacement_params", value)
    
    @property
    def saturation_params(self) -> Saturation_params:
        """"""
        return Saturation_params(
            self._quam, self._path + "saturation_params/", self._index,
            self._schema["properties"]["saturation_params"]
        )
    @saturation_params.setter
    def saturation_params(self, value: dict):
        import quam_sdk.crud
        quam_sdk.crud.replace_layer_in_quam_data(self, "saturation_params", value)
    
    @property
    def wiring(self) -> Wiring2:
        """"""
        return Wiring2(
            self._quam, self._path + "wiring/", self._index,
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
        if isinstance(other, Storage):
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


class StorageList(object):

    def __init__(self, quam, path, index, schema):
        self._quam = quam
        self._path = path
        self._index = index
        self._schema = schema
        self._freeze_attributes = True

    def __getitem__(self, key) -> Storage:
        return Storage(
            self._quam,
            self._path + "[]/",
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
        """Adds a new storage by adding a JSON dictionary with following schema
{
  "T1": {
    "type": "number"
  },
  "decay_rate": {
    "type": "number"
  },
  "displacement_params": {
    "type": "object",
    "title": "displacement_params",
    "properties": {
      "amplitude": {
        "type": "number"
      },
      "length": {
        "type": "integer"
      }
    },
    "required": [
      "amplitude",
      "length"
    ]
  },
  "frequency": {
    "type": "number"
  },
  "lo": {
    "type": "number"
  },
  "name": {
    "type": "string"
  },
  "saturation_params": {
    "type": "object",
    "title": "saturation_params",
    "properties": {
      "amplitude": {
        "type": "number"
      },
      "length": {
        "type": "number"
      }
    },
    "required": [
      "amplitude",
      "length"
    ]
  },
  "wiring": {
    "type": "object",
    "title": "wiring",
    "properties": {
      "I": {
        "type": "object",
        "title": "I",
        "properties": {
          "ao": {
            "type": "integer"
          },
          "con": {
            "type": "string"
          }
        },
        "required": [
          "ao",
          "con"
        ]
      },
      "Q": {
        "type": "object",
        "title": "Q",
        "properties": {
          "ao": {
            "type": "integer"
          },
          "con": {
            "type": "string"
          }
        },
        "required": [
          "ao",
          "con"
        ]
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
        self._quam = self
        self._path = ""
        self._index = []
        self._record_updates = False
        self._updates = {"keys":[], "indexes":[], "values":[], "items":[]}
        self._schema_flat = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'name': 'QuAM storage format', 'description': 'optimized data structure for communication and storage', 'type': 'object', 'properties': {'qubits[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'resonators[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'storage[]_len': {'anyof': [{'type': 'array'}, {'type': 'integer'}]}, 'opx_ip': {'type': 'string'}, 'qubits[]/T1': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/T2': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/anharmonicity': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/f_01': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/lo': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/name': {'type': 'array', 'items': {'type': 'string'}}, 'qubits[]/pulse_params/ac_stark_shift': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/amplitude/minus_x90': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/amplitude/minus_y90': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/amplitude/x180': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/amplitude/x90': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/amplitude/y180': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/amplitude/y90': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/drag_coef': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/length': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/saturation_amp': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/saturation_len': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/short_pi_amp': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/pulse_params/short_pi_len': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/storage_chi': {'type': 'array', 'items': {'type': 'number'}}, 'qubits[]/wiring/I/ao': {'type': 'array', 'items': {'type': 'integer'}}, 'qubits[]/wiring/I/con': {'type': 'array', 'items': {'type': 'string'}}, 'qubits[]/wiring/Q/ao': {'type': 'array', 'items': {'type': 'integer'}}, 'qubits[]/wiring/Q/con': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[]/frequency': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/lo': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/name': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[]/time_constant': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/readout_params/ge_threshold': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/readout_params/readout_amplitude': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/readout_params/readout_length': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/readout_params/rotation_angle': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/tof': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[]/wiring_input/I/ao': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[]/wiring_input/I/con': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[]/wiring_input/Q/ao': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[]/wiring_input/Q/con': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[]/wiring_output/I/ai': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[]/wiring_output/I/con': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[]/wiring_output/I/offset': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/wiring_output/Q/ai': {'type': 'array', 'items': {'type': 'integer'}}, 'resonators[]/wiring_output/Q/con': {'type': 'array', 'items': {'type': 'string'}}, 'resonators[]/wiring_output/Q/offset': {'type': 'array', 'items': {'type': 'number'}}, 'resonators[]/wiring_output/gain': {'type': 'array', 'items': {'type': 'integer'}}, 'storage[]/T1': {'type': 'array', 'items': {'type': 'number'}}, 'storage[]/decay_rate': {'type': 'array', 'items': {'type': 'number'}}, 'storage[]/displacement_params/amplitude': {'type': 'array', 'items': {'type': 'number'}}, 'storage[]/displacement_params/length': {'type': 'array', 'items': {'type': 'integer'}}, 'storage[]/frequency': {'type': 'array', 'items': {'type': 'number'}}, 'storage[]/lo': {'type': 'array', 'items': {'type': 'number'}}, 'storage[]/name': {'type': 'array', 'items': {'type': 'string'}}, 'storage[]/saturation_params/amplitude': {'type': 'array', 'items': {'type': 'number'}}, 'storage[]/saturation_params/length': {'type': 'array', 'items': {'type': 'number'}}, 'storage[]/wiring/I/ao': {'type': 'array', 'items': {'type': 'integer'}}, 'storage[]/wiring/I/con': {'type': 'array', 'items': {'type': 'string'}}, 'storage[]/wiring/Q/ao': {'type': 'array', 'items': {'type': 'integer'}}, 'storage[]/wiring/Q/con': {'type': 'array', 'items': {'type': 'string'}}}, 'additionalProperties': False}
        self._schema = {'$schema': 'https://json-schema.org/draft/2020-12/schema', 'type': 'object', 'title': 'QuAM', 'properties': {'opx_ip': {'type': 'string'}, 'qubits': {'type': 'array', 'items': {'type': 'object', 'title': 'qubit', 'properties': {'T1': {'type': 'number'}, 'T2': {'type': 'number'}, 'anharmonicity': {'type': 'number'}, 'f_01': {'type': 'number'}, 'lo': {'type': 'number'}, 'name': {'type': 'string'}, 'pulse_params': {'type': 'object', 'title': 'pulse_params', 'properties': {'ac_stark_shift': {'type': 'number'}, 'amplitude': {'type': 'object', 'title': 'amplitude', 'properties': {'minus_x90': {'type': 'number'}, 'minus_y90': {'type': 'number'}, 'x180': {'type': 'number'}, 'x90': {'type': 'number'}, 'y180': {'type': 'number'}, 'y90': {'type': 'number'}}, 'required': ['minus_x90', 'minus_y90', 'x180', 'x90', 'y180', 'y90']}, 'drag_coef': {'type': 'number'}, 'length': {'type': 'number'}, 'saturation_amp': {'type': 'number'}, 'saturation_len': {'type': 'number'}, 'short_pi_amp': {'type': 'number'}, 'short_pi_len': {'type': 'number'}}, 'required': ['ac_stark_shift', 'amplitude', 'drag_coef', 'length', 'saturation_amp', 'saturation_len', 'short_pi_amp', 'short_pi_len']}, 'storage_chi': {'type': 'number'}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'I': {'type': 'object', 'title': 'I', 'properties': {'ao': {'type': 'integer'}, 'con': {'type': 'string'}}, 'required': ['ao', 'con']}, 'Q': {'type': 'object', 'title': 'Q', 'properties': {'ao': {'type': 'integer'}, 'con': {'type': 'string'}}, 'required': ['ao', 'con']}}, 'required': ['I', 'Q']}}, 'required': ['T1', 'T2', 'anharmonicity', 'f_01', 'lo', 'name', 'pulse_params', 'storage_chi', 'wiring']}}, 'resonators': {'type': 'array', 'items': {'type': 'object', 'title': 'resonator', 'properties': {'frequency': {'type': 'number'}, 'lo': {'type': 'number'}, 'name': {'type': 'string'}, 'time_constant': {'type': 'number'}, 'readout_params': {'type': 'object', 'title': 'readout_params', 'properties': {'ge_threshold': {'type': 'number'}, 'readout_amplitude': {'type': 'number'}, 'readout_length': {'type': 'number'}, 'rotation_angle': {'type': 'number'}}, 'required': ['ge_threshold', 'readout_amplitude', 'readout_length', 'rotation_angle']}, 'tof': {'type': 'integer'}, 'wiring_input': {'type': 'object', 'title': 'wiring_input', 'properties': {'I': {'type': 'object', 'title': 'I', 'properties': {'ao': {'type': 'integer'}, 'con': {'type': 'string'}}, 'required': ['ao', 'con']}, 'Q': {'type': 'object', 'title': 'Q', 'properties': {'ao': {'type': 'integer'}, 'con': {'type': 'string'}}, 'required': ['ao', 'con']}}, 'required': ['I', 'Q']}, 'wiring_output': {'type': 'object', 'title': 'wiring_output', 'properties': {'I': {'type': 'object', 'title': 'I', 'properties': {'ai': {'type': 'integer'}, 'con': {'type': 'string'}, 'offset': {'type': 'number'}}, 'required': ['ai', 'con', 'offset']}, 'Q': {'type': 'object', 'title': 'Q', 'properties': {'ai': {'type': 'integer'}, 'con': {'type': 'string'}, 'offset': {'type': 'number'}}, 'required': ['ai', 'con', 'offset']}, 'gain': {'type': 'integer'}}, 'required': ['I', 'Q', 'gain']}}, 'required': ['frequency', 'lo', 'name', 'time_constant', 'readout_params', 'tof', 'wiring_input', 'wiring_output']}}, 'storage': {'type': 'array', 'items': {'type': 'object', 'title': 'storage', 'properties': {'T1': {'type': 'number'}, 'decay_rate': {'type': 'number'}, 'displacement_params': {'type': 'object', 'title': 'displacement_params', 'properties': {'amplitude': {'type': 'number'}, 'length': {'type': 'integer'}}, 'required': ['amplitude', 'length']}, 'frequency': {'type': 'number'}, 'lo': {'type': 'number'}, 'name': {'type': 'string'}, 'saturation_params': {'type': 'object', 'title': 'saturation_params', 'properties': {'amplitude': {'type': 'number'}, 'length': {'type': 'number'}}, 'required': ['amplitude', 'length']}, 'wiring': {'type': 'object', 'title': 'wiring', 'properties': {'I': {'type': 'object', 'title': 'I', 'properties': {'ao': {'type': 'integer'}, 'con': {'type': 'string'}}, 'required': ['ao', 'con']}, 'Q': {'type': 'object', 'title': 'Q', 'properties': {'ao': {'type': 'integer'}, 'con': {'type': 'string'}}, 'required': ['ao', 'con']}}, 'required': ['I', 'Q']}}, 'required': ['T1', 'decay_rate', 'displacement_params', 'frequency', 'lo', 'name', 'saturation_params', 'wiring']}}}, 'required': ['opx_ip', 'qubits', 'resonators', 'storage']}
        if data is not None:
            if type(data) is str:
                with open(data, "r") as file:
                    data = json.load(file)
            import quam_sdk.crud
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
                item[1] +"[]/", item[2], new_item=True
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
    def opx_ip(self) -> str:
        """"""
        
        value = self._quam._json[self._path + "opx_ip"]
        for i in range(len(self._index)):
            value = value[self._index[i]]
        return value


    @opx_ip.setter
    def opx_ip(self, value: str):
        """"""
        if not isinstance(value, str):
            raise TypeError(f"Expected str but received {type(value)}")
        if self._quam._record_updates:
            self._quam._updates["keys"].append(self._path + "opx_ip")
            self._quam._updates["indexes"].append(self._index)
            self._quam._updates["values"].append(value)
        if (len(self._index) > 0):
            value_ref = self._quam._json[self._path + "opx_ip"]
            for i in range(len(self._index)-1):
                value_ref = value_ref[self._index[i]]
            value_ref[self._index[-1]] = value
        else:
            self._quam._json[self._path + "opx_ip"] = value

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
    def storage(self) -> StorageList:
        """"""
        return StorageList(
            self._quam, self._path + "storage", self._index,
            self._schema["properties"]["storage"]
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


