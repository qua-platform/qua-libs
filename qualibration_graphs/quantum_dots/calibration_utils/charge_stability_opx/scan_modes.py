from abc import ABC, abstractmethod
from typing import Generator, Sequence, Tuple, Dict

import numpy as np
from qm.qua import declare, fixed, for_, for_each_
from qualang_tools.loops import from_array


__all__ = ["ScanMode", "RasterScan", "SwitchRasterScan"]

class ScanMode(ABC):
    """Abstract base class for scan modes (raster, spiral, etc.)."""
    _registry: Dict[str, type["ScanMode"]] = {}
    def __init_subclass__(cls, **kwargs): 
        super().__init_subclass__(**kwargs)
        if cls.name is not None: 
            ScanMode._registry[cls.name] = cls
        
    @classmethod
    def from_name(cls, name: str, **kwargs) -> "ScanMode":
        if name not in cls._registry:
            raise ValueError(f"Unknown scan mode: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs)

    @abstractmethod
    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the x and y index arrays defining the scan pattern."""
        pass

    @abstractmethod
    def scan(
        self, x_vals: Sequence[float], y_vals: Sequence[float]
    ) -> Generator[Tuple, None, None]:
        """Yield (x, y) QUA variables while generating the scan loop structure."""
        pass


class RasterScan(ScanMode):
    """Standard row-by-row raster scan."""
    name = "raster"

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        x_idxs = np.tile(np.arange(x_points), y_points)
        y_idxs = np.repeat(np.arange(y_points), x_points)
        return x_idxs, y_idxs

    def scan(
        self, x_vals: Sequence[float], y_vals: Sequence[float]
    ) -> Generator[Tuple, None, None]:
        x = declare(fixed)
        y = declare(fixed)
        with for_(*from_array(y, y_vals)):
            with for_(*from_array(x, x_vals)):
                yield x, y


class SwitchRasterScan(ScanMode):
    """Raster scan starting from middle, alternating outward (useful for bias tee considerations)."""
    name = "switch_raster"

    def __init__(self, start_from_middle: bool = True):
        self.start_from_middle = start_from_middle

    @staticmethod
    def interleave_arr(arr: np.ndarray, start_from_middle: bool = True) -> np.ndarray:
        mid_idx = len(arr) // 2
        if len(arr) % 2:
            interleaved = [arr[mid_idx]]
            arr1 = arr[mid_idx + 1:]
            arr2 = arr[mid_idx - 1::-1]
            interleaved += [elem for pair in zip(arr1, arr2) for elem in pair]
        else:
            arr1 = arr[mid_idx:]
            arr2 = arr[mid_idx - 1::-1]
            interleaved = [elem for pair in zip(arr1, arr2) for elem in pair]
        return np.array(interleaved) if start_from_middle else np.array(interleaved[::-1])

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        y_idxs = self.interleave_arr(np.arange(y_points), self.start_from_middle)
        x_idxs = np.tile(np.arange(x_points), y_points)
        y_idxs = np.repeat(y_idxs, x_points)
        return x_idxs, y_idxs

    def scan(
        self, x_vals: Sequence[float], y_vals: Sequence[float]
    ) -> Generator[Tuple, None, None]:
        x = declare(fixed)
        y = declare(fixed)
        y_vals_interleaved = self.interleave_arr(np.asarray(y_vals), self.start_from_middle)
        with for_each_(y, y_vals_interleaved.tolist()):
            with for_(*from_array(x, x_vals)):
                yield x, y