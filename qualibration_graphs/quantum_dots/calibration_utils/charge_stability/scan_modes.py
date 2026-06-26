from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator, Tuple, Dict

import numpy as np
from qm.qua import Cast, assign, declare, else_, fixed, for_, for_each_, if_
from qualang_tools.loops import from_array

if TYPE_CHECKING:
    import xarray as xr


__all__ = ["ScanMode", "RasterScan", "SwitchRasterScan", "SpiralScan"]


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
            raise ValueError(
                f"Unknown scan mode: {name}. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name](**kwargs)

    @abstractmethod
    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the x and y index arrays defining the scan pattern."""
        pass

    def get_y_axis_order(self, y_volts) -> np.ndarray:
        """Return y-axis coordinate array in the order rows are written to the stream."""
        return np.asarray(y_volts)

    def get_x_axis_order(self, x_volts) -> np.ndarray:
        """Return x-axis coordinate array in the order columns are written to the stream."""
        return np.asarray(x_volts)

    def qua_stream_processing(self, stream, x_points: int, y_points: int):
        """Apply stream processing appropriate for this scan pattern.
        Returns the processed stream ready to be saved."""
        return stream.buffer(x_points).buffer(y_points).average()

    def reorder_dataset(self, ds: "xr.Dataset") -> "xr.Dataset":
        """Reorder dataset data variables into the correct spatial grid.
        Default is a no-op for scan modes whose stream order already matches coords."""
        return ds

    @abstractmethod
    def qua_scan(self, sequence, x_obj, y_obj, x_volts, y_volts, params):
        """Generator yielding (x, y, save_now) QUA variables.

        save_now must evaluate to 1 when buffered points should be saved
        to the streams for this scan mode."""
        pass

    @abstractmethod
    def get_save_buffer_size(self, x_volts, y_volts) -> int:
        """Return the max buffered points between save events."""
        pass

    def get_scan_voltages(self, x_volts, y_volts):
        x_idxs, y_idxs = self.get_idxs(len(x_volts), len(y_volts))
        return x_volts[x_idxs], y_volts[y_idxs]

    def compensate(self, seq, params): 
        if params.per_line_compensation:
            seq.apply_compensation_pulse(go_to_zero=True, return_to_zero=True)
        else:
            seq.ramp_to_zero(ramp_duration=16)

class RasterScan(ScanMode):
    """Standard row-by-row raster scan."""

    name = "raster"

    def __init__(self, use_precomputed_scan: bool = True): 
        pass

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        x_idxs = np.tile(np.arange(x_points), y_points)
        y_idxs = np.repeat(np.arange(y_points), x_points)
        return x_idxs, y_idxs

    def qua_scan(self, seq, x_obj, y_obj, x_volts, y_volts, params):
        x = declare(fixed)
        y = declare(fixed)
        save_now = declare(int)
        x_idx = declare(int)
        y_outer = self.get_y_axis_order(y_volts)
        with for_each_(y, y_outer.tolist()):
            assign(x_idx, 0)
            if params.per_line_wait > 0:
                assign(x, float(x_volts[0]))
                seq.ramp_to_voltages(
                    {x_obj.name: x, y_obj.name: y},
                    duration=params.per_line_wait,
                    ramp_duration=params.ramp_duration,
                )
            with for_(*from_array(x, x_volts)):
                assign(save_now, 0)
                with if_(x_idx == len(x_volts) - 1):
                    assign(save_now, 1)
                yield x, y, save_now
                assign(x_idx, x_idx + 1)

    def get_save_buffer_size(self, x_volts, y_volts) -> int:
        return len(x_volts)


class SwitchRasterScan(RasterScan):
    """Raster scan starting from middle, alternating outward (useful for bias tee considerations)."""

    name = "switch_raster"

    def __init__(self, start_from_middle: bool = True, use_precomputed_scan: bool = True):
        self.start_from_middle = start_from_middle

    @staticmethod
    def interleave_arr(arr: np.ndarray, start_from_middle: bool = True) -> np.ndarray:
        mid_idx = len(arr) // 2
        if len(arr) % 2:
            interleaved = [arr[mid_idx]]
            arr1 = arr[mid_idx + 1 :]
            arr2 = arr[mid_idx - 1 :: -1]
            interleaved += [elem for pair in zip(arr1, arr2) for elem in pair]
        else:
            arr1 = arr[mid_idx:]
            arr2 = arr[mid_idx - 1 :: -1]
            interleaved = [elem for pair in zip(arr1, arr2) for elem in pair]
        return (
            np.array(interleaved) if start_from_middle else np.array(interleaved[::-1])
        )

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        y_idxs = self.interleave_arr(np.arange(y_points), self.start_from_middle)
        x_idxs = np.tile(np.arange(x_points), y_points)
        y_idxs = np.repeat(y_idxs, x_points)
        return x_idxs, y_idxs

    def get_y_axis_order(self, y_volts) -> np.ndarray:
        return self.interleave_arr(np.asarray(y_volts), self.start_from_middle)


class SpiralScan(ScanMode):
    """Center-outward spiral scan."""

    name = "spiral"

    def __init__(self, use_precomputed_scan: bool = False):
        self.use_precomputed_scan = use_precomputed_scan

    @staticmethod
    def _center_out_spiral(x_points: int, y_points: int) -> list[tuple[int, int]]:
        """Generate center-out spiral indices without storing voltage lookup lists."""
        n_total = x_points * y_points
        x_idx = (x_points - 1) // 2
        y_idx = (y_points - 1) // 2
        direction = 0  # 0:right, 1:up, 2:left, 3:down
        leg_length = 1
        leg_step_count = 0
        legs_done = 0

        result: list[tuple[int, int]] = []
        max_side = max(x_points, y_points) + 2
        max_steps = (2 * max_side + 1) ** 2

        for _ in range(max_steps):
            if len(result) >= n_total:
                break
            if 0 <= x_idx < x_points and 0 <= y_idx < y_points:
                result.append((x_idx, y_idx))

            if direction == 0:
                x_idx += 1
            elif direction == 1:
                y_idx += 1
            elif direction == 2:
                x_idx -= 1
            else:
                y_idx -= 1

            leg_step_count += 1
            if leg_step_count == leg_length:
                leg_step_count = 0
                direction = (direction + 1) % 4
                legs_done += 1
                if legs_done == 2:
                    legs_done = 0
                    leg_length += 1

        if len(result) != n_total:
            raise RuntimeError(
                f"Failed to generate spiral covering all points ({len(result)} != {n_total})."
            )
        return result

    def get_idxs(self, x_points: int, y_points: int) -> Tuple[np.ndarray, np.ndarray]:
        pairs = self._center_out_spiral(x_points, y_points)
        x_idxs = np.array([p[0] for p in pairs])
        y_idxs = np.array([p[1] for p in pairs])
        return x_idxs, y_idxs

    @staticmethod
    def _save_flags_from_x_idxs(x_idxs: np.ndarray, x_zero_idx: int) -> np.ndarray:
        save_flags = np.zeros(len(x_idxs), dtype=int)
        if len(x_idxs) == 0:
            return save_flags
        for k, x_idx in enumerate(x_idxs):
            if (x_idx == x_zero_idx and k > 0) or (k == len(x_idxs) - 1):
                save_flags[k] = 1
        return save_flags

    def reorder_dataset(self, ds: "xr.Dataset") -> "xr.Dataset":
        """Scatter buffer-order data back to the correct (x_idx, y_idx) grid positions.

        .buffer(x_points).buffer(y_points) maps flat scan position k to
        arr[..., k % x_points, k // x_points] under the (x_volts, y_volts)
        dataset dimension convention.

        We scatter each k to its actual spiral grid position (x_idxs[k], y_idxs[k]).
        """
        import xarray as xr
        x_points = ds.sizes["x_volts"]
        y_points = ds.sizes["y_volts"]
        x_idxs, y_idxs = self.get_idxs(x_points, y_points)

        flat_k = np.arange(x_points * y_points)
        x_src = flat_k % x_points
        y_src = flat_k // x_points

        result = ds.copy()
        for var in ds.data_vars:
            arr = ds[var].values  # [..., x_points, y_points] in buffer order
            out = np.empty_like(arr)
            out[..., x_idxs, y_idxs] = arr[..., x_src, y_src]
            result[var] = xr.DataArray(out, dims=ds[var].dims, coords=ds[var].coords)
        return result

    def qua_scan(self, seq, x_obj, y_obj, x_volts, y_volts, params):
        """Generate spiral scan points with mode-selectable implementation."""
        x = declare(fixed)
        y = declare(fixed)
        save_now = declare(int)
        x_zero_idx = int(np.argmin(np.abs(x_volts)))
        x_idxs, _ = self.get_idxs(len(x_volts), len(y_volts))
        save_flags = self._save_flags_from_x_idxs(x_idxs, x_zero_idx)

        if self.use_precomputed_scan:
            x_scan, y_scan = self.get_scan_voltages(x_volts, y_volts)
            with for_each_(
                (x, y, save_now),
                (x_scan.tolist(), y_scan.tolist(), save_flags.tolist()),
            ):
                yield x, y, save_now
            return

        x_idx = declare(int)
        y_idx = declare(int)
        direction = declare(int)
        leg_length = declare(int)
        leg_step_count = declare(int)
        legs_done = declare(int)
        emitted = declare(int)

        x_start = float(x_volts[0])
        y_start = float(y_volts[0])
        x_step = float(x_volts[1] - x_volts[0]) if len(x_volts) > 1 else 0.0
        y_step = float(y_volts[1] - y_volts[0]) if len(y_volts) > 1 else 0.0

        n_total = len(x_volts) * len(y_volts)

        assign(x_idx, (len(x_volts) - 1) // 2)
        assign(y_idx, (len(y_volts) - 1) // 2)
        assign(direction, 0)  # 0:right, 1:up, 2:left, 3:down
        assign(leg_length, 1)
        assign(leg_step_count, 0)
        assign(legs_done, 0)
        assign(emitted, 0)

        step_iter = declare(int)
        with for_(step_iter, 0, emitted < n_total, step_iter + 1):
            with if_(x_idx >= 0):
                with if_(x_idx < len(x_volts)):
                    with if_(y_idx >= 0):
                        with if_(y_idx < len(y_volts)):
                            assign(x, x_start + Cast.mul_fixed_by_int(x_step, x_idx))
                            assign(y, y_start + Cast.mul_fixed_by_int(y_step, y_idx))
                            assign(save_now, 0)
                            with if_(x_idx == x_zero_idx):
                                with if_(emitted > 0):
                                    assign(save_now, 1)
                            with if_(emitted == n_total - 1):
                                assign(save_now, 1)
                            yield x, y, save_now
                            assign(emitted, emitted + 1)

            with if_(direction == 0):
                assign(x_idx, x_idx + 1)
            with else_():
                with if_(direction == 1):
                    assign(y_idx, y_idx + 1)
                with else_():
                    with if_(direction == 2):
                        assign(x_idx, x_idx - 1)
                    with else_():
                        assign(y_idx, y_idx - 1)

            assign(leg_step_count, leg_step_count + 1)
            with if_(leg_step_count == leg_length):
                assign(leg_step_count, 0)
                assign(direction, direction + 1)
                with if_(direction == 4):
                    assign(direction, 0)
                assign(legs_done, legs_done + 1)
                with if_(legs_done == 2):
                    assign(legs_done, 0)
                    assign(leg_length, leg_length + 1)

    def get_save_buffer_size(self, x_volts, y_volts) -> int:
        x_zero_idx = int(np.argmin(np.abs(np.asarray(x_volts))))
        x_idxs, _ = self.get_idxs(len(x_volts), len(y_volts))
        save_flags = self._save_flags_from_x_idxs(x_idxs, x_zero_idx)
        max_chunk = 0
        chunk_len = 0
        for k in range(len(x_idxs)):
            chunk_len += 1
            if save_flags[k]:
                max_chunk = max(max_chunk, chunk_len)
                chunk_len = 0
        return max_chunk

    def compensate(self, seq, params): 
        pass