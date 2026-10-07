import numpy as np

ZETA_FIELDS = ("zeta_ground_hz", "zeta_excited_hz")


def waveform_peak(pulse_type, kwargs: dict) -> float:
    """Largest |I| or |Q| sample of the waveform the pulse would put in the config."""
    waveform = np.asarray(pulse_type(**kwargs).waveform_function())
    return float(max(np.abs(waveform.real).max(), np.abs(waveform.imag).max()))


def scale_pulse_kwargs(kwargs: dict, factor: float, compensate_kerr: bool = False) -> dict:
    """Copy of the pulse kwargs with the amplitude multiplied by `factor`.

    The self-Kerr correction depends on zeta * amplitude**2, so with `compensate_kerr` the zetas are divided
    by factor**2: the waveform is then exactly `factor` times the unscaled one, i.e. the scan still probes the
    zeta it is labelled with instead of zeta * factor**2.
    """
    scaled = {**kwargs, "amplitude": kwargs["amplitude"] * factor}
    if compensate_kerr:
        for field in ZETA_FIELDS:
            scaled[field] = kwargs.get(field, 0.0) / factor**2
    return scaled


def amplitude_scale_to_fit(
    pulse_type, kwargs_list: list[dict], limit: float, compensate_kerr: bool = False, max_iterations: int = 10
) -> float:
    """Common amplitude factor (<= 1) so every pulse in kwargs_list peaks at or below `limit`.

    Iterated because the waveform is not exactly linear in amplitude once the self-Kerr correction
    (zeta != 0) is active and not compensated.
    """
    factor = 1.0
    for _ in range(max_iterations):
        peak = max(waveform_peak(pulse_type, scale_pulse_kwargs(kw, factor, compensate_kerr)) for kw in kwargs_list)
        if peak <= limit:
            return factor
        factor *= limit / peak
    raise ValueError(f"Could not scale the waveforms below a peak of {limit} (last factor {factor:.3g}).")
