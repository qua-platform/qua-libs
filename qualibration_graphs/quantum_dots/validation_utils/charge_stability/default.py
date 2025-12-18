from time import perf_counter
from typing import Any, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
try:
    from qarray import ChargeSensedDotArray, WhiteNoise, TelegraphNoise, LatchingModel
    from qarray.functions import charge_state_changes
except ImportError:
     print(f'Failed to import ChargeSensedDotArray due to missing dependencies.')


class InitDotModel:
    """
    Static factory that builds the default six-dot charge-sensed quantum dot array model.

    Calling an instance of this class returns a fully configured ``ChargeSensedDotArray`` that
    mirrors the behaviour of the original ``init_dot_model`` function. Individual components can be
    overridden by providing keyword arguments to the call, for example::

        model = init_dot_model(Cgs=[[...]])

    which reuses every default except the gate-sensor capacitance matrix.
    """

    n_dots: int = 6

    @staticmethod
    def dot_dot_capacitance() -> List[List[float]]:
        """Return the dot-dot mutual capacitance matrix for the default device."""
        return [
            [0.12, 0.08, 0.00, 0.00, 0.00, 0.00],
            [0.08, 0.13, 0.08, 0.00, 0.00, 0.00],
            [0.00, 0.08, 0.12, 0.08, 0.00, 0.00],
            [0.00, 0.00, 0.08, 0.12, 0.08, 0.00],
            [0.00, 0.00, 0.00, 0.08, 0.12, 0.08],
            [0.00, 0.00, 0.00, 0.00, 0.08, 0.11],
        ]

    @staticmethod
    def dot_gate_capacitance() -> List[List[float]]:
        """Return the dot-gate mutual capacitance matrix for the default device."""
        return [
            [0.13, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.09, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.13, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.13, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.00],
        ]

    @staticmethod
    def dot_sensor_capacitance() -> List[List[float]]:
        """Return the default dot-sensor capacitance matrix."""
        return [[0.002, 0.002, 0.002, 0.002, 0.002, 0.002]]

    @staticmethod
    def gate_sensor_capacitance() -> List[List[float]]:
        """Return the default gate-sensor capacitance matrix."""
        return [[0.001, 0.002, 0.000, 0.000, 0.000, 0.000, 0.100]]

    @staticmethod
    def white_noise() -> WhiteNoise:
        """Return the default white noise model."""
        return WhiteNoise(amplitude=2.e-2)

    @staticmethod
    def telegraph_noise() -> TelegraphNoise:
        """Return the default telegraph noise model."""
        return TelegraphNoise(amplitude=5e-4, p01=5e-3, p10=5e-3)

    @classmethod
    def noise_model(cls) -> Any:
        """Combine the default noise models into a single noise configuration."""
        return cls.white_noise() + cls.telegraph_noise()

    @classmethod
    def latching_model(cls) -> LatchingModel:
        """Return the default latching model for the six-dot configuration."""
        return LatchingModel(
            n_dots=cls.n_dots,
            p_leads=0.95,
            p_inter=0.005,
        )

    def __call__(
        self,
        *,
        Cdd: Optional[Sequence[Sequence[float]]] = None,
        Cgd: Optional[Sequence[Sequence[float]]] = None,
        Cds: Optional[Sequence[Sequence[float]]] = None,
        Cgs: Optional[Sequence[Sequence[float]]] = None,
        noise_model: Optional[Any] = None,
        latching_model: Optional['LatchingModel'] = None,
        coulomb_peak_width: float = 0.9,
        T: float = 50.0,
        algorithm: str = 'default',
        implementation: str = 'jax',
    ) -> 'ChargeSensedDotArray':
        """
        Build and return the default charge-sensed dot array model.

        Keyword arguments allow overriding individual components while falling back to the default
        static factories when omitted.
        """
        return ChargeSensedDotArray(
            Cdd=Cdd if Cdd is not None else self.dot_dot_capacitance(),
            Cgd=Cgd if Cgd is not None else self.dot_gate_capacitance(),
            Cds=Cds if Cds is not None else self.dot_sensor_capacitance(),
            Cgs=Cgs if Cgs is not None else self.gate_sensor_capacitance(),
            coulomb_peak_width=coulomb_peak_width,
            T=T,
            algorithm=algorithm,
            implementation=implementation,
            noise_model=noise_model if noise_model is not None else self.noise_model(),
            latching_model=latching_model if latching_model is not None else self.latching_model(),
        )



init_dot_model: InitDotModel = InitDotModel()


if __name__ == '__main__':
    updated_cgs = [[0.001, 0.002, 0.000, 0.000, 0.000, 0.000, 0.100]]
    model = init_dot_model(Cgs=updated_cgs)
    n_charges = [1, 3, 0, 0, 0, 0, 5]
    optimal_voltage_configuration = model.optimal_Vg(
        n_charges=n_charges,
    )

    # Create voltage composer
    voltage_composer = model.gate_voltage_composer

    # Sweep sensor dot
    vs_min, vs_max = -10, 10
    ns = 200
    sensor_sweep = np.linspace(vs_min, vs_max, ns)
    z, n = model.do1d_open(
        7, vs_min, vs_max, ns
    )

    plt.plot(sensor_sweep, z)
    plt.xlabel('$Vx$')
    plt.ylabel('Signal (au)')
    plt.show()
    print('Sensor optimum : ', sensor_sweep[np.argmax(z)])
    optimal_voltage_configuration[-1] = sensor_sweep[np.argmax(z)]
    # Define min and max values for the 2D voltage sweep
    vx_min, vx_max = -50, 50
    vy_min, vy_max = -50, 50

    # Create voltage array for 2D sweep (gates 1 and 2)
    vg = voltage_composer.do2d(1, vx_min, vx_max, 200, 2, vy_min, vy_max, 200)
    vg += optimal_voltage_configuration

    # Compute charge sensor response
    t0 = perf_counter()
    z, n = model.charge_sensor_open(vg)
    print(f'Compute time: {perf_counter() - t0:.2f} s')

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.set_size_inches(10, 5)

    # Plot charge stability diagram
    axes[0].imshow(z, extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto', cmap='hot')
    axes[0].set_xlabel('$Vx$')
    axes[0].set_ylabel('$Vy$')
    axes[0].set_title('$z$')

    # Plot charge state changes
    axes[1].imshow(charge_state_changes(n), extent=[vx_min, vx_max, vy_min, vy_max], origin='lower', aspect='auto',
                   cmap='hot')
    plt.show()
