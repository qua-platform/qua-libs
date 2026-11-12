from typing import ClassVar, Optional, Sequence, Type

from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam

from quam_config.flux_lru_config import add_flux_lru_elements
from quam_config.lru_components import LruFluxTunableTransmon


# Define the QUAM class that will be used in all calibration nodes
# Should inherit from either FixedFrequencyQuam or FluxTunableQuam
@quam_dataclass
class Quam(FluxTunableQuam):
    # Qubits are built with f-LRU support (qubit.lru, qubit.flux_lru). With FixedFrequencyQuam, use
    # LruFixedFrequencyTransmon instead. States saved before keep the class they name.
    qubit_type: ClassVar[Type[LruFluxTunableTransmon]] = LruFluxTunableTransmon

    def generate_config(self, flux_lru_qubits: Optional[Sequence] = None):
        """Generate the QUA config; ``flux_lru_qubits`` (qubits or names) adds their flux-LRU elements.

        Those qubits' flux ports run at 2 GS/s in the returned config, with their flux pulses upsampled,
        so ``qubit.flux_lru.play()`` can be used in the program (see ``quam_config.flux_lru_config``). A state
        saved with another root class loads as that class, without this argument; nodes therefore call
        ``flux_lru_config.generate_flux_lru_config``, which works with any root.
        """
        config = super().generate_config()
        if flux_lru_qubits:
            qubits = [self.qubits[q] if isinstance(q, str) else q for q in flux_lru_qubits]
            add_flux_lru_elements(config, qubits)
        return config
