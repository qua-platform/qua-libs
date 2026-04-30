"""Parameter definitions for single-qubit gate set tomography experiments.

This module defines the parameters used for configuring GST experiments,
including circuit lengths, number of shots, and operation types.
"""

from typing import ClassVar, Literal

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters
# import pygsti
from .gst_sequences import _load_pygsti_model_pack


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for single-qubit GST experiments."""

    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    max_length: int = 256
    """Maximum number of gates that appear in a repeated germ. Default is 256."""
    log_scale: bool = True
    """If True, use log-scale lengths: 1, 2, 4, 8, 16, ... up to max_length. Default is True."""
    delta_length: int = 20
    """Step between lengths in linear scale mode. Default is 20."""
    model: Literal["smq1Q_XY", "smq1Q_XYI"] = "smq1Q_XY"
    """Model to use for the GST experiment. Default is "smq1Q_XY"."""
    # use_state_discrimination: bool = True
    # """Whether to use state discrimination for readout. Default is True."""
    # use_strict_timing: bool = False
    # """Use strict timing in the QUA program. Default is False."""
    # use_input_stream: bool = False
    # """Whether to use input streams for circuit execution. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for single-qubit GST experiments."""

    # GST_SEQUENCE_COUNT_LIMIT = 2000

    # prep_fiducials: list[str] = []
    # meas_fiducials: list[str] = []
    # germs: list[str] = []
    # gst_sequences: list[str] = []
    # target_model: pygsti.model.ExplicitModel = None
    # """Target model for GST."""
    # prep_fiducial_map: dict[str, int] = {"{}": 0}
    # """Map of preparation fiducials to indices."""
    # meas_fiducial_map: dict[str, int] = {"{}": 0}
    # """Map of measurement fiducials to indices."""
    # germ_map: dict[str, int] = {"{}": 0}
    # """Map of germs to indices."""

    def get_lengths(self) -> list[int]:
        """Generate circuit depths based on the parameter configuration.

        Length *L* means at most L gates in the germ repetition.
        Germs consisting of N gates will be repeated at most L/N times.
        Germs that are longer than L will not feature in the GST sequences.
        This is the standard GST convention.

        - If ``log_scale`` is True, lengths follow a power-of-two progression:
          1, 2, 4, 8, 16, ... up to ``max_length``.
        - If ``log_scale`` is False, lengths are linearly spaced using
          ``delta_length`` until ``max_length``.  The first value
          is always set to 1.

        Returns
        -------
        list[int]
            Sorted circuit lengths.
        """
        if self.log_scale:
            lengths: list[int] = []
            current_length = 1
            while current_length <= self.max_length:
                lengths.append(current_length)
                current_length *= 2
            return lengths

        assert (
            self.max_length / self.delta_length
        ).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
        lengths = list(range(0, self.max_length + 1, self.delta_length))
        lengths[0] = 1
        return lengths
    
    def get_gst_components(self) -> tuple[list[str], list[str], list[str]]:
        """Get the GST components from the model.
        
        Returns:
            Lists of preparation fiducials, measurement fiducials, germs, and target model.
        """
        pack = _load_pygsti_model_pack(self.model)
        # import pygsti.modelpacks as modelpacks_pkg  # noqa: PLC0415
        # pack = modelpacks_pkg.get_model_pack(self.model)

        prep_fiducials = pack.prep_fiducials()
        meas_fiducials = pack.meas_fiducials()
        germs = pack.germs()
        target_model = pack.target_model()

        return prep_fiducials, meas_fiducials, germs, target_model

    # prep_fiducials, meas_fiducials, germs, target_model = self.get_gst_components()

    # def get_gst_sequences(self) -> list[str]:
    #     """Get the GST sequences from the model.
        
    #     Returns:
    #         List of GST sequences converted to strings.
    #     """
    #     if len(self.prep_fiducials) == 0 or len(self.meas_fiducials) == 0 or len(self.germs) == 0 or self.target_model is None:
    #         self.get_gst_components()

    #     max_lengths = self.get_lengths()
    #     lsgst_lists = pygsti.circuits.create_lsgst_circuit_lists(
    #         target_model, prep_fiducials, meas_fiducials, germs, max_lengths
    #     )

    #     self.gst_sequences = [circuit.str for circuit in lsgst_lists[-1]]
    #     if len(self.gst_sequences) > self.GST_SEQUENCE_COUNT_LIMIT:
    #         raise ValueError(
    #             f"GST sequence count ({len(self.gst_sequences)}) exceeds the limit ({self.GST_SEQUENCE_COUNT_LIMIT}). "
    #             "Reduce max_lengths, the fiducial set, or the germ set."
    #         )
    #     return self.gst_sequences
    
    # def _build_gate_map(self, circuit_strings: list[str]) -> dict[str, int]:
    #     """Build a gate map from a list of circuit strings.
    #     Used to construct the PREP_FIDUCIAL_MAP, MEAS_FIDUCIAL_MAP and GERM_MAP dictionaries.
        
    #     Args:
    #         circuit_strings: List of circuit strings.
    #     """
    #     gate_map = {"{}": 0}
    #     idx = 1
    #     for circuit_string in circuit_strings:
    #         if not circuit_string in gate_map.keys():
    #             gate_map[circuit_string] = idx
    #             idx += 1
    #     return gate_map
    



# PREP_FIDUCIAL_MAP: dict[str, int] = {"{}": 0}
# MEAS_FIDUCIAL_MAP: dict[str, int] = {"{}": 0}
# GERM_MAP: dict[str, int] = {"{}": 0}