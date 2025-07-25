from typing import Optional, Literal

import numpy as np
from qm.qua import *

from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair
from quam_builder.architecture.superconducting.components.cross_resonance import CrossResonanceMW, CrossResonanceIQ
from qm.qua._dsl import QuaExpression, QuaVariable


qua_T = QuaVariable | QuaExpression


def get_cr_elements(qp: AnyTransmonPair):
    qc = qp.qubit_control
    qt = qp.qubit_target
    cr = qp.cross_resonance
    cr_elems = [qc.xy.name, qt.xy.name, cr.name]
    return qc, qt, cr, cr_elems


def play_cross_resonance(
    qubit_pair: AnyTransmonPair,
    cr_type: Literal["direct", "direct+cancel", "direct+echo", "direct+cancel+echo"] = "direct",
    cr_drive_amp_scaling: Optional[float | qua_T] = None,
    cr_drive_phase: Optional[float | qua_T] = None,
    cr_cancel_amp_scaling: Optional[float | qua_T] = None,
    cr_cancel_phase: Optional[float | qua_T] = None,
    cr_duration_clock_cycles: Optional[float | qua_T] = None,
    wf_type: Literal["square", "cosine", "gauss", "flattop"] = "square",
):
    qc, qt, cr, elems = get_cr_elements(qubit_pair)

    def _play_cr_pulse(
        elem,
        wf_type: str,
        amp_scale: Optional[float | qua_T],
        duration: Optional[float | qua_T],
        sgn: int = 1,
    ):
        if amp_scale is None and duration is None:
            elem.play(wf_type)
        elif amp_scale is None:
            elem.play(wf_type, duration=duration)
        elif duration is None:
            elem.play(wf_type, amplitude_scale=sgn * amp_scale)
        else:
            elem.play(wf_type, amplitude_scale=sgn * amp_scale, duration=duration)

    def cr_drive_shift_phase():
        if cr_drive_phase is not None:
            cr.frame_rotation_2pi(cr_drive_phase)


    def cr_cancel_shift_phase():
        if cr_cancel_phase is not None:
            qt.xy.frame_rotation_2pi(cr_cancel_phase)
        

    def cr_drive_play(
        sgn: Literal["direct", "echo"] = "direct",
        wf_type=wf_type,
    ):
        _play_cr_pulse(
            elem=cr,
            wf_type=wf_type,
            amp_scale=cr_drive_amp_scaling,
            duration=cr_duration_clock_cycles,
            sgn=1 if sgn == "direct" else -1,
        )

    def cr_cancel_play(
        sgn: Literal["direct", "echo"] = "direct",
        wf_type=wf_type,
    ):
        _play_cr_pulse(
            elem=qt.xy,
            wf_type=f"cr_{wf_type}_{qubit_pair.name}",
            amp_scale=cr_cancel_amp_scaling,
            duration=cr_duration_clock_cycles,
            sgn=1 if sgn == "direct" else -1,
        )

    if cr_type == "direct":
        cr_drive_shift_phase()
        align(*elems)

        cr_drive_play(sgn="direct")
        align(*elems)

        reset_frame(cr.name)
        align(*elems)

    elif cr_type == "direct+echo":
        cr_drive_shift_phase()
        align(*elems)

        cr_drive_play(sgn="direct")
        align(*elems)

        qc.xy.play("x180")
        align(*elems)

        cr_drive_play(sgn="echo")
        align(*elems)

        qc.xy.play("x180")
        align(*elems)

        reset_frame(cr.name)
        align(*elems)

    elif cr_type == "direct+cancel":
        cr_drive_shift_phase()
        cr_cancel_shift_phase()
        align(*elems)

        cr_drive_play(sgn="direct")
        cr_cancel_play(sgn="direct")
        align(*elems)

        reset_frame(cr.name)
        reset_frame(qt.xy.name)
        align(*elems)

    elif cr_type == "direct+cancel+echo":
        cr_drive_shift_phase()
        cr_cancel_shift_phase()
        align(*elems)

        cr_drive_play(sgn="direct")
        cr_cancel_play(sgn="direct")
        align(*elems)

        qc.xy.play("x180")
        align(*elems)

        cr_drive_play(sgn="echo")
        cr_cancel_play(sgn="echo")
        align(*elems)

        qc.xy.play("x180")
        align(*elems)

        reset_frame(cr.name)
        reset_frame(qt.xy.name)
        align(*elems)
