"""Circuit table shared by the QUA program and the analysis of the CZ unitary (MEADD + Floquet) node.

Every circuit is one row (exp, prep, ro, ncz). The QUA program runs the rows in this order inside the
shot-averaging loop, and the analysis looks rows up by the same integer codes.

Conventions:
    * L = qubit_pair.qubit_control, R = qubit_pair.qubit_target.
    * Two-qubit basis |b_L b_R>, index 2 * b_L + b_R. b = 1 means the qubit was found in |1>.
"""

from typing import Dict, List

# Sub-experiments (repeated block per iteration of the depth loop)
EXP_PHI = 0  # CZ (with virtual-Z corrections), then X on L and R
EXP_THETA_XX = 1  # CZ (no virtual-Z corrections), then X on L and R
EXP_THETA_YX = 2  # CZ (no virtual-Z corrections), then Y on L and X on R
EXP_FLOQUET = 3  # CZ (with virtual-Z corrections) alone

# State preparations (P = |+>), written |L R>
PREP_P0 = 0  # y90 on L
PREP_0P = 1  # y90 on R
PREP_10 = 2  # x180 on L
PREP_P1 = 3  # y90 on L, x180 on R
PREP_1P = 4  # x180 on L, y90 on R

# Readout rotations played before measuring both qubits in Z
RO_XX = 0  # -y90 on L and R
RO_YY = 1  # x90 on L and R
RO_ZZ = 2  # nothing
RO_XODD = 3  # Bell readout of X_odd: R -y90, CZ, then R y90 and L -y90
RO_YODD = 4  # Bell readout of Y_odd: R -y90, CZ, then R y90 and L x90


def build_circuit_table(
    max_cz_meadd: int, step_cz_meadd: int, max_cz_floquet: int, include_floquet_phi: bool = False
) -> List[Dict[str, int]]:
    """Return the list of circuits as dicts with keys exp, prep, ro and ncz, in execution order.

    MEADD circuits use depths 0, step, 2 * step, ..., max_cz_meadd (the number of CZ gates, always even).
    Floquet circuits use every depth from 0 to max_cz_floquet so that the phase unwrap works.
    """
    if step_cz_meadd % 2 or max_cz_meadd % step_cz_meadd:
        raise ValueError("step_cz_meadd must be even and max_cz_meadd must be a multiple of step_cz_meadd.")

    rows = []
    for ncz in range(0, max_cz_meadd + 1, step_cz_meadd):
        for prep in (PREP_0P, PREP_P0):
            for ro in (RO_XX, RO_YY):
                rows.append(dict(exp=EXP_PHI, prep=prep, ro=ro, ncz=ncz))
        for exp in (EXP_THETA_XX, EXP_THETA_YX):
            for ro in (RO_ZZ, RO_XODD, RO_YODD):
                rows.append(dict(exp=exp, prep=PREP_10, ro=ro, ncz=ncz))

    floquet_preps = (PREP_0P, PREP_P0) + ((PREP_P1, PREP_1P) if include_floquet_phi else ())
    for ncz in range(0, max_cz_floquet + 1):
        for prep in floquet_preps:
            for ro in (RO_XX, RO_YY):
                rows.append(dict(exp=EXP_FLOQUET, prep=prep, ro=ro, ncz=ncz))
    return rows
