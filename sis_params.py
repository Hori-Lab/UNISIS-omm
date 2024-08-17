from dataclasses import dataclass, field
from typing import Dict

@dataclass
class SISForceField:
    bond_k: float = 15.0
    bond_r0: float = 6.13

    angle_k: float = 10.0
    angle_a0: float = 2.618

    dih_exp_k: float = 1.4
    dih_exp_w: float = 3.0
    dih_exp_phi0: float = 0.267

    wca_sigma: float = 10.0
    wca_epsilon: float = 2.0

    dS0 = -1.0

    GC_bond_k: float = 1.0
    GC_bond_r: float = 1.0
    GC_angl_k1: float = 1.5
    GC_angl_k2: float = 1.5
    GC_angl_k3: float = 1.5
    GC_angl_k4: float = 1.5
    GC_angl_theta1: float = 1.8326
    GC_angl_theta2: float = 1.8326
    GC_angl_theta3: float = 0.9425
    GC_angl_theta4: float = 0.9425
    GC_dihd_k1: float = 0.5
    GC_dihd_k2: float = 0.5
    GC_dihd_phi1: float = 1.8326
    GC_dihd_phi2: float = 1.1345

    AU_bond_k: float = 1.0
    AU_bond_r: float = 1.0
    AU_angl_k1: float = 1.5
    AU_angl_k2: float = 1.5
    AU_angl_k3: float = 1.5
    AU_angl_k4: float = 1.5
    AU_angl_theta1: float = 1.8326
    AU_angl_theta2: float = 1.8326
    AU_angl_theta3: float = 0.9425
    AU_angl_theta4: float = 0.9425
    AU_dihd_k1: float = 0.5
    AU_dihd_k2: float = 0.5
    AU_dihd_phi1: float = 1.8326
    AU_dihd_phi2: float = 1.1345

    GU_bond_k: float = 1.0
    GU_bond_r: float = 1.0
    GU_angl_k1: float = 1.5
    GU_angl_k2: float = 1.5
    GU_angl_k3: float = 1.5
    GU_angl_k4: float = 1.5
    GU_angl_theta1: float = 1.8326
    GU_angl_theta2: float = 1.8326
    GU_angl_theta3: float = 0.9425
    GU_angl_theta4: float = 0.9425
    GU_dihd_k1: float = 0.5
    GU_dihd_k2: float = 0.5
    GU_dihd_phi1: float = 1.8326
    GU_dihd_phi2: float = 1.1345

@dataclass
class NearestNeighbor:
    dH: Dict[str, float] = field(default_factory=lambda: {
        'GC_CG':  -16.52,
        'CC_GG':  -13.94,
        'GA_CU':  -13.75,
        'CG_GC':   -9.61,
        'AC_UG':  -11.98,
        'CA_GU':  -10.47,
        'AG_UC':   -9.34,
        'UA_AU':   -9.16,
        'AU_UA':   -8.91,
        'AA_UU':   -7.44,
        'GC_UG':  -14.73,
        'CU_GG':   -9.26,
        'GG_CU':  -12.41,
        'CG_GU':   -5.64,
        'AU_UG':   -9.23,
        'GA_UU':  -10.58,
        'UG_GU':   -8.76,
        'UA_GU':   -2.72,
        'GG_UU':   -9.06,
        'GU_UG':   -7.66,
        'AG_UU':   -5.10,
    })

    dS: Dict[str, float] = field(default_factory=lambda: {
        'GC_CG':  -42.13,
        'CC_GG':  -34.41,
        'GA_CU':  -36.53,
        'CG_GC':  -23.46,
        'AC_UG':  -31.37,
        'CA_GU':  -27.08,
        'AG_UC':  -23.66,
        'UA_AU':  -25.40,
        'AU_UA':  -25.22,
        'AA_UU':  -20.98,
        'GC_UG':  -40.32,
        'CU_GG':  -23.64,
        'GG_CU':  -34.23,
        'CG_GU':  -14.83,
        'AU_UG':  -27.32,
        'GA_UU':  -32.19,
        'UG_GU':  -27.04,
        'UA_GU':   -8.08,
        'GG_UU':  -28.57,
        'GU_UG':  -24.11,
        'AG_UU':  -16.53,
    })

    end_dH: Dict[str, float] = field(default_factory=lambda: {
        'AUonAU':  4.36,
        'AUonCG':  3.17,
        'AUonGU':  5.16,
        'GUonCG':  3.91,
        'GUonAU':  3.65,
        'GUonGU':  6.23,
    })

    end_dS: Dict[str, float] = field(default_factory=lambda: {
        'AUonAU': 13.35,
        'AUonCG':  8.79,
        'AUonGU': 18.96,
        'GUonCG': 12.17,
        'GUonAU': 12.78,
        'GUonGU': 22.47,
    })

    

