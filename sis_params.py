from dataclasses import dataclass, field
from typing import Dict
from simtk.unit import Quantity, angstrom, kilocalorie_per_mole, radian

@dataclass
class SISForceField:
    # Switches
    bond:       bool = True
    angle:      bool = False
    angle_ReB:  bool = True
    dihexp:     bool = True
    wca:        bool = True
    bp:         bool = True

    # Bond
    bond_k:  Quantity = field(default_factory=lambda: Quantity(15.0, kilocalorie_per_mole/(angstrom**2)))
    bond_r0: Quantity = field(default_factory=lambda: Quantity(6.13, angstrom))

    # Angle
    angle_k:  Quantity = field(default_factory=lambda: Quantity(5.0, kilocalorie_per_mole/(radian**2)))
    angle_a0: Quantity = field(default_factory=lambda: Quantity(2.6358, radian))

    # Angle (Restricted Bending)
    angle_ReB_k:  Quantity = field(default_factory=lambda: Quantity(5.0, kilocalorie_per_mole))
    angle_ReB_a0: Quantity = field(default_factory=lambda: Quantity(2.6358, radian))

    # Dihedral (exponential)
    dihexp_k:  Quantity = field(default_factory=lambda: Quantity(1.4, kilocalorie_per_mole))
    dihexp_w:  Quantity = field(default_factory=lambda: Quantity(3.0, radian**(-2)))
    dihexp_p0: Quantity = field(default_factory=lambda: Quantity(0.267, radian))

    # WCA
    wca_epsilon: Quantity = field(default_factory=lambda: Quantity(2.0, kilocalorie_per_mole))
    wca_sigma:   Quantity = field(default_factory=lambda: Quantity(10.0, angstrom))
    wca_exclusions: Dict[str, bool] = field(default_factory=lambda: {'1-2': True, '1-3': True})

    # Base pair
    dS0: float = -17.0
    NNend: bool = True

    GC_bond_k:  Quantity = field(default_factory=lambda: Quantity(1.26,  angstrom**(-2)))
    GC_bond_r:  Quantity = field(default_factory=lambda: Quantity(15.06, angstrom))
    GC_angl_k1: Quantity = field(default_factory=lambda: Quantity(7.12,  radian**(-2)))
    GC_angl_k2: Quantity = field(default_factory=lambda: Quantity(7.04,  radian**(-2)))
    GC_angl_k3: Quantity = field(default_factory=lambda: Quantity(6.35,  radian**(-2)))
    GC_angl_k4: Quantity = field(default_factory=lambda: Quantity(9.23,  radian**(-2)))
    GC_angl_theta1: Quantity = field(default_factory=lambda: Quantity(1.81, radian))
    GC_angl_theta2: Quantity = field(default_factory=lambda: Quantity(1.82, radian))
    GC_angl_theta3: Quantity = field(default_factory=lambda: Quantity(0.92, radian))
    GC_angl_theta4: Quantity = field(default_factory=lambda: Quantity(0.92, radian))
    GC_dihd_k1: float = 2.11
    GC_dihd_k2: float = 2.31
    GC_dihd_phi1: Quantity = field(default_factory=lambda: Quantity(1.96,  radian))
    GC_dihd_phi2: Quantity = field(default_factory=lambda: Quantity(1.15,  radian))

    AU_bond_k:  Quantity = field(default_factory=lambda: Quantity(0.79,  angstrom**(-2)))
    AU_bond_r:  Quantity = field(default_factory=lambda: Quantity(15.01, angstrom))
    AU_angl_k1: Quantity = field(default_factory=lambda: Quantity(7.74,  radian**(-2)))
    AU_angl_k2: Quantity = field(default_factory=lambda: Quantity(6.79,  radian**(-2)))
    AU_angl_k3: Quantity = field(default_factory=lambda: Quantity(7.16,  radian**(-2)))
    AU_angl_k4: Quantity = field(default_factory=lambda: Quantity(8.92,  radian**(-2)))
    AU_angl_theta1: Quantity = field(default_factory=lambda: Quantity(1.81, radian))
    AU_angl_theta2: Quantity = field(default_factory=lambda: Quantity(1.81, radian))
    AU_angl_theta3: Quantity = field(default_factory=lambda: Quantity(0.93, radian))
    AU_angl_theta4: Quantity = field(default_factory=lambda: Quantity(0.92, radian))
    AU_dihd_k1: float = 2.61
    AU_dihd_k2: float = 2.06
    AU_dihd_phi1: Quantity = field(default_factory=lambda: Quantity(1.98,  radian))
    AU_dihd_phi2: Quantity = field(default_factory=lambda: Quantity(1.09,  radian))

    GU_bond_k:  Quantity = field(default_factory=lambda: Quantity(0.9,  angstrom**(-2)))
    GU_bond_r:  Quantity = field(default_factory=lambda: Quantity(14.85, angstrom))
    GU_angl_k1: Quantity = field(default_factory=lambda: Quantity(12.12,  radian**(-2)))
    GU_angl_k2: Quantity = field(default_factory=lambda: Quantity( 3.37,  radian**(-2)))
    GU_angl_k3: Quantity = field(default_factory=lambda: Quantity( 6.54,  radian**(-2)))
    GU_angl_k4: Quantity = field(default_factory=lambda: Quantity(16.87,  radian**(-2)))
    GU_angl_theta1: Quantity = field(default_factory=lambda: Quantity(1.86, radian))
    GU_angl_theta2: Quantity = field(default_factory=lambda: Quantity(1.77, radian))
    GU_angl_theta3: Quantity = field(default_factory=lambda: Quantity(0.96, radian))
    GU_angl_theta4: Quantity = field(default_factory=lambda: Quantity(0.94, radian))
    GU_dihd_k1: float = 2.12
    GU_dihd_k2: float = 1.99
    GU_dihd_phi1: Quantity = field(default_factory=lambda: Quantity(1.89,  radian))
    GU_dihd_phi2: Quantity = field(default_factory=lambda: Quantity(1.16,  radian))

    def __str__(self):
        tab = '    '
        s =  "Force field:\n"
        s += tab + "Switches\n"
        s += tab + tab + f"bond:      {self.bond}\n"
        s += tab + tab + f"angle:     {self.angle}\n"
        s += tab + tab + f"angle_ReB: {self.angle_ReB}\n"
        s += tab + tab + f"dihexp:    {self.dihexp}\n"
        s += tab + tab + f"wca:       {self.wca}\n"
        s += tab + tab + f"bp:        {self.bp}\n"
        if self.bond:
            s += tab + "Bond:\n"
            s += tab + tab + f"k:  {self.bond_k}\n"
            s += tab + tab + f"r0: {self.bond_r0}\n"
        if self.angle:
            s += tab + "Angle:\n"
            s += tab + tab + f"k:  {self.angle_k}\n"
            s += tab + tab + f"a0: {self.angle_a0}\n"
        if self.angle_ReB:
            s += tab + "Angle_ReB:\n"
            s += tab + tab + f"k:  {self.angle_ReB_k}\n"
            s += tab + tab + f"a0: {self.angle_ReB_a0}\n"
        if self.dihexp:
            s += tab + "Dihedral(exponential)\n"
            s += tab + tab + f"k:  {self.dihexp_k}\n"
            s += tab + tab + f"w:  {self.dihexp_w}\n"
            s += tab + tab + f"p0: {self.dihexp_p0}\n"
        if self.wca:
            s += tab + "WCA:\n"
            s += tab + tab + f"epsilon: {self.wca_epsilon}\n"
            s += tab + tab + f"sigma:   {self.wca_sigma}\n"
            s += tab + tab + f"exclusions: {self.wca_exclusions}\n"
        if self.bp:
            s += tab + "Basepair:\n"
            s += tab + tab + f"dS0: {self.dS0}\n"
            s += tab + tab + f"NNend: {self.NNend}\n"
            s += tab + tab + f"GC_bond_k:  {self.GC_bond_k}\n"
            s += tab + tab + f"GC_bond_r:  {self.GC_bond_r}\n"
            s += tab + tab + f"GC_angl_k1: {self.GC_angl_k1}\n"
            s += tab + tab + f"GC_angl_k2: {self.GC_angl_k2}\n"
            s += tab + tab + f"GC_angl_k3: {self.GC_angl_k3}\n"
            s += tab + tab + f"GC_angl_k4: {self.GC_angl_k4}\n"
            s += tab + tab + f"GC_angl_theta1: {self.GC_angl_theta1}\n"
            s += tab + tab + f"GC_angl_theta2: {self.GC_angl_theta2}\n"
            s += tab + tab + f"GC_angl_theta3: {self.GC_angl_theta3}\n"
            s += tab + tab + f"GC_angl_theta4: {self.GC_angl_theta4}\n"
            s += tab + tab + f"GC_dihd_k1:     {self.GC_dihd_k1}\n"
            s += tab + tab + f"GC_dihd_k2:     {self.GC_dihd_k2}\n"
            s += tab + tab + f"GC_dihd_phi1:   {self.GC_dihd_phi1}\n"
            s += tab + tab + f"GC_dihd_phi2:   {self.GC_dihd_phi2}\n"
            s += tab + tab + f"AU_bond_k:  {self.AU_bond_k}\n"
            s += tab + tab + f"AU_bond_r:  {self.AU_bond_r}\n"
            s += tab + tab + f"AU_angl_k1: {self.AU_angl_k1}\n"
            s += tab + tab + f"AU_angl_k2: {self.AU_angl_k2}\n"
            s += tab + tab + f"AU_angl_k3: {self.AU_angl_k3}\n"
            s += tab + tab + f"AU_angl_k4: {self.AU_angl_k4}\n"
            s += tab + tab + f"AU_angl_theta1: {self.AU_angl_theta1}\n"
            s += tab + tab + f"AU_angl_theta2: {self.AU_angl_theta2}\n"
            s += tab + tab + f"AU_angl_theta3: {self.AU_angl_theta3}\n"
            s += tab + tab + f"AU_angl_theta4: {self.AU_angl_theta4}\n"
            s += tab + tab + f"AU_dihd_k1:     {self.AU_dihd_k1}\n"
            s += tab + tab + f"AU_dihd_k2:     {self.AU_dihd_k2}\n"
            s += tab + tab + f"AU_dihd_phi1:   {self.AU_dihd_phi1}\n"
            s += tab + tab + f"AU_dihd_phi2:   {self.AU_dihd_phi2}\n"
            s += tab + tab + f"GU_bond_k:  {self.GU_bond_k}\n"
            s += tab + tab + f"GU_bond_r:  {self.GU_bond_r}\n"
            s += tab + tab + f"GU_angl_k1: {self.GU_angl_k1}\n"
            s += tab + tab + f"GU_angl_k2: {self.GU_angl_k2}\n"
            s += tab + tab + f"GU_angl_k3: {self.GU_angl_k3}\n"
            s += tab + tab + f"GU_angl_k4: {self.GU_angl_k4}\n"
            s += tab + tab + f"GU_angl_theta1: {self.GU_angl_theta1}\n"
            s += tab + tab + f"GU_angl_theta2: {self.GU_angl_theta2}\n"
            s += tab + tab + f"GU_angl_theta3: {self.GU_angl_theta3}\n"
            s += tab + tab + f"GU_angl_theta4: {self.GU_angl_theta4}\n"
            s += tab + tab + f"GU_dihd_k1:     {self.GU_dihd_k1}\n"
            s += tab + tab + f"GU_dihd_k2:     {self.GU_dihd_k2}\n"
            s += tab + tab + f"GU_dihd_phi1:   {self.GU_dihd_phi1}\n"
            s += tab + tab + f"GU_dihd_phi2:   {self.GU_dihd_phi2}\n"
        return s[:-1]  # not including the last "\n"

    def read_toml(self, tomlfile):
        import toml
        tm = toml.load(tomlfile)

        self.bond       = False
        self.angle      = False
        self.angle_ReB  = False
        self.dihexp     = False
        self.wca        = False
        self.bp         = False

        if 'bond' in tm['potential']:
            self.bond = True
            self.bond_k  = tm['potential']['bond']['k'] * kilocalorie_per_mole/(angstrom**2)
            self.bond_r0 = tm['potential']['bond']['r0'] * angstrom

        if 'angle' in tm['potential']:
            self.angle = True
            self.angle_k   = tm['potential']['angle']['k'] * kilocalorie_per_mole/(radian**2)
            self.angle_a0  = tm['potential']['angle']['a0'] * radian

        if 'angle_ReB' in tm['potential']:
            self.angle_ReB = True
            self.angle_k   = tm['potential']['angle_ReB']['k'] * kilocalorie_per_mole
            self.angle_a0  = tm['potential']['angle_ReB']['a0'] * radian

        if 'dihedral_exp' in tm['potential']:
            self.dihexp = True
            self.dihexp_k  = tm['potential']['dihedral_exp']['k'] * kilocalorie_per_mole
            self.dihexp_w  = tm['potential']['dihedral_exp']['w'] / (radian**2)
            self.dihexp_p0 = tm['potential']['dihedral_exp']['phi0'] * radian

        if 'basepair' in tm['potential']:
            self.bp = True
            self.dS0 = tm['potential']['basepair']['dS0']
            self.NNend = tm['potential']['basepair']['NNend']

            self.GC_bond_k = tm['potential']['basepair']['GC']['bond_k'] * angstrom**(-2)
            self.GC_bond_r = tm['potential']['basepair']['GC']['bond_r'] * angstrom
            self.GC_angl_k1 = tm['potential']['basepair']['GC']['angl_k1'] * radian**(-2)
            self.GC_angl_k2 = tm['potential']['basepair']['GC']['angl_k2'] * radian**(-2)
            self.GC_angl_k3 = tm['potential']['basepair']['GC']['angl_k3'] * radian**(-2)
            self.GC_angl_k4 = tm['potential']['basepair']['GC']['angl_k4'] * radian**(-2)
            self.GC_angl_theta1 = tm['potential']['basepair']['GC']['angl_theta1'] * radian
            self.GC_angl_theta2 = tm['potential']['basepair']['GC']['angl_theta2'] * radian
            self.GC_angl_theta3 = tm['potential']['basepair']['GC']['angl_theta3'] * radian
            self.GC_angl_theta4 = tm['potential']['basepair']['GC']['angl_theta4'] * radian
            self.GC_dihd_k1 = tm['potential']['basepair']['GC']['dihd_k1']
            self.GC_dihd_k2 = tm['potential']['basepair']['GC']['dihd_k2']
            self.GC_dihd_phi1 = tm['potential']['basepair']['GC']['dihd_phi1'] * radian
            self.GC_dihd_phi2 = tm['potential']['basepair']['GC']['dihd_phi2'] * radian

            self.AU_bond_k = tm['potential']['basepair']['AU']['bond_k'] * angstrom**(-2)
            self.AU_bond_r = tm['potential']['basepair']['AU']['bond_r'] * angstrom
            self.AU_angl_k1 = tm['potential']['basepair']['AU']['angl_k1'] * radian**(-2)
            self.AU_angl_k2 = tm['potential']['basepair']['AU']['angl_k2'] * radian**(-2)
            self.AU_angl_k3 = tm['potential']['basepair']['AU']['angl_k3'] * radian**(-2)
            self.AU_angl_k4 = tm['potential']['basepair']['AU']['angl_k4'] * radian**(-2)
            self.AU_angl_theta1 = tm['potential']['basepair']['AU']['angl_theta1'] * radian
            self.AU_angl_theta2 = tm['potential']['basepair']['AU']['angl_theta2'] * radian
            self.AU_angl_theta3 = tm['potential']['basepair']['AU']['angl_theta3'] * radian
            self.AU_angl_theta4 = tm['potential']['basepair']['AU']['angl_theta4'] * radian
            self.AU_dihd_k1 = tm['potential']['basepair']['AU']['dihd_k1']
            self.AU_dihd_k2 = tm['potential']['basepair']['AU']['dihd_k2']
            self.AU_dihd_phi1 = tm['potential']['basepair']['AU']['dihd_phi1'] * radian
            self.AU_dihd_phi2 = tm['potential']['basepair']['AU']['dihd_phi2'] * radian

            self.GU_bond_k = tm['potential']['basepair']['GU']['bond_k'] * angstrom**(-2)
            self.GU_bond_r = tm['potential']['basepair']['GU']['bond_r'] * angstrom
            self.GU_angl_k1 = tm['potential']['basepair']['GU']['angl_k1'] * radian**(-2)
            self.GU_angl_k2 = tm['potential']['basepair']['GU']['angl_k2'] * radian**(-2)
            self.GU_angl_k3 = tm['potential']['basepair']['GU']['angl_k3'] * radian**(-2)
            self.GU_angl_k4 = tm['potential']['basepair']['GU']['angl_k4'] * radian**(-2)
            self.GU_angl_theta1 = tm['potential']['basepair']['GU']['angl_theta1'] * radian
            self.GU_angl_theta2 = tm['potential']['basepair']['GU']['angl_theta2'] * radian
            self.GU_angl_theta3 = tm['potential']['basepair']['GU']['angl_theta3'] * radian
            self.GU_angl_theta4 = tm['potential']['basepair']['GU']['angl_theta4'] * radian
            self.GU_dihd_k1 = tm['potential']['basepair']['GU']['dihd_k1']
            self.GU_dihd_k2 = tm['potential']['basepair']['GU']['dihd_k2']
            self.GU_dihd_phi1 = tm['potential']['basepair']['GU']['dihd_phi1'] * radian
            self.GU_dihd_phi2 = tm['potential']['basepair']['GU']['dihd_phi2'] * radian

        if 'wca' in tm['potential']:
            self.wca = True
            self.wca_epsilon = tm['potential']['wca']['epsilon'] * kilocalorie_per_mole
            self.wca_sigma   = tm['potential']['wca']['sigma'] * angstrom

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

