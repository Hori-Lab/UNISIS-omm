#!/usr/bin/env python
""" OpenMM script to run SIS-ML simulations

Code to run OpenMM simulation of the Single-Interaction-Site RNA model 
with a machine learning potential trained by TorchMD-Net.

references:
 https://github.com/openmm/openmm-torch/blob/master/tutorials/openmm-torch-nnpops.ipynb
 https://github.com/openmm/openmm-torch/issues/135
 The original framework of the code was adopted from https://github.com/tienhungf91/RNA_llps
"""
__author__ = "Naoto Hori"

import os
import sys
import time
import re
import itertools as it
import argparse
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict
from math import sqrt, pi, cos

from numpy import diag
from simtk import unit
from simtk.unit import Quantity
from openmm import app
import openmm as omm

from sis_params import SISForceField

"""
* Following modules will be imported later if needed.
    import subprocess
    import yaml
    import toml
    import torch
    from openmmtorch import TorchForce
    from torchmdnet.models.model import load_model
"""

################################################
#         Utility functions
################################################

def prev_and_next(iterable):
    prevs, items, nexts = it.tee(iterable, 3)
    prevs = it.chain([None], prevs)
    nexts = it.chain(it.islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)
    # For N, this will geneerate [None,0,1], [0,1,2], ..., [N-3,N-2,N-1], [N-2,N-1,None]

def fours(iterable):
    a, b, c, d = it.tee(iterable, 4)
    next(b, None)
    next(c, None); next(c, None)
    next(d, None); next(d, None); next(d, None)
    return zip(a, b, c, d)
    # For N, this will geneerate [0,1,2,3], [1,2,3,4], ..., [N-4,N-3,N-2,N-1] (no None)

def atm_index(res):
    #return res.atoms[0].index
    for atom in res.atoms():
        return atom.index

def get_atom(res):
    for atom in res.atoms():
        return atom

################################################
#         Constants
################################################
#KELVIN_TO_KT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB / unit.kilocalorie_per_mole
#print KELVIN_TO_KT
FILENAME_XML_DEFAULT = 'rna_cg2.xml'

################################################
#          Parser
################################################

parser = argparse.ArgumentParser(description='OpenMM script for the SIS-RNA model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser_ctrl = parser.add_mutually_exclusive_group(required=True)
parser_ctrl.add_argument('--input', type=str, help='TOML format input file')
parser_ctrl.add_argument('--tmyaml', type=str, help='TorchMD format YAML file')

parser.add_argument('--ff', type=str, help='TOML format force-field file')
parser.add_argument('--xml', type=str, default=None, help='XML file for topology information')

parser.add_argument('--cuda', action='store_true', default=False)
#parser_device = parser.add_mutually_exclusive_group()
#parser_device.add_argument('--cpu', action='store_true', default=False)
#parser_device.add_argument('--cuda', action='store_true', default=False)

#parser_init = parser.add_mutually_exclusive_group(required=True)
#parser_init.add_argument('-i','--inixyz', type=str, default=None, help='initial xyz file')
#parser_init.add_argument('-N','--nbead', type=int, default=0, help='number of beads')
#
#parser.add_argument('-n','--step', type=int, default='10000',
#                    help='Number of step [10000]')
#
#parser.add_argument('--dt', type=float, default='50.0',
#                    help='integration time step in fs [50.0]')
#
#parser.add_argument('-T','--temperature', type=float, default='300.0',
#                    help='Temperature (K) [300.0]')
#
#parser.add_argument('-a','--angle', action='store_true',
#                    help='Enable the angle potential')
#
#parser.add_argument('-b','--ReB', action='store_true',
#                    help='Enable the Restricted Bending potential')
#
#parser.add_argument('-d','--dihexp', action='store_true',
#                    help='Enable the dihedral potential (exponential form)')
#
#parser.add_argument('-I','--ionic_strength', type=float, default='-1.',
#                    help='Ionic strength (M) [Default: no electrostatic (-1)]')
#parser.add_argument('-c','--cutoff', type=float, default='30.',
#                    help='Electrostatic cutoff (A) [30.0]')
#
#parser.add_argument('-t','--traj', type=str, default='saw.dcd',
#                    help='trajectory output')
#parser.add_argument('-e','--energy', type=str, default='saw.ene',
#                    help='energy decomposition')
#parser.add_argument('-o','--output', type=str, default='saw.out',
#                    help='status and energy output')
#parser.add_argument('-x','--frequency', type=int, default='100',
#                    help='output and restart frequency')
#parser.add_argument('-f','--finalxyz', type=str, default='final.xyz',
#                    help='final structure xyz output')
#
#parser.add_argument('-R','--restart', action='store_true',
#                    help='flag to restart simulation')
#parser.add_argument('-r','--res_file', type=str, default='saw.chk',
#                    help='checkpoint file for restart')

#parser.add_argument('--platform', type=str, default=None,
#                    help='Platform')
#parser.add_argument('--CUDAdevice', type=str, default=None,
#                    help='CUDA device ID')

args = parser.parse_args()

################################################
#   Output the program and execution information
################################################
print('Program: OpenMM script for the SIS RNA model')
print('    File: ' + os.path.realpath(__file__))
# Output Git hash
try:
    import subprocess
    src_path = os.path.dirname(os.path.realpath(__file__))
    h = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=src_path, stderr=subprocess.DEVNULL).decode('ascii').strip()
    print('    Git hash: ' + h)
except:
    pass
print('')

print('Execution:')
print('    Time: ' + str(datetime.now()) + ' (UTC: ' + str(datetime.utcnow()) + ')')
print('    Host: ' + os.uname().nodename)
print('    OS: ' + os.uname().version)
print('    Python: ' + sys.version)
print('    OpenMM version: ' + omm.version.full_version)
print('    OpenMM library path: ' + omm.version.openmm_library_path)
print('    Directory: ' + os.getcwd())
print('    Command:', end='')
for s in sys.argv:
    print(' '+s, end='')
print('')
print('')


################################################
#          Control object
################################################
@dataclass
class Control:    ### structure to group all simulation parameter
    device: str = ''
    restart: bool = False
    restart_file: str = None
    minimization: bool = False

    Nstep: int        = 10
    Nstep_save: int   = 1
    Nstep_log:  int   = 1

    xml:         str = None
    ff:          str = None
    infile_pdb:  str = None
    outfile_log: str = './md.log'
    outfile_out: str = './md.out'
    outfile_dcd: str = './md.dcd'
    outfile_rst: str = './md.rst'

    temp: Quantity     = field(default_factory=lambda: Quantity(300.0, unit.kelvin))
    LD_temp: Quantity  = field(default_factory=lambda: Quantity(300.0, unit.kelvin))
    LD_gamma: Quantity = field(default_factory=lambda: Quantity(0.5, unit.picosecond**(-1)))
    LD_dt: Quantity    = field(default_factory=lambda: Quantity(50, unit.femtoseconds))

    ele: bool = False
    ele_ionic_strength: float = 0.15
    ele_cutoff_type: int = 1
    ele_cutoff_factor: float = 50.0
    ele_no_charge: List = field(default_factory=lambda: [])
    ele_length_per_charge: Quantity = field(default_factory=lambda: Quantity(4.38178046, unit.angstrom))
    ele_exclusions: Dict = field(default_factory=lambda: {'1-2': True, '1-3': False})

    use_NNP: bool = False
    NNP_model: str = ''
    NNP_emblist: List = None
        #field(default_factory=lambda: 
        #   [5,2,3,4,1,4,2,1,2,2,4,3,1,4,1,3,1,4,3,2,4,3,1,4,1,2,3,1,5])

    #box = 0.
    #Kconc: float = -1.
    #b: float = 4.38178046 * unit.angstrom / unit.elementary_charge
    #cutoff: float = 0.
    #epsilon: float = 0.

    def __str__(self):
        return (f"Control:\n"
              + f"    device: {self.device}\n"
              + f"    restart: {self.restart}\n"
              + f"    restart_file: {self.restart_file}\n"
              + f"    minimization: {self.minimization}\n"
              + f"    Nstep: {self.Nstep}\n"
              + f"    Nstep_save: {self.Nstep_save}\n"
              + f"    Nstep_log {self.Nstep_log}\n"
              + f"    xml: {self.xml}\n"
              + f"    ff: {self.ff}\n"
              + f"    infile_pdb: {self.infile_pdb}\n"
              + f"    outfile_log: {self.outfile_log}\n"
              + f"    outfile_out: {self.outfile_out}\n"
              + f"    outfile_dcd: {self.outfile_dcd}\n"
              + f"    outfile_rst: {self.outfile_rst}\n"
              + f"    temp: {self.temp}\n"
              + f"    LD_temp: {self.LD_temp}\n"
              + f"    LD_gamma: {self.LD_gamma}\n"
              + f"    LD_dt: {self.LD_dt}\n"
              + f"    ele: {self.ele}\n"
              + f"    ele_ionic_strength: {self.ele_ionic_strength}\n"
              + f"    ele_cutoff_type: {self.ele_cutoff_type}\n"
              + f"    ele_cutoff_factor: {self.ele_cutoff_factor}\n"
              + f"    ele_no_charge: {self.ele_no_charge}\n"
              + f"    ele_length_per_charge: {self.ele_length_per_charge}\n"
              + f"    ele_exclusions: {self.ele_exclusions}\n"
              + f"    use_NNP: {self.use_NNP}\n"
              + f"    NNP_model: {self.NNP_model}\n"
              + f"    NNP_emblist: {self.NNP_emblist}\n"
                )

ctrl = Control()

if args.cuda:
    ctrl.device = 'CUDA'
else:
    ctrl.device = 'default'

################################################
#          Load input TOML
################################################
#tomldata = toml.load(sys.argv[1])
toml_input = None
if args.input is not None:
    import toml
    with open(args.input) as stream:
        try:
            toml_input = toml.load(stream)
        except:
            print ("Error: could not read the input TOML file.")
            raise

if toml_input is not None:
    if 'xml' in toml_input['Files']['In']:
        ctrl.xml = toml_input['Files']['In']['xml']
    if 'ff' in toml_input['Files']['In']:
        ctrl.ff = toml_input['Files']['In']['ff']
    ctrl.infile_pdb   = toml_input['Files']['In']['pdb_ini']
    ctrl.Nstep        = toml_input['MD']['nstep']
    ctrl.Nstep_save   = toml_input['MD']['nstep_save']
    ctrl.Nstep_out    = toml_input['Progress']['step']
    ctrl.outfile_dcd  = toml_input['Files']['Out']['prefix'] + '.dcd'
    ctrl.outfile_log  = toml_input['Files']['Out']['prefix'] + '.log'
    ctrl.outfile_out  = toml_input['Files']['Out']['prefix'] + '.out'
    ctrl.outfile_rst  = toml_input['Files']['Out']['prefix'] + '.rst'
    ctrl.temp         = toml_input['Condition']['tempK'] * unit.kelvin
    ctrl.LD_temp      = toml_input['Condition']['tempK'] * unit.kelvin
    ctrl.LD_gamma     = toml_input['MD']['friction'] / unit.picosecond
    ctrl.LD_dt        = toml_input['MD']['dt'] * unit.femtoseconds
    ctrl.ele          = False

    if 'Electrostatic' in toml_input.keys():
        ctrl.ele = True
        ctrl.ele_ionic_strength = toml_input['Electrostatic']['ionic_strength']
        ctrl.ele_cutoff_type    = toml_input['Electrostatic']['cutoff_type']
        ctrl.ele_cutoff_factor  = toml_input['Electrostatic']['cutoff']
        ctrl.ele_no_charge      = toml_input['Electrostatic']['no_charge']
        ctrl.ele_length_per_charge = toml_input['Electrostatic']['length_per_charge'] * unit.angstrom
        if toml_input['Electrostatic']['exclude_covalent_bond_pairs']:
            ctrl.ele_exclusions['1-2'] = True

    if 'NNP' in toml_input.keys():
        ctrl.use_NNP      = True
        ctrl.NNP_model    = toml_input['Files']['In']['TMnet_ckpt']
        #ctrl.NNP_emblist  = toml_input['external']['embeddings']

################################################
#          Load TorchMD input yaml
################################################

tmyaml_input = None
if args.tmyaml is not None:
    import yaml
    with open(args.tmyaml) as stream:
        try:
            tmyaml_input = yaml.safe_load(stream)
            #print(tmyml_input)
        except yaml.YAMLError as err:
            print (err)
            raise

if tmyaml_input is not None:
    ctrl.infile_pdb   = tmyaml_input['structure']
    ctrl.Nstep        = tmyaml_input['steps']
    ctrl.Nstep_save   = tmyaml_input['save_period']
    ctrl.Nstep_out    = tmyaml_input['output_period']
    ctrl.outfile_dcd  = tmyaml_input['output'] + '.dcd'
    ctrl.outfile_log  = tmyaml_input['output'] + '.log'
    ctrl.outfile_out  = tmyaml_input['output'] + '.out'
    ctrl.outfile_rst  = tmyaml_input['output'] + '.rst'
    ctrl.temp         = tmyaml_input['temperature'] * unit.kelvin
    ctrl.LD_temp      = tmyaml_input['langevin_temperature'] * unit.kelvin
    ctrl.LD_gamma     = tmyaml_input['langevin_gamma'] / unit.picosecond
    ctrl.LD_dt        = tmyaml_input['timestep'] * unit.femtoseconds
    ctrl.NNP_model    = tmyaml_input['external']['file']
    ctrl.NNP_emblist  = tmyaml_input['external']['embeddings']
    ctrl.use_NNP      = True


################################################
#          Arguments override
################################################
# Argument --xml overrides "xml" in the TOML input
if args.xml is not None:
    ctrl.xml = args.xml
# If neigher --xml or 'xml' in TOML exist, try to find the default xml file.
elif ctrl.xml is None:
    ctrl.xml = os.path.dirname(os.path.realpath(__file__)) + '/' + FILENAME_XML_DEFAULT

# Argument --ff overrides "ff" in the TOML input
if args.ff is not None:
    ctrl.ff = args.ff


print(ctrl)


################################################
#   Load topology and positions from the PDB
################################################
topology = None
positions = None

cgpdb = app.PDBFile(ctrl.infile_pdb)
topology = cgpdb.getTopology()
positions = cgpdb.getPositions()
#name_map = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'U': 'URA'}
#name_map = {'A': 'RA', 'C': 'RC', 'G': 'RG', 'U': 'RU', 'D': 'RD'}

for c in topology.chains():
    for prev, item, nxt in prev_and_next(c.residues()):

        #item.name = name_map[item.name]

        if prev is None or nxt is None:
            item.name += 'T'

        if prev is not None:
            topology.addBond(get_atom(prev), get_atom(item))

#topology.setPeriodicBoxVectors([[ctrl.box.value_in_unit(unit.nanometers),0,0], [0,ctrl.box.value_in_unit(unit.nanometers),0], [0,0,ctrl.box.value_in_unit(unit.nanometers)]])

################################################
#             Load force field
################################################
ff = SISForceField()

if ctrl.ff is not None:
    ff.read_toml(ctrl.ff)
else:
    print("WARNING: Force field (ff) file was not specified. Default values are used.")

print(ff)

if ctrl.ele:
    def set_ele(T, C, lp, cut_type, cut_factor):
        tab = "    "
        tab2 = tab + tab 
        # Input: T, Temperature
        #        C, Ionic strength
        #        lp, Length per charge
        # Output: cutoff, scale, kappa
        print(tab + "Debye-Huckel electrostatics:")
        print(tab2 + "Ionic strength: ", C, " M")
        print(tab2 + "Temperature: ", T)
        Tc = T/unit.kelvin - 273.15
        diele = 87.740 - 0.4008*Tc + 9.398e-4*Tc**2 - 1.410e-6*Tc**3
        print(tab2 + "Dielectric constant (T dependent): ", diele)
        ELEC = 1.602176634e-19   # Elementary charge [C]
        EPS0 = 8.8541878128e-12  # Electric constant [F/m]
        BOLTZ_J = 1.380649e-23   # Boltzmann constant [J/K]
        N_AVO = 6.02214076e23    # Avogadro constant [/mol]
        JOUL2KCAL_MOL = 1.0/4184.0 * N_AVO  # (J -> kcal/mol)
        lb = ELEC**2 / (4.0*pi*EPS0*diele*BOLTZ_J*T/unit.kelvin) * 1.0e10 * unit.angstrom
        print(tab2 + "Bjerrum length: ", lb)
        Zp = - lp / lb
        print(tab2 + "Reduced charge: ", Zp)
        lambdaD  = 1.0e10 * sqrt( (1.0e-3 * EPS0 * diele * BOLTZ_J)
                 / (2.0 * N_AVO * ELEC**2)  ) * sqrt(T/unit.kelvin / C) * unit.angstrom
        print(tab2 + "lambda_D:", lambdaD)

        if cut_type == 1:
            cutoff = cut_factor * unit.angstrom
        elif cut_type == 2:
            cutoff = cut_factor * lambdaD
        else:
            print("Error: Unknown cutoff_type for Electrostatic.")
            sys.exit(2)

        kappa = 1.0 / lambdaD
        scale = JOUL2KCAL_MOL * 1.0e10 * ELEC**2 / (4.0 * pi * EPS0 * diele) * Zp**2 * unit.kilocalorie_per_mole
        print(tab2 + "Cutoff: ", cutoff)
        print(tab2 + "Scale: ", scale)
        print(tab2 + "kappa: ", kappa)

        return cutoff, scale, kappa

    ele_cutoff, ele_scale, ele_kappa = set_ele(ctrl.temp,  # T
                                   ctrl.ele_ionic_strength,  # C
                                   ctrl.ele_length_per_charge, # lp 
                                   ctrl.ele_cutoff_type,
                                   ctrl.ele_cutoff_factor)
print('')

################################################
#          Check XML file
################################################
if not os.path.isfile(ctrl.xml):
    print("Error: could not find topology XML file, " + ctrl.xml)
    print("You can specify the correct path by either --xml or 'xml' in the TOML input")
    sys.exit(2)

################################################
#             Construct forces
################################################
print("Constructing forces:")
system = app.ForceField(ctrl.xml).createSystem(topology)

totalforcegroup = -1
groupnames = []

########## Bond
if ff.bond:
    bondforce = omm.HarmonicBondForce()
    for bond in topology.bonds():
        bondforce.addBond(bond[0].index, bond[1].index, ff.bond_r0, ff.bond_k)

    bondforce.setUsesPeriodicBoundaryConditions(False)
    totalforcegroup += 1
    bondforce.setForceGroup(totalforcegroup)
    print(f"    {totalforcegroup:2d}:    Bond")
    groupnames.append("Ubond")
    system.addForce(bondforce)

########## Angle
if ff.angle:
    angleforce = omm.HarmonicAngleForce()

    for chain in topology.chains():
        for prev, item, nxt in prev_and_next(chain.residues()):
            if prev == None or nxt == None:
                continue

            angleforce.addAngle(prev.index, item.index, nxt.index, ff.angle_a0, ff.angle_k)

    angleforce.setUsesPeriodicBoundaryConditions(False)
    totalforcegroup += 1
    angleforce.setForceGroup(totalforcegroup)
    print(f"    {totalforcegroup:2d}:    Angle")
    groupnames.append("Uangl")
    system.addForce(angleforce)

########## Restricted Bending (ReB)
if ff.angle_ReB:
    #ReB_energy_function = '0.5 * ReB_k * (cos(theta) - cos_ReB_a0)^2 / (sin(theta)^2)'
    ReB_energy_function = '0.5 * ReB_k * (cos(theta) - cos_ReB_a0)^2 / (1.0 - cos(theta)^2)'

    cos_0 = cos(ff.angle_a0 / unit.radian)
    ReBforce = omm.CustomAngleForce(ReB_energy_function)
    ReBforce.addGlobalParameter("ReB_k",      ff.angle_k)
    ReBforce.addGlobalParameter("cos_ReB_a0", cos_0)
    #ReBforce.addPerAngleParameter("ReB_k")
    #ReBforce.addPerAngleParameter("cos_ReB_a0")

    for chain in topology.chains():
        for prev, item, nxt in prev_and_next(chain.residues()):
            if prev == None or nxt == None:
                continue

            #ReBforce.addAngle(prev.index, item.index, nxt.index, [ReB_k, cos_ReB_a0])
            ReBforce.addAngle(prev.index, item.index, nxt.index)

    ReBforce.setUsesPeriodicBoundaryConditions(False)
    totalforcegroup += 1
    ReBforce.setForceGroup(totalforcegroup)
    print(f"    {totalforcegroup:2d}:    Angle ReB")
    groupnames.append("Uangl")
    system.addForce(ReBforce)

########## Dihedral (exponential)
if ff.dihexp:
    dihedral_energy_function = '-dihexp_k * exp(-0.5 * dihexp_w * (theta - dihexp_p0)^2)'

    dihedralforce = omm.CustomTorsionForce(dihedral_energy_function)
    dihedralforce.addGlobalParameter('dihexp_k',  ff.dihexp_k)
    dihedralforce.addGlobalParameter('dihexp_w',  ff.dihexp_w)
    dihedralforce.addGlobalParameter('dihexp_p0', ff.dihexp_p0)
    #dihedralforce.addPerTorsionParameter("dihexp_k")
    #dihedralforce.addPerTorsionParameter("dihexp_w")
    #dihedralforce.addPerTorsionParameter("dihexp_p0")

    for chain in topology.chains():
        for a, b, c, d in fours(chain.residues()):
            dihedralforce.addTorsion(a.index, b.index, c.index, d.index)
            #dihedralforce.addTorsion(a.index, b.index, c.index, d.index,
            #                           [dihexp_k, dihexp_w, dihexp_p0])

    totalforcegroup += 1
    dihedralforce.setForceGroup(totalforcegroup)
    print(f"    {totalforcegroup:2d}:    Dihexp")
    groupnames.append("Udih")
    system.addForce(dihedralforce)

########## WCA
if ff.wca:
    energy_function =  'step(sig-r) * ep * ((R6 - 2)*R6 + 1);'
    energy_function += 'R6=(sig/r)^6;'

    WCAforce = omm.CustomNonbondedForce(energy_function)
    WCAforce.addGlobalParameter('ep',  ff.wca_epsilon)
    WCAforce.addGlobalParameter('sig', ff.wca_sigma)

    for atom in topology.atoms():
        WCAforce.addParticle([])

    # Exclusion for bond
    if ff.wca_exclusions['1-2']:
        for bond in topology.bonds():
            WCAforce.addExclusion(bond[0].index, bond[1].index)

    # Exclusion for angle
    if ff.wca_exclusions['1-3']:
        for chain in topology.chains():
            for prev, item, nxt in prev_and_next(chain.residues()):
                if prev == None or nxt == None:
                    continue
                WCAforce.addExclusion(atm_index(prev), atm_index(nxt))

    WCAforce.setCutoffDistance(ff.wca_sigma)
    totalforcegroup += 1
    WCAforce.setForceGroup(totalforcegroup)
    print(f"    {totalforcegroup:2d}:    WCA")
    groupnames.append("Uwca")
    WCAforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffNonPeriodic)
    system.addForce(WCAforce)

########## Debye-Huckel
if ctrl.ele:
    DHforce = omm.CustomNonbondedForce("scale*exp(-kappa*r)/r")
    DHforce.addGlobalParameter("scale", ele_scale)
    DHforce.addGlobalParameter("kappa", ele_kappa)

    for atom in topology.atoms():
        DHforce.addParticle([])

    if ctrl.ele_exclusions['1-2']:
        for bond in topology.bonds():
            DHforce.addExclusion(bond[0].index, bond[1].index)

    #if ctrl.ele_exclusions['1-3']:
    if True:
        for chain in topology.chains():
            for prev, item, nxt in prev_and_next(chain.residues()):
                if prev == None or nxt == None:
                    continue
                DHforce.addExclusion(atm_index(prev), atm_index(nxt))

    DHforce.setCutoffDistance(ele_cutoff)
    totalforcegroup += 1
    DHforce.setForceGroup(totalforcegroup)
    print(f"    {totalforcegroup:2d}:    Ele")
    groupnames.append("Uele")
    DHforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffNonPeriodic)
    system.addForce(DHforce)

########## NNP
if ctrl.use_NNP:
    import torch
    from openmmtorch import TorchForce
    from torchmdnet.models.model import load_model

    #torch.set_default_dtype(torch.float32)
    #torch.set_default_dtype(torch.float64)

    #model = load_model(ctrl.NNP_model, derivative=False)
    #torch.jit.script(model).save('model.pt')

    class ForceModule(torch.nn.Module):
        def __init__(self, epochfile, embeddings):
            super(ForceModule, self).__init__()

            #self.model = torch.jit.load('model.pt')
            #self.model = torch.jit.script(load_model(epochfile, derivative=True))
            self.model = torch.jit.script(load_model(epochfile, derivative=False))
            '''derivative=False : Let OpenMM calculate force by back-propagation.'''

            self.model.eval()
            # Worked
            if ctrl.device == 'CUDA':
                self.embeddings = embeddings.cuda()
            else:
                self.embeddings = embeddings
            #self.batch = torch.arange(1).repeat_interleave(embeddings.size(0)).cuda()
            #self.embeddings = embeddings
            #self.batch = torch.arange(1).repeat_interleave(embeddings.size(0))
            self.batch = None
            self.box = None

        def forward(self, positions):
            positions_A = positions*10  # nm -> angstrom
            energy, _ = self.model(self.embeddings, positions_A, self.batch, self.box)
            #energy, force = self.model(self.embeddings, positions_A, self.batch, self.box)
            #print('energy=', energy)
            #print('force=', force)
            #return energy*4.814, force*41.84
            return energy*4.814

    embeddings = torch.tensor(ctrl.NNP_emblist, dtype=torch.long)

    module = torch.jit.script(ForceModule(ctrl.NNP_model, embeddings))
    torch_force = TorchForce(module)
    totalforcegroup += 1
    torch_force.setForceGroup(totalforcegroup)
    print(f"    {totalforcegroup:2d}:    NNP")
    groupnames.append("Unn")
    system.addForce(torch_force)

print('')

################################################
#             Simulation set up
################################################

class EnergyReporter(object):
    def __init__ (self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._out.write('#     1:Step    2:T        3:Ekin        4:Epot')
                        #123456789012 123456 1234567890123 1234567890123'
        icol = 4
        if ff.bond:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Ebond')
        if ff.angle:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Eangl')
        if ff.angle_ReB:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Eangl')
        if ff.dihexp:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Edih')
        if ff.wca:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Ewca')
        if ctrl.ele:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Eele')
        if ctrl.use_NNP:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Enn')

        self._out.write("\n")
        self._out.flush()

    def __del__ (self):
        self._out.close()

    def describeNextReport(self, simulation):
        step = self._reportInterval - simulation.currentStep%self._reportInterval
        return (step, False, False, False, True)
        #return (step, position, velocity, force, energy)

    def report(self, simulation, state):
        energy = []
        self._out.write(f"{simulation.currentStep:12d}")
        self._out.write(f" {ctrl.LD_temp/unit.kelvin:6.2f}")
        state = simulation.context.getState(getEnergy=True)
        energy = state.getKineticEnergy() / unit.kilocalorie_per_mole
        self._out.write(f" {energy:13.6g}")
        energy = state.getPotentialEnergy() / unit.kilocalorie_per_mole
        self._out.write(f" {energy:13.6g}")
        for i in range(totalforcegroup + 1):
            state = simulation.context.getState(getEnergy=True, groups=2**i)
            energy = state.getPotentialEnergy() / unit.kilocalorie_per_mole
            self._out.write(f" {energy:13.6g}")
        self._out.write("\n")
        self._out.flush()

#integrator = omm.LangevinIntegrator(ctrl.LD_temp, ctrl.LD_gamma, ctrl.LD_dt)
integrator = omm.LangevinMiddleIntegrator(ctrl.LD_temp, ctrl.LD_gamma, ctrl.LD_dt)

platform = None
properties = None

if ctrl.device == 'CUDA':
    platform = omm.Platform.getPlatformByName('CUDA')
    #properties = {'Precision': 'double'}
    properties = {'Precision': 'single'}

#if ctrl.device == 'CPU':
#    platform = omm.Platform.getPlatformByName('CPU')
#
#
#else:
#    print("Error: unknown device.")
#    sys.exit(2)

#properties = {}
#properties["DeviceIndex"] = "0"
#if 'GPU' in tomldata:
#    if 'platform' in tomldata['GPU']:
#        platform = omm.Platform.getPlatformByName(tomldata.GPU.platform)
#        print("Platform ", tomldata.GPU.platform)
#
#    if 'CUDAdevice' in tomldata['GPU']:
#        properties = {} 
#        properties["DeviceIndex"] = tomldata.GPU.CUDAdevice
#        print("DeviceIndex ", tomldata.GPU.CUDAdevice)

#simulation = app.Simulation(topology, system, integrator, platform)
#simulation = app.Simulation(topology, system, integrator)
simulation = app.Simulation(topology, system, integrator, platform, properties)

if ctrl.restart == False:

    simulation.context.setPositions(positions)

    #boxvector = diag([ctrl.box/unit.angstrom for i in range(3)]) * unit.angstrom
    #simulation.context.setPeriodicBoxVectors(*boxvector)
    #print(simulation.usesPeriodicBoundaryConditions())

    ## Write PDB before minimization
    #state = simulation.context.getState(getPositions=True)
    #app.PDBFile.writeFile(topology, state.getPositions(), open("before_minimize.pdb", "w"), keepIds=True)

    #print('Minimizing ...')
    #simulation.minimizeEnergy(1*unit.kilocalorie_per_mole, 10000)
    #simulation.minimizeEnergy()

    ## Write PDB after minimization
    #state = simulation.context.getState(getPositions=True)
    #app.PDBFile.writeFile(topology, state.getPositions(), open("after_minimize.pdb", "w"), keepIds=True)

    #simulation.context.setVelocitiesToTemperature(ctrl.temp)
    ## This does not work (https://github.com/openmm/openmm-torch/issues/61)

else:
    print("Loading checkpoint ...")
    simulation.loadCheckpoint(ctrl.restart_file)

simulation.reporters.append(app.StateDataReporter(ctrl.outfile_log, ctrl.Nstep_out, 
                            step=True, potentialEnergy=True, temperature=True, 
                            remainingTime=True, totalSteps=ctrl.Nstep, separator='  '))
simulation.reporters.append(EnergyReporter(ctrl.outfile_out, ctrl.Nstep_save))
simulation.reporters.append(app.DCDReporter(ctrl.outfile_dcd, ctrl.Nstep_save))
simulation.reporters.append(app.CheckpointReporter(ctrl.outfile_rst, 1000000))

print('Simulation starting ...')
sys.stdout.flush()
sys.stderr.flush()

t0 = time.time()

simulation.step(ctrl.Nstep)

#simulation.saveState('checkpoint.xml')
prodtime = time.time() - t0
print("Simulation speed: % .2e steps/day" % (86400*ctrl.Nstep/(prodtime)))
