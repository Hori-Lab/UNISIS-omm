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
import toml
import yaml
import itertools as it
import argparse
from dataclasses import dataclass, field
from typing import List
from math import sqrt #, acos, atan2
from simtk import unit
from simtk.unit import Quantity
from numpy import diag

from openmm import app
import openmm as omm

from sis_params import SISForceField


################################################
#         Utility functions
################################################

def prev_and_next(iterable):
    prevs, items, nexts = it.tee(iterable, 3)
    prevs = it.chain([None], prevs)
    nexts = it.chain(it.islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)

def prev_and_next_and_after(iterable):
    prevs, items, nexts, afters = it.tee(iterable, 4)
    prevs = it.chain([None], prevs)
    nexts = it.chain(it.islice(nexts, 1, None), [None])
    afters = it.chain(it.islice(afters, 2, None), [None, None])
    return zip(prevs, items, nexts, afters)

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
xml_default = os.path.dirname(os.path.realpath(__file__)) + '/rna_cg2.xml'


################################################
#          Parser
################################################

parser = argparse.ArgumentParser(description='OpenMM script for the SIS-RNA model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser_ctrl = parser.add_mutually_exclusive_group(required=True)
parser_ctrl.add_argument('--input', type=str, help='TOML format input file')
parser_ctrl.add_argument('--tmyaml', type=str, help='TorchMD format YAML file')

parser.add_argument('--ff', type=str, help='TOML format force-field file')
parser.add_argument('--xml', type=str, default=xml_default, help='XML file for topology information')

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
from datetime import datetime
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
    xml: str = ''
    restart: bool = False
    restart_file: str = None
    minimization: bool = False

    Nstep: int        = 1000000
    Nstep_save: int   = 1000
    Nstep_log:  int   = 1000

    infile_pdb: str   = './T2HP_unfolded.pdb'
    outfile_log: str  = './md.log'
    outfile_out: str  = './md.out'
    outfile_dcd: str  = './md.dcd'
    outfile_rst: str  = './md.rst'

    temp: Quantity        = 300.0 * unit.kelvin
    LD_temp: Quantity     = 300.0 * unit.kelvin
    LD_gamma: Quantity    = 0.5 / unit.picosecond
    LD_dt: Quantity       = 50 * unit.femtoseconds

    use_NNP: bool = False
    NNP_model: str = ''
    NNP_emblist: List = field(default_factory=lambda: 
                        [5,2,3,4,1,4,2,1,2,2,4,3,1,4,1,3,1,4,3,2,4,3,1,4,1,2,3,1,5])

    #box = 0.
    #Kconc: float = -1.
    #b: float = 4.38178046 * unit.angstrom / unit.elementary_charge
    #cutoff: float = 0.
    #epsilon: float = 0.

    def __str__(self):
        return (f"Control:\n"
              + f"    device: {self.device}\n"
              + f"    xml: {self.xml}\n"
              + f"    restart: {self.restart}\n"
              + f"    restart_file: {self.restart_file}\n"
              + f"    minimization: {self.minimization}\n"
              + f"    Nstep: {self.Nstep}\n"
              + f"    Nstep_save: {self.Nstep_save}\n"
              + f"    Nstep_log {self.Nstep_log}\n"
              + f"    infile_pdb: {self.infile_pdb}\n"
              + f"    outfile_log: {self.outfile_log}\n"
              + f"    outfile_out: {self.outfile_out}\n"
              + f"    outfile_dcd: {self.outfile_dcd}\n"
              + f"    outfile_rst: {self.outfile_rst}\n"
              + f"    temp: {self.temp}\n"
              + f"    LD_temp: {self.LD_temp}\n"
              + f"    LD_gamma: {self.LD_gamma}\n"
              + f"    LD_dt: {self.LD_dt}\n"
              + f"    use_NNP: {self.use_NNP}\n"
              + f"    NNP_model: {self.NNP_model}\n"
              + f"    NNP_emblist: {self.NNP_emblist}\n"
                )

ctrl = Control()

if not os.path.isfile(args.xml):
    print("Error: could not find topology XML file, " + args.xml)
    sys.exit(2)
ctrl.xml = args.xml

if args.cuda:
    ctrl.device = 'CUDA'
else:
    ctrl.device = 'default'

#tomldata = toml.load(sys.argv[1])

################################################
#          Load TorchMD input yaml
################################################

tmyaml_input = None
if args.tmyaml is not None:
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

if args.ff is not None:
    ff.read_toml(args.ff)

print(ff)

################################################
#             Construct forces
################################################
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

########## Dihedral (exponential)
if ff.dihexp:
    dihedral_energy_function = '-dihexp_k * exp(-0.5 * dihexp_w * (theta - dihexp_p0)^2)'

    dihedralforce = omm.CustomTorsionForce(dihedral_energy_function)
    dihedralforce.addGlobalParameter('dihexp_k',  ff.dihexp_k)
    dihedralforce.addGlobalParameter('dihexp_w',  ff.dihexp_w)
    dihedralforce.addGlobalParameter('dihexp_p0', ff.dihexp_p0)
    #dihedralforce.addPerTorsionParameter("dihexp_k");
    #dihedralforce.addPerTorsionParameter("dihexp_w");
    #dihedralforce.addPerTorsionParameter("dihexp_p0");

    for chain in topology.chains():
        for prev, item, nxt, aft in prev_and_next_and_after(chain.residues()):
            if prev == None or aft == None:
                continue
            dihedralforce.addTorsion(prev.index, item.index, nxt.index, aft.index)
            #dihedralforce.addTorsion(prev.index, item.index, nxt.index, aft.index, 
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

        if ff.dihexp:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Edih')

        if ff.wca:
            icol += 1
            self._out.write(' %13s' % f'{icol}:Ewca')

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

print('Running ...')
sys.stdout.flush()
sys.stderr.flush()

t0 = time.time()

simulation.step(ctrl.Nstep)

#simulation.saveState('checkpoint.xml')
prodtime = time.time() - t0
print("Simulation speed: % .2e steps/day" % (86400*ctrl.Nstep/(prodtime)))
