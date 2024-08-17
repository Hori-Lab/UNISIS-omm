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

import sys
import time
import re
import toml
import yaml
import itertools as it
from math import sqrt #, acos, atan2
from simtk import unit
from numpy import diag

import torch
from openmm import app
import openmm as omm
from openmmtorch import TorchForce
from torchmdnet.models.model import load_model

from sis_params import SISForceField


###################################################################
# Utility functions
def prev_and_next(iterable):
    prevs, items, nexts = it.tee(iterable, 3)
    prevs = it.chain([None], prevs)
    nexts = it.chain(it.islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)

def atm_index(res):
    #return res.atoms[0].index
    for atom in res.atoms():
        return atom.index

def get_atom(res):
    for atom in res.atoms():
        return atom

###################################################################
# Constants
KELVIN_TO_KT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB / unit.kilocalorie_per_mole
#print KELVIN_TO_KT

###################################################################

if len(sys.argv) != 2:
    print('SCRIPT [yaml input]')
    sys.exit(2)

tmyaml_input = None
tmyaml_file = sys.argv[1]
with open(tmyaml_file) as stream:
    try:
        tmyaml_input = yaml.safe_load(stream)
        #print(tmyml_input)
    except yaml.YAMLError as err:
        print (err)
        raise


class Control:    ### structure to group all simulation parameter
    #box = 0.
    #Kconc: float = -1.
    #b: float = 4.38178046 * unit.angstrom / unit.elementary_charge
    #cutoff: float = 0.
    #epsilon: float = 0.
    restart: bool = False
    restart_file: str = None
    minimization: bool = False

    Nstep       = 1000000
    Nstep_save  = 1000
    Nstep_out   = 1000

    outfile_dcd = './md.dcd'
    outfile_out = './md.out'
    outfile_ene = './md.ene'
    outfile_rst = './md.rst'

    temp        = 300.0 * unit.kelvin
    LD_temp     = 300.0 * unit.kelvin
    LD_gamma    = 0.5 / unit.picosecond
    LD_dt       = 50 * unit.femtoseconds

ctrl = Control()
#if len(sys.argv) == 2:
#    ctrl.restart = False
#elif len(sys.argv) == 3:
#    ctrl.restart = True
#    ctrl.restart_file = sys.argv[2]
#else:
#    print('Usage: SCRIPT (input toml) [restart file]')
#    sys.exit(2)

# Output the date
from datetime import datetime
print('Executed: ' + str(datetime.now()) + ' (UTC: ' + str(datetime.utcnow()) + ')')

# Output Git hash
try:
    import os
    import subprocess
    src_path = os.path.dirname(os.path.realpath(__file__))
    h = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=src_path, stderr=subprocess.DEVNULL).decode('ascii').strip()
    print('Git hash: ' + h)
except:
    pass

print('')


#tomldata = toml.load(sys.argv[1])

if tmyaml_input is not None:
    ctrl.Nstep       = tmyaml_input['steps']
    ctrl.Nstep_save  = tmyaml_input['save_period']
    ctrl.Nstep_out   = tmyaml_input['output_period']
    ctrl.outfile_dcd = tmyaml_input['output'] + '.dcd'
    ctrl.outfile_out = tmyaml_input['output'] + '.out'
    ctrl.outfile_ene = tmyaml_input['output'] + '.ene'
    ctrl.outfile_rst = tmyaml_input['output'] + '.rst'
    ctrl.temp        = tmyaml_input['temperature'] * unit.kelvin
    ctrl.LD_temp     = tmyaml_input['langevin_temperature'] * unit.kelvin
    ctrl.LD_gamma    = tmyaml_input['langevin_gamma'] / unit.picosecond
    ctrl.LD_dt       = tmyaml_input['timestep'] * unit.femtoseconds
    epochfile        = tmyaml_input['external']['file']

    #embeddings       = tmyaml_input['external']['embeddings']
    embeddings = torch.tensor(tmyaml_input['external']['embeddings'])


else:
    embeddings = torch.tensor([5, 2, 3, 4, 1, 4, 2, 1, 2, 2, 4, 3, 1, 4, 1, 3, 1, 4, 3, 2, 4, 3, 1, 4, 1, 2, 3, 1, 5])
    epochfile = 'epoch=295-val_loss=0.1475-test_loss=0.2365.ckpt'

ctrl.box = 0.

forcefield = app.ForceField('rna_cg2.xml')
topology = None
positions = None

seq = 'DGCUAUGAGGUCAUACAUCGUCAUAGCAD'
cgpdb = app.PDBFile('T2HP_unfolded.pdb')

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

ff = SISForceField()

system = forcefield.createSystem(topology)

totalforcegroup = -1
groupnames = []

########## bond force
bondforce = omm.HarmonicBondForce()
for bond in topology.bonds():
    bondforce.addBond(bond[0].index, bond[1].index, ff.bond_r0*unit.angstroms, ff.bond_k*unit.kilocalorie_per_mole/(unit.angstrom**2))

bondforce.setUsesPeriodicBoundaryConditions(False)
totalforcegroup += 1
bondforce.setForceGroup(totalforcegroup)
print("Force group bond: ", totalforcegroup)
groupnames.append("Ubond")
system.addForce(bondforce)

######## WCA force
WCA_cutoff = ff.wca_sigma*unit.angstroms
energy_function =  'step(sig-r) * ep * ((R6 - 2)*R6 + 1);'
energy_function += 'R6=(sig/r)^6;'

WCAforce = omm.CustomNonbondedForce(energy_function)
WCAforce.addGlobalParameter('ep',  ff.wca_epsilon*unit.kilocalorie_per_mole)
WCAforce.addGlobalParameter('sig', WCA_cutoff)

for atom in topology.atoms():
    WCAforce.addParticle([])

for bond in topology.bonds():
    WCAforce.addExclusion(bond[0].index, bond[1].index)

for chain in topology.chains():
    for prev, item, nxt in prev_and_next(chain.residues()):
        if prev == None or nxt == None:
            continue
        WCAforce.addExclusion(atm_index(prev), atm_index(nxt))

WCAforce.setCutoffDistance(WCA_cutoff)
totalforcegroup += 1
WCAforce.setForceGroup(totalforcegroup)
print("Force group WCA: ", totalforcegroup)
groupnames.append("Uwca")
WCAforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffNonPeriodic)
system.addForce(WCAforce)

#print ([atom.element.atomic_number for atom in topology.atoms()])
#sys.exit(1)

####### ML force
class ForceModule(torch.nn.Module):
    def __init__(self, epochfile, embeddings):
        super(ForceModule, self).__init__()

        #self.model = torch.jit.script(load_model(epochfile, derivative=True))
        self.model = torch.jit.script(load_model(epochfile, derivative=False))
        # derivative=False : Let OpenMM calculate force by back-propagation

        self.model.eval()
        self.embeddings = embeddings
        self.batch = torch.arange(1).repeat_interleave(embeddings.size(0))
        self.box = None

    def forward(self, positions):
        positions_A = positions*10
        energy, _ = self.model(self.embeddings, positions_A, self.batch, self.box)
        #energy, force = self.model(self.embeddings, positions_A, self.batch, self.box)
        #print('energy=', energy)
        #print('force=', force)
        #return energy*4.814, force*41.84
        return energy*4.814


#infile_pt = 'epoch=295-val_loss=0.1475-test_loss=0.2365.ckpt.pt'
module = torch.jit.script(ForceModule(epochfile, embeddings))
torch_force = TorchForce(module)
totalforcegroup += 1
torch_force.setForceGroup(totalforcegroup)
print("Force group NNP: ", totalforcegroup)
groupnames.append("Unn")
system.addForce(torch_force)

########## Simulation ############
class EnergyReporter(object):
    def __init__ (self, file, reportInterval):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self._out.write('#   Steps   ')
        self._out.write('  Ukinetic   ')
        self._out.write('  Upotential ')
        for gn in groupnames:
            self._out.write(f' {gn:^12s}')
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
        state = simulation.context.getState(getEnergy=True)
        energy = state.getKineticEnergy() / unit.kilocalorie_per_mole
        self._out.write(f" {energy:12.6g}")
        energy = state.getPotentialEnergy() / unit.kilocalorie_per_mole
        self._out.write(f" {energy:12.6g}")
        for i in range(totalforcegroup + 1):
            state = simulation.context.getState(getEnergy=True, groups=2**i)
            energy = state.getPotentialEnergy() / unit.kilocalorie_per_mole
            self._out.write(f" {energy:12.6g}")
        self._out.write("\n")
        self._out.flush()

#integrator = omm.LangevinIntegrator(ctrl.LD_temp, ctrl.LD_gamma, ctrl.LD_dt)
integrator = omm.LangevinMiddleIntegrator(ctrl.LD_temp, ctrl.LD_gamma, ctrl.LD_dt)
#platform = omm.Platform.getPlatformByName('CUDA')
#platform = omm.Platform.getPlatformByName('CPU')
#properties = {'CudaPrecision': 'mixed'}

platform = None
properties = None
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
simulation = app.Simulation(topology, system, integrator, platform, properties)
#simulation = app.Simulation(topology, system, integrator)

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

simulation.reporters.append(app.StateDataReporter(ctrl.outfile_out, ctrl.Nstep_out, 
                            step=True, potentialEnergy=True, temperature=True, 
                            remainingTime=True, totalSteps=ctrl.Nstep, separator='  '))
simulation.reporters.append(EnergyReporter(ctrl.outfile_ene, ctrl.Nstep_save))
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
