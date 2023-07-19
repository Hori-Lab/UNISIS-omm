#!/usr/bin/env python

from openmm import app
import openmm as omm
from simtk import unit
import itertools as it
import time
import sys
from math import sqrt, cos
import argparse

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



parser = argparse.ArgumentParser(description='Simple polymer simulation using OpenMM',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser_init = parser.add_mutually_exclusive_group(required=True)
parser_init.add_argument('-i','--inixyz', type=str, default=None, help='initial xyz file')
parser_init.add_argument('-N','--nbead', type=int, default=0, help='number of beads')

parser.add_argument('-n','--step', type=int, default='10000',
                    help='Number of step [10000]')

parser.add_argument('--dt', type=float, default='50.0',
                    help='integration time step in fs [50.0]')

parser.add_argument('-T','--temperature', type=float, default='300.0',
                    help='Temperature (K) [300.0]')

parser.add_argument('-a','--angle', action='store_true',
                    help='Enable the angle potential')

parser.add_argument('-b','--ReB', action='store_true',
                    help='Enable the Restricted Bending potential')

parser.add_argument('-d','--dihexp', action='store_true',
                    help='Enable the dihedral potential (exponential form)')

parser.add_argument('-I','--ionic_strength', type=float, default='-1.',
                    help='Ionic strength (M) [Default: no electrostatic (-1)]')
parser.add_argument('-c','--cutoff', type=float, default='30.',
                    help='Electrostatic cutoff (A) [30.0]')

parser.add_argument('-t','--traj', type=str, default='saw.dcd',
                    help='trajectory output')
parser.add_argument('-e','--energy', type=str, default='saw.ene',
                    help='energy decomposition')
parser.add_argument('-o','--output', type=str, default='saw.out',
                    help='status and energy output')
parser.add_argument('-x','--frequency', type=int, default='100',
                    help='output and restart frequency')
parser.add_argument('-f','--finalxyz', type=str, default='final.xyz',
                    help='final structure xyz output')

parser.add_argument('-R','--restart', action='store_true',
                    help='flag to restart simulation')
parser.add_argument('-r','--res_file', type=str, default='saw.chk',
                    help='checkpoint file for restart')

#parser.add_argument('--platform', type=str, default=None,
#                    help='Platform')
#parser.add_argument('--CUDAdevice', type=str, default=None,
#                    help='CUDA device ID')


args = parser.parse_args()

if args.inixyz is None and args.nbead <= 0:
    print('Error: either inixyz or nbead > 0 is needed')
    sys.exit(2)


class simu:    ### structure to group all simulation parameter
    temp = args.temperature * unit.kelvin
    Kconc = args.ionic_strength
    Nstep = args.step
    dt = args.dt
    cutoff = args.cutoff
    epsilon = 0.
    b = 4.38178046 * unit.angstrom / unit.elementary_charge
    restart = args.restart
    restart_file = args.res_file
    xyz_file = args.inixyz
    nbead = args.nbead
    templist = []
    steplist = []

flg_angle = args.angle
flg_ReB = args.ReB
flg_dihexp = args.dihexp

# Output the date
from datetime import datetime
print(str(datetime.now()) + ' (UTC: ' + str(datetime.utcnow()) + ')')

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


#######################
# Parameters setup
#######################

print ('Parameters set up')

print ('Backbone parameters:')
bond_k  = 1.5 * unit.kilocalorie_per_mole/(unit.angstrom**2)
bond_r0 = 5.84 * unit.angstrom
print ('    bond_k = ', bond_k)
print ('    bond_r0 = ', bond_r0)

angle_k  = 10.0 * unit.kilocalorie_per_mole/(unit.radian**2)
angle_a0 = 2.643 * unit.radian
print ('    angle_k = ', angle_k)
print ('    angle_a0 = ', angle_a0)
print ('')

ReB_k  = 5.0 * unit.kilocalorie_per_mole
ReB_a0 = 2.643
cos_ReB_a0 = cos(ReB_a0)
print ('    ReB_k = ', ReB_k)
print ('    ReB_a0 = ', ReB_a0 * unit.radian)
print ('')

dihexp_k = 1.4 * unit.kilocalorie_per_mole
dihexp_w = 3.0 /(unit.radian**2)
dihexp_p0 = 0.28 * unit.radian
print ('    dihexp_k = ', dihexp_k)
print ('    dihexp_w = ', dihexp_w)
print ('    dihexp_p0 = ', dihexp_p0)
print ('')

print ('Excluded volume parameters:')
wca_epsilon = 2.0 * unit.kilocalorie_per_mole
wca_sigma   = 10.0 * unit.angstrom
wca_cutoff  = wca_sigma
print ('    wca_epsilon = ', wca_epsilon)
print ('    wca_sigma = ', wca_sigma)
print ('')


## Debye-Huckel is enabled if simu.Kconc is given (>= 0).
if simu.Kconc >= 0.:
    T_unitless = simu.temp * unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB / unit.kilocalorie_per_mole
    print("Electrostatic parameters:")
    print("    [K] ", simu.Kconc, " mM")
    simu.epsilon = 296.0736276 - 619.2813716 * T_unitless + 531.2826741 * T_unitless**2 - 180.0369914 * T_unitless**3;
    print("    Dielectric constant ", simu.epsilon)
    #simu.l_Bjerrum = 1./(simu.epsilon * unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB * simu.temp)
    simu.l_Bjerrum = 332.0637*unit.angstroms / simu.epsilon
    print("    Bjerrum length  ", simu.l_Bjerrum / T_unitless)
    simu.Q = simu.b * T_unitless * unit.elementary_charge**2 / simu.l_Bjerrum
    print("    Phosphate charge   ", -simu.Q)
    simu.kappa = unit.sqrt (4*3.14159 * simu.l_Bjerrum * 2*simu.Kconc*6.022e-7 / (T_unitless * unit.angstrom**3))
    print("    kappa   ", simu.kappa)
    print("    Debye length ", 1.0/simu.kappa)
    print("")

forcefield = app.ForceField('rna_cg1.xml')
topology = None
positions = None

# Read xyz file
name_map = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'U': 'URA'}

seq = []
positions = []
topology = app.Topology()
chain = topology.addChain()

if simu.xyz_file is not None:
    N = 0

    for il, l in enumerate(open(simu.xyz_file)):
        if il > 1:
            lsp = l.split()
            positions.append([float(lsp[1])*unit.angstrom, float(lsp[2])*unit.angstrom, float(lsp[3])*unit.angstrom])

            seq.append(lsp[0])
            symbol = name_map[lsp[0]]
            if len(seq) == 1 or len(seq) == N:
                symbol = symbol + "T"

            res = topology.addResidue(symbol, chain)
            atom = forcefield._templates[symbol].atoms[0]
            topology.addAtom(atom.name, forcefield._atomTypes[atom.type].element, res)

        if il == 0:
            N = int(l)

    if len(seq) != N:
        print('Error: len(seq) != N in reading xyz')
        sys.exit(2)

    simu.nbead = N

else:
    for i in range(simu.nbead):
        positions.append([i*bond_r0, 0.0*unit.angstrom, 0.0*unit.angstrom])
        seq.append('C')
        symbol = name_map['C']
        if len(seq) == 1 or len(seq) == simu.nbead:
            symbol = symbol + "T"

        res = topology.addResidue(symbol, chain)
        atom = forcefield._templates[symbol].atoms[0]
        topology.addAtom(atom.name, forcefield._atomTypes[atom.type].element, res)

for prev, item, nxt in prev_and_next(chain.residues()):
    if prev != None:
        topology.addBond(get_atom(prev), get_atom(item))

system = forcefield.createSystem(topology)

totalforcegroup = -1
groupnames = []

########## bond force
bondforce = omm.HarmonicBondForce()
for bond in topology.bonds():
    bondforce.addBond(bond[0].index, bond[1].index, bond_r0, bond_k)

bondforce.setUsesPeriodicBoundaryConditions(False)
totalforcegroup += 1
bondforce.setForceGroup(totalforcegroup)
print("Force group bond: ", totalforcegroup)
groupnames.append("Ubond")
system.addForce(bondforce)

########## angle force
if flg_angle:
    angleforce = omm.HarmonicAngleForce()

    for chain in topology.chains():
        for prev, item, nxt in prev_and_next(chain.residues()):
            if prev == None or nxt == None:
                continue

            angleforce.addAngle(prev.index, item.index, nxt.index, angle_a0, angle_k)

    angleforce.setUsesPeriodicBoundaryConditions(False)
    totalforcegroup += 1
    angleforce.setForceGroup(totalforcegroup)
    print("Force group angle: ", totalforcegroup)
    groupnames.append("Uangl")
    system.addForce(angleforce)

########## Restricted Bending (ReB) force
if flg_ReB:

    ReB_energy_function = '0.5 * ReB_k * (cos(theta) - cos_ReB_a0)^2 / (sin(theta)^2)'

    ReBforce = omm.CustomAngleForce(ReB_energy_function)
    ReBforce.addPerAngleParameter("ReB_k");
    ReBforce.addPerAngleParameter("cos_ReB_a0");

    for chain in topology.chains():
        for prev, item, nxt in prev_and_next(chain.residues()):
            if prev == None or nxt == None:
                continue

            ReBforce.addAngle(prev.index, item.index, nxt.index, [ReB_k, cos_ReB_a0])

    ReBforce.setUsesPeriodicBoundaryConditions(False)
    totalforcegroup += 1
    ReBforce.setForceGroup(totalforcegroup)
    print("Force group ReB: ", totalforcegroup)
    groupnames.append("Uangl")
    system.addForce(ReBforce)

########### dihedral (exp) force
if flg_dihexp:
    dihedral_energy_function = '-dihexp_k * exp(-0.5 * dihexp_w * (theta - dihexp_p0)^2)'

    dihedralforce = omm.CustomTorsionForce(dihedral_energy_function)
    dihedralforce.addPerTorsionParameter("dihexp_k");
    dihedralforce.addPerTorsionParameter("dihexp_w");
    dihedralforce.addPerTorsionParameter("dihexp_p0");

    for chain in topology.chains():
        for prev, item, nxt, aft in prev_and_next_and_after(chain.residues()):
            if prev == None or aft == None:
                continue

            dihedralforce.addTorsion(prev.index, item.index, nxt.index, aft.index, [dihexp_k, dihexp_w, dihexp_p0])

    totalforcegroup += 1
    dihedralforce.setForceGroup(totalforcegroup)
    print("Force group dihedral: ", totalforcegroup)
    groupnames.append("Udih")
    system.addForce(dihedralforce)


######## WCA force
energy_function =  'step(sig-r) * ep * ((R6 - 2)*R6 + 1);'
energy_function += 'R6=(sig/r)^6;'

WCAforce = omm.CustomNonbondedForce(energy_function)
WCAforce.addGlobalParameter('ep',  wca_epsilon)
WCAforce.addGlobalParameter('sig', wca_sigma)

for atom in topology.atoms():
    WCAforce.addParticle([])

for bond in topology.bonds():
    WCAforce.addExclusion(bond[0].index, bond[1].index)

if flg_angle or flg_ReB:
   for chain in topology.chains():
       for prev, item, nxt in prev_and_next(chain.residues()):
           if prev == None or nxt == None:
               continue
           WCAforce.addExclusion(atm_index(prev), atm_index(nxt))

WCAforce.setCutoffDistance(wca_cutoff)
totalforcegroup += 1
WCAforce.setForceGroup(totalforcegroup)
print("Force group WCA: ", totalforcegroup)
groupnames.append("Uwca")
WCAforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffNonPeriodic)
system.addForce(WCAforce)

######## Debye-Huckel
if simu.Kconc >= 0.:
    DHforce = omm.CustomNonbondedForce("scale*exp(-kappa*r)/r")
    DHforce.addGlobalParameter("scale", simu.l_Bjerrum * simu.Q**2 * unit.kilocalorie_per_mole / unit.elementary_charge**2)
    DHforce.addGlobalParameter("kappa", simu.kappa)
    
    for atom in topology.atoms():
        DHforce.addParticle([])
    
    for bond in topology.bonds():
        DHforce.addExclusion(bond[0].index, bond[1].index)
    
    for chain in topology.chains():
        for prev, item, nxt in prev_and_next(chain.residues()):
            if prev == None or nxt == None:
                continue
            DHforce.addExclusion(atm_index(prev), atm_index(nxt))
    
    DHforce.setCutoffDistance(simu.cutoff)
    totalforcegroup += 1
    DHforce.setForceGroup(totalforcegroup)
    print("Force group Debye-Huckel: ", totalforcegroup)
    groupnames.append("Uele")
    DHforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffNonPeriodic)
    system.addForce(DHforce)

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

integrator = omm.LangevinIntegrator(simu.temp, 0.5/unit.picosecond, simu.dt*unit.femtoseconds)
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

if simu.restart == False:

    simulation.context.setPositions(positions)

    # Write PDB before minimization
    #state = simulation.context.getState(getPositions=True)
    #app.PDBFile.writeFile(topology, state.getPositions(), open("before_minimize.pdb", "w"), keepIds=True)

    print('Minimizing ...')
    simulation.minimizeEnergy(1*unit.kilocalorie_per_mole, 10000)

    # Write PDB after minimization
    #state = simulation.context.getState(getPositions=True)
    #app.PDBFile.writeFile(topology, state.getPositions(), open("after_minimize.pdb", "w"), keepIds=True)

    simulation.context.setVelocitiesToTemperature(simu.temp)

else:
    print("Loading checkpoint ...")
    simulation.loadCheckpoint(simu.restart_file)

simulation.reporters.append(app.DCDReporter(args.traj, args.frequency))
simulation.reporters.append(app.StateDataReporter(args.output, args.frequency, step=True, potentialEnergy=True, temperature=True, remainingTime=True, totalSteps=simu.Nstep, separator='  '))
simulation.reporters.append(EnergyReporter(args.energy, args.frequency))
simulation.reporters.append(app.CheckpointReporter(args.res_file, int(args.frequency)*100))

print('Running ...')
sys.stdout.flush()
sys.stderr.flush()

t0 = time.time()

simulation.step(simu.Nstep)

fout = open(args.finalxyz, 'w')
fout.write('%i\n' % simu.nbead)
fout.write('\n')
positions = simulation.context.getState(getPositions=True).getPositions()
for i in range(simu.nbead):
    fout.write(seq[i])
    fout.write('  %10.3f  %10.3f  %10.3f\n' % (positions[i][0]/unit.angstrom, positions[i][1]/unit.angstrom, positions[i][2]/unit.angstrom))
fout.close()


#simulation.saveState('checkpoint.xml')
prodtime = time.time() - t0
print("Simulation speed: % .2e steps/day" % (86400*simu.Nstep/(prodtime)))
