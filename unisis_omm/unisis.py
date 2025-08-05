#!/usr/bin/env python
"""
OpenMM script to run coarse-grained simulations using the Single-Interaction-Site RNA model 
A part of the code was adopted from https://github.com/tienhungf91/RNA_llps
"""
__author__ = "Naoto Hori"

import sys
print("sys.path =", sys.path)

import os
import sys
import time
import argparse
from datetime import datetime
from math import sqrt, pi, cos, log

from numpy import diag
from openmm import unit
from openmm import app
import openmm as omm

from unisis_omm.sis_params import SISForceField
from unisis_omm.control import Control
from unisis_omm.utils import *

"""
* Following modules will be imported later if needed.
    import subprocess
    import yaml
    import toml
    import torch
    from openmmtorch import TorchForce
    from torchmdnet.models.model import load_model
    from dcd import DcdFile
"""

# For timing
t0 = time.time()

################################################
#         Constants
################################################
#KELVIN_TO_KT = unit.AVOGADRO_CONSTANT_NA * unit.BOLTZMANN_CONSTANT_kB / unit.kilocalorie_per_mole
#print KELVIN_TO_KT
FILENAME_XML_DEFAULT = 'rna_cg2.xml'
u_A = unit.angstrom
u_kcalmol = unit.kilocalorie_per_mole

def main():
    ################################################
    #          Parser
    ################################################

    parser = argparse.ArgumentParser(description='OpenMM script for the SIS-RNA model',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input', type=str, help='TOML format input file (use --tmyaml if it is TorchMD YAML)')

    parser.add_argument('--tmyaml', action='store_true', help='Use this flag when the input is TorcMD format YAML file.')

    parser.add_argument('--ff', type=str, help='TOML format force-field file')
    parser.add_argument('--xml', type=str, default=None, help='XML file for topology information')
    parser.add_argument('-x','--statexml', type=str, default=None, help='State XML file to restart')
    parser.add_argument('-c','--checkpoint', type=str, default=None, help='Checkpoint file to restart')
    parser.add_argument('-r','--restart', type=str, default=None, help='Checkpoint file to restart (deprecated)')

    parser.add_argument('--cuda', action='store_true')

    #parser_device = parser.add_mutually_exclusive_group()
    #parser_device.add_argument('--cpu', action='store_true', default=False)
    #parser_device.add_argument('--cuda', action='store_true', default=False)

    parser.add_argument('--platform', type=str, default=None, help='Platform')
    parser.add_argument('--nthreads', type=int, default=None, help='Number of CPU threads')
    #parser.add_argument('--CUDAdevice', type=str, default=None,
    #                    help='CUDA device ID')

    args = parser.parse_args()

    if args.nthreads:
        if args.platform != 'CPU':
            raise argparse.ArgumentTypeError('--nthreads option can be used only when --platform=CPU')

    if args.checkpoint and args.restart:
        raise argparse.ArgumentTypeError('Checkpoint and restart cannot be used together.')

    if (args.checkpoint or args.restart) and args.statexml:
        raise argparse.ArgumentTypeError('State XML and checkpoint (restart) cannot be used together.')

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
    print('    Time: ' + datetime.now().astimezone().isoformat())
    print('    Host: ' + os.uname().nodename)
    print('    OS: ' + os.uname().version)
    print('    Python: ' + sys.version)
    print('    OpenMM version: ' + omm.version.full_version)
    print('    OpenMM library: ' + omm.version.openmm_library_path)
    print('    CWD: ' + os.getcwd())
    print('    Command:', end='')
    for s in sys.argv:
        print(' '+s, end='')
    print('')
    print('')


    ################################################
    #          Load input file
    ###############################################

    ctrl = Control()

    tmyaml_input = None
    toml_input = None
    """ It is not supposed to use both TOML input and TM-YAML input."""
    if args.tmyaml:
        import yaml
        with open(args.input) as stream:
            try:
                tmyaml_input = yaml.safe_load(stream)
            except yaml.YAMLError as err:
                print (err)
                raise

        # Load TorchMD input yaml
        ctrl.load_TorchMDyaml(tmyaml_input)

    else:
        if sys.version_info >= (3, 11):
            # tomllib is available as standard for python >= 3.11.
            import tomllib as toml
            toml_mode = "rb"
        else:
            # otherwise, toml package is needed.
            try:
                import toml
                toml_mode = "r"
            except ImportError:
                raise ImportError("For Python 3.10 or earlier, please install toml by pip/mamba/conda install toml.")
        with open(args.input, mode=toml_mode) as stream:
            try:
                toml_input = toml.load(stream)
            except:
                print ("Error: could not read the input TOML file.")
                raise

        # Load input TOML
        ctrl.load_toml(toml_input)

    ################################################
    #          Arguments override
    ################################################
    if args.restart or args.statexml or args.checkpoint:
        if ctrl.job_type != 'MD':
            print("Error: Job type has to be 'MD' in the input file to use the restart file.")
            sys.exit(2)

        ctrl.restart = True
        if args.restart is not None:
            ctrl.chk_file = args.restart
        elif args.checkpoint is not None:
            ctrl.chk_file = args.checkpoint
        elif args.statexml is not None:
            ctrl.xml_file = args.statexml

        if ctrl.chk_file is not None:
            if not os.path.isfile(ctrl.chk_file):
                print("Error: could not find the checkpoint file, " + ctrl.chk_file)
                sys.exit(2)

        elif ctrl.xml_file is not None:
            if not os.path.isfile(ctrl.xml_file):
                print("Error: could not find the state XML file, " + ctrl.xml_file)
                sys.exit(2)

    # Argument --xml overrides "xml" in the TOML input
    if args.xml is not None:
        ctrl.xml = args.xml
    # If neigher --xml or 'xml' in TOML exist, try to find the default xml file.
    elif ctrl.xml is None:
        ctrl.xml = os.path.dirname(os.path.realpath(__file__)) + '/../params/' + FILENAME_XML_DEFAULT

    # Argument --ff overrides "ff" in the TOML input
    if args.ff is not None:
        ctrl.ff = args.ff

    if args.cuda:
        ctrl.device = 'CUDA'
    else:
        if args.platform in ('CUDA', 'CPU', 'OpenCL'):
            ctrl.device = args.platform
        else:
            ctrl.device = None

    if args.nthreads is not None:
        ctrl.nthreads = args.nthreads

    print(ctrl)

    ################################################
    #   Create "forcefield" from XML file
    ################################################
    if not os.path.isfile(ctrl.xml):
        print("Error: could not find topology XML file, " + ctrl.xml)
        print("You can specify the correct path by either --xml or 'xml' in the TOML input")
        sys.exit(2)

    forcefield = app.ForceField(ctrl.xml)

    ################################################
    #   Construct topology and positions
    ################################################

    # Sequence and structure are read from PDB if the input format is TorchMD YAML
    # The residue names have to be 'RA', 'RG', 'RU', 'RC', and 'RD'.
    if tmyaml_input is not None:
        cgpdb = app.PDBFile(ctrl.infile_pdb)
        topology = cgpdb.getTopology()
        positions = cgpdb.getPositions()
        nnt = len(positions)

    # Otherwise, sequence is read from FASTA, structure is read from PDB or XYZ
    # Note that any sequence information in PDB and XYZ will be ignored. Only 
    # coordinates therein will be used.
    else:
        topology = app.Topology()
        positions = []

        print('Reading sequences from FASTA file, ', ctrl.fasta)
        chain_names = []
        seqs = []
        seq = ''
        chain_name_save = None

        try:
            with open(ctrl.fasta, 'r') as stream:
                for line_no, line in enumerate(stream, 1):
                    if line.startswith('>'):
                        if len(seq) > 0:
                            seqs.append(seq)
                            chain_names.append(chain_name_save)
                            seq = ''
                        chain_name_save = line.strip()
                        continue
                    else:
                        # Validate nucleotide sequence
                        seq_line = line.strip().upper()
                        if seq_line and not all(c in 'ACGUNDT' for c in seq_line):
                            print(f'Error: Invalid nucleotide character found at line {line_no} in FASTA file')
                            print(f'       Valid characters are: A, C, G, U, N, D, T')
                            sys.exit(2)
                        seq += seq_line

                # Don't forget the last sequence
                if len(seq) > 0:
                    seqs.append(seq)
                    chain_names.append(chain_name_save)

        except FileNotFoundError:
            print(f'Error: cannot find the FASTA file: {ctrl.fasta}')
            sys.exit(2)
        except IOError as e:
            print(f'Error: cannot read the FASTA file: {ctrl.fasta}')
            print(f'       {e}')
            sys.exit(2)

        # Validation
        if not seqs:
            print('Error: No sequences found in FASTA file')
            sys.exit(2)

        if any(len(seq) == 0 for seq in seqs):
            print('Error: Empty sequence found in FASTA file')
            sys.exit(2)

        print_length = 80
        nnt = 0  # Number of the total nucleotides
        for i, seq in enumerate(seqs):
            nnt += len(seq)
            print('+--------1---------2---------3---------4---------5---------6---------7---------8')
            print(f'> Chain {i+1}, {len(seq)} nts.  {chain_names[i]}')
            for j in range(0, len(seq), print_length):
                print(seq[0+j:print_length+j])
        print('+------------------------------------------------------------------------------+')
        print(f'Total number of particles: {nnt}\n')

        if ctrl.infile_pdb is not None:
            print(f'Reading the initial structure from PDB file, {ctrl.infile_pdb}')
            try:
                with open(ctrl.infile_pdb, 'r') as stream:
                    atom_count = 0
                    for line_no, line in enumerate(stream, 1):
                        if line.startswith('ATOM  '):
                            try:
                                # Ensure proper PDB format (columns 30-38, 38-46, 46-54)
                                if len(line) < 54:
                                    print(f'Error: Invalid PDB format at line {line_no}')
                                    print(f'       Line too short for coordinate parsing')
                                    sys.exit(2)

                                x_str = line[30:38].strip()
                                y_str = line[38:46].strip()
                                z_str = line[46:54].strip()

                                if not x_str or not y_str or not z_str:
                                    print(f'Error: Missing coordinates at line {line_no} in PDB file')
                                    sys.exit(2)

                                x = float(x_str) * u_A
                                y = float(y_str) * u_A
                                z = float(z_str) * u_A
                                positions.append([x, y, z])
                                atom_count += 1

                            except ValueError as e:
                                print(f'Error: Invalid coordinate value at line {line_no} in PDB file')
                                print(f'       {e}')
                                sys.exit(2)

                    print(f"    {atom_count} ATOM records processed from PDB file")

            except FileNotFoundError:
                print(f'Error: cannot find the PDB file: {ctrl.infile_pdb}')
                sys.exit(2)
            except IOError as e:
                print(f'Error: cannot read the PDB file: {ctrl.infile_pdb}')
                print(f'       {e}')
                sys.exit(2)

        elif ctrl.infile_xyz is not None:
            print(f'Reading the initial structure from XYZ file, {ctrl.infile_xyz}')
            try:
                with open(ctrl.infile_xyz, 'r') as stream:
                    lines = stream.readlines()

                    if len(lines) < 2:
                        print('Error: XYZ file must have at least 2 lines (number of atoms and comment)')
                        sys.exit(2)

                    try:
                        expected_atoms = int(lines[0].strip())
                    except ValueError:
                        print('Error: First line of XYZ file must contain the number of atoms')
                        sys.exit(2)

                    atom_count = 0
                    for line_no, line in enumerate(lines[2:], 3):  # Skip first two lines
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            parts = line.split()
                            if len(parts) < 4:
                                print(f'Error: Invalid XYZ format at line {line_no}')
                                print(f'       Expected: element x y z')
                                sys.exit(2)

                            element = parts[0]
                            x = float(parts[1]) * u_A
                            y = float(parts[2]) * u_A
                            z = float(parts[3]) * u_A
                            positions.append([x, y, z])
                            atom_count += 1

                        except (ValueError, IndexError) as e:
                            print(f'Error: Invalid coordinate at line {line_no} in XYZ file')
                            print(f'       {e}')
                            sys.exit(2)

                    if atom_count != expected_atoms:
                        print(f'Error: XYZ file inconsistency')
                        print(f'       Expected {expected_atoms} atoms, found {atom_count}')
                        sys.exit(2)

                    print(f"    {atom_count} atoms processed from XYZ file")

            except FileNotFoundError:
                print(f'Error: cannot find the XYZ file: {ctrl.infile_xyz}')
                sys.exit(2)
            except IOError as e:
                print(f'Error: cannot read the XYZ file: {ctrl.infile_xyz}')
                print(f'       {e}')
                sys.exit(2)
        else:
            print("Error: either PDB or XYZ is required for the initial structure.")
            print("       Specify 'pdb_ini' or 'xyz_ini' in the input file.")
            sys.exit(2)

        print(f"    Coordinates for {len(positions)} particles loaded.\n")

        if len(positions) != nnt:
            print(f'Error: inconsistency between FASTA file and PDB/XYZ file.')
            print(f'       Number of nucleotides in the FASTA file, {nnt}.')
            print(f'       Number of nucleotides in the PDB/XYZ file, {len(positions)}.')
            sys.exit(2)

        name_map = {'A': 'RA', 'C': 'RC', 'G': 'RG', 'U': 'RU', 'D': 'RD', 'N': 'RN'}
        for seq in seqs:
            chain = topology.addChain()
            for i, s in enumerate(seq):
                c = name_map[s]
                res = topology.addResidue(c, chain)
                atom = forcefield._templates[c].atoms[0]
                topology.addAtom(atom.name, forcefield._atomTypes[atom.type].element, res)


    for c in topology.chains():
        for prev, item, nxt in prev_and_next(c.residues()):
            #item.name = name_map[item.name]

            if prev is None or nxt is None:
                if len(item.name) == 1:
                    item.name += 'T'

            if prev is not None:
                topology.addBond(get_atom(prev), get_atom(item))

    if ctrl.PBC:
        print("Setting the periodic boundary condition box")
        print("    ", ctrl.PBC_size, '\n')
        topology.setPeriodicBoxVectors(([ctrl.PBC_size[0] * 0.1, 0., 0.],
                                        [0., ctrl.PBC_size[1] * 0.1, 0.],
                                        [0., 0., ctrl.PBC_size[2] * 0.1]))
                        # This argument vector has to be in nanometer.

    ################################################
    #             Load force field
    ################################################
    ff = SISForceField()

    if ctrl.ff is not None:
        ff.read_toml(ctrl.ff)
    else:
        print("WARNING: Force field (ff) file was not specified. Default values are used.")

    print(ff)

    def calculate_electrostatic_parameters(temperature, ionic_strength, length_per_charge, cutoff_type, cutoff_factor):
        """
        Calculate electrostatic parameters for Debye-Huckel model.

        Parameters:
        -----------
        temperature : unit.Quantity (unit.kelvin)
            Temperature
        ionic_strength : float
            Ionic strength in M
        length_per_charge : unit.Quantity (unit.angstrom)
            Length per charge in Angstrom
        cutoff_type : int
            1 for fixed cutoff, 2 for Debye-length based cutoff
        cutoff_factor : float
            Cutoff factor

        Returns:
        --------
        tuple : (cutoff, scale, kappa, reduced_charge)
            Calculated electrostatic parameters
        """
        # Physical constants
        ELEC = 1.602176634e-19      # Elementary charge [C]
        EPS0 = 8.8541878128e-12     # Electric constant [F/m]
        BOLTZ_J = 1.380649e-23      # Boltzmann constant [J/K]
        N_AVO = 6.02214076e23       # Avogadro constant [/mol]
        JOUL2KCAL_MOL = 1.0 / 4184.0 * N_AVO  # (J -> kcal/mol)

        # Input validation
        T_value = temperature.value_in_unit(unit.kelvin)
        if T_value <= 0:
            raise ValueError(f"Temperature must be positive, got {T_value} K")
        if ionic_strength <= 0:
            raise ValueError(f"Ionic strength must be positive, got {ionic_strength} M")
        if length_per_charge.value_in_unit(unit.angstrom) <= 0:
            raise ValueError(f"Length per charge must be positive, got {length_per_charge}")
        if cutoff_type not in [1, 2]:
            raise ValueError(f"Cutoff type must be 1 or 2, got {cutoff_type}")
        if cutoff_factor <= 0:
            raise ValueError(f"Cutoff factor must be positive, got {cutoff_factor}")

        # Calculate temperature-dependent dielectric constant for water
        Tc = T_value - 273.15
        if Tc < -10 or Tc > 100:
            print(f"Warning: Temperature {Tc}°C is outside the typical range (-10°C to 100°C)")
            print(f"         Dielectric constant calculation may be inaccurate")

        diele = 87.740 - 0.40008 * Tc + 9.398e-4 * Tc**2 - 1.410e-6 * Tc**3

        if diele <= 0:
            raise ValueError(f"Calculated dielectric constant is non-positive: {diele}")

        # Bjerrum length
        lb = ELEC**2 / (4.0 * pi * EPS0 * diele * BOLTZ_J * T_value) * 1.0e10 * u_A

        # Reduced charge
        Zp = -length_per_charge / lb

        # Debye length
        lambdaD = 1.0e10 * sqrt((1.0e-3 * EPS0 * diele * BOLTZ_J) / 
                               (2.0 * N_AVO * ELEC**2)) * sqrt(T_value / ionic_strength) * u_A

        # Set cutoff distance
        if cutoff_type == 1:
            cutoff = cutoff_factor * u_A
        elif cutoff_type == 2:
            cutoff = cutoff_factor * lambdaD

        # Screening parameter and energy scale
        kappa = 1.0 / lambdaD
        scale = JOUL2KCAL_MOL * 1.0e10 * ELEC**2 / (4.0 * pi * EPS0 * diele) * u_kcalmol * u_A

        # Output information
        print(f"    Debye-Huckel electrostatics:")
        print(f"        Ionic strength: {ionic_strength} M")
        print(f"        Temperature: {temperature}")
        print(f"        Dielectric constant (T dependent): {diele}")
        print(f"        Bjerrum length: {lb}")
        print(f"        Reduced charge: {Zp}")
        print(f"        Debye length (lambda_D): {lambdaD}")
        print(f"        Cutoff: {cutoff}")
        print(f"        Scale: {scale}")
        print(f"        Screening parameter (kappa): {kappa}")

        return cutoff, scale, kappa, Zp

    if ctrl.ele:
        ele_cutoff, ele_scale, ele_kappa, ele_Zp = calculate_electrostatic_parameters(
            ctrl.temp,                      # T
            ctrl.ele_ionic_strength,        # C
            ctrl.ele_length_per_charge,     # lp
            ctrl.ele_cutoff_type,           # cut_type
            ctrl.ele_cutoff_factor          # cut_factor
        )

    print('')


    ################################################
    #             Construct forces
    ################################################
    print("Constructing forces:")
    system = forcefield.createSystem(topology)

    totalforcegroup = 0
    ### Avoid use of default forcegroup = 0 as CMMotionRemove is already set with forcegroup = 0.
    forcegroup = {}
    groupnames = []

    ########## Bond
    if ff.bond:
        totalforcegroup += 1
        forcegroup['Bond'] = totalforcegroup
        print(f"    {totalforcegroup:2d}:    Bond")
        groupnames.append("Ubond")

        bondforce = omm.HarmonicBondForce()
        for bond in topology.bonds():
            bondforce.addBond(bond[0].index, bond[1].index, ff.bond_r0, ff.bond_k)

        bondforce.setForceGroup(totalforcegroup)
        system.addForce(bondforce)

    ########## Angle
    if ff.angle:
        totalforcegroup += 1
        forcegroup['Angle'] = totalforcegroup
        print(f"    {totalforcegroup:2d}:    Angle")
        groupnames.append("Uangl")

        angleforce = omm.HarmonicAngleForce()

        for chain in topology.chains():
            for prev, item, nxt in prev_and_next(chain.residues()):
                if prev == None or nxt == None:
                    continue

                angleforce.addAngle(prev.index, item.index, nxt.index, ff.angle_a0, ff.angle_k)

        angleforce.setForceGroup(totalforcegroup)
        system.addForce(angleforce)

    ########## Restricted Bending (ReB)
    if ff.angle_ReB:
        totalforcegroup += 1
        forcegroup['ReB'] = totalforcegroup
        print(f"    {totalforcegroup:2d}:    Angle ReB")
        groupnames.append("Uangl")

        #ReB_energy_function = '0.5 * ReB_k * (cos(theta) - cos_ReB_a0)^2 / (sin(theta)^2)'
        ReB_energy_function = '0.5 * ReB_k * (cos(theta) - cos_ReB_a0)^2 / (1.0 - cos(theta)^2)'

        cos_0 = cos(ff.angle_a0.value_in_unit(unit.radian))
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

        ReBforce.setForceGroup(totalforcegroup)
        system.addForce(ReBforce)

    ########## Dihedral (exponential)
    if ff.dihexp:
        totalforcegroup += 1
        forcegroup['DihExp'] = totalforcegroup
        print(f"    {totalforcegroup:2d}:    Dihexp")
        groupnames.append("Udih")

        dihedral_energy_function = '-dihexp_k * exp(-0.5 * dihexp_w * (theta - dihexp_p0)^2)'
        # This is not strictly correct when theta is close to -pi (theta is defined in [-pi, pi]), 
        # but the error is negligible because dihexp_p0 is around zero (~0.267), far from pi,
        # thus the exponential is almost zero anyway.
        # For theta = -3.13, the energy difference between the true expression and this is
        # ~ 0.000005 kcal/mol (k = 1.4 kcal/mol, dihexp_w = 3.0)

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

        dihedralforce.setForceGroup(totalforcegroup)
        system.addForce(dihedralforce)

    ########## Base pair
    if ff.bp and ctrl.BP_model > 0:

        totalforcegroup += 1
        forcegroup['BP'] = totalforcegroup
        print(f"    {totalforcegroup:2d}:    BP")
        groupnames.append("Ubp")

        """ For specific secondary structure """
        bps = []
        for l in open(ctrl.infile_bpcoef):
            lsp = l.split()
            imp = int(lsp[1])
            jmp = int(lsp[2])
            imp3 = lsp[3][0:3]
            jmp3 = lsp[3][4:7]
            u0 = float(lsp[4])
            bps.append((imp, jmp, imp3, jmp3, u0))

        #energy_function = "step(-penalty) * Ubp0 * exp(penalty);"
        energy_function = "select(step(dcut - abs(r-r0)), Ubp0 * exp(-penalty), 0);"
        #energy_function = "select(step(dcut - abs(r-r0)), select(step(penalty), Ubp0 * exp(-penalty), 0), 0);"
        energy_function += "penalty = "
        energy_function += "kr *(r - r0)^2"
        energy_function += "+ kt1*(angle(p1, p2, p4) - theta1)^2"
        energy_function += "+ kt2*(angle(p2, p1, p3) - theta2)^2"
        energy_function += "+ kt3*(angle(p1, p2, p6) - theta3)^2"
        energy_function += "+ kt4*(angle(p2, p1, p5) - theta4)^2"
        energy_function += "+ kp1*(1. + cos(dihedral(p4, p2, p1, p3) + phi1))"
        energy_function += "+ kp2*(1. + cos(dihedral(p6, p2, p1, p5) + phi2));"
        energy_function += "r = distance(p1, p2)"

        cutoff_ddist_GC = sqrt(log(abs(10.0/0.01)) / ff.GC_bond_k.value_in_unit(u_A**(-2))) * u_A
        cutoff_ddist_AU = sqrt(log(abs(10.0/0.01)) / ff.AU_bond_k.value_in_unit(u_A**(-2))) * u_A
        cutoff_ddist_GU = sqrt(log(abs(10.0/0.01)) / ff.GU_bond_k.value_in_unit(u_A**(-2))) * u_A
        #cutoff_GC = ff.GC_bond_r + cutoff_ddist_GC
        #cutoff_AU = ff.AU_bond_r + cutoff_ddist_AU
        #cutoff_GU = ff.GU_bond_r + cutoff_ddist_GU
        #print('           - cutoff_ddist_GC', cutoff_ddist_GC)
        #print('           - cutoff_ddist_AU', cutoff_ddist_AU)
        #print('           - cutoff_ddist_GU', cutoff_ddist_GU)
        #print('           - cutoff_GC', cutoff_GC)
        #print('           - cutoff_AU', cutoff_AU)
        #print('           - cutoff_GU', cutoff_GU)
        para_list_GC = [cutoff_ddist_GC, ff.GC_bond_k, ff.GC_bond_r,
                        ff.GC_angl_k1, ff.GC_angl_k2, ff.GC_angl_k3, ff.GC_angl_k4,
                        ff.GC_angl_theta1, ff.GC_angl_theta2, ff.GC_angl_theta3, ff.GC_angl_theta4,
                        ff.GC_dihd_k1, ff.GC_dihd_k2, ff.GC_dihd_phi1, ff.GC_dihd_phi2]
        para_list_AU = [cutoff_ddist_AU, ff.AU_bond_k, ff.AU_bond_r,
                        ff.AU_angl_k1, ff.AU_angl_k2, ff.AU_angl_k3, ff.AU_angl_k4,
                        ff.AU_angl_theta1, ff.AU_angl_theta2, ff.AU_angl_theta3, ff.AU_angl_theta4,
                        ff.AU_dihd_k1, ff.AU_dihd_k2, ff.AU_dihd_phi1, ff.AU_dihd_phi2]
        para_list_GU = [cutoff_ddist_GU, ff.GU_bond_k, ff.GU_bond_r,
                        ff.GU_angl_k1, ff.GU_angl_k2, ff.GU_angl_k3, ff.GU_angl_k4,
                        ff.GU_angl_theta1, ff.GU_angl_theta2, ff.GU_angl_theta3, ff.GU_angl_theta4,
                        ff.GU_dihd_k1, ff.GU_dihd_k2, ff.GU_dihd_phi1, ff.GU_dihd_phi2]

        for imp, jmp, imp3, jmp3, u0 in bps:
            p = imp3[1] + jmp3[1]
            i = imp - 1
            j = jmp - 1

            CCBforce = omm.CustomCompoundBondForce(6, energy_function)
            CCBforce.addPerBondParameter('Ubp0')
            CCBforce.addPerBondParameter('dcut')
            CCBforce.addPerBondParameter('kr')
            CCBforce.addPerBondParameter('r0')
            CCBforce.addPerBondParameter('kt1')
            CCBforce.addPerBondParameter('kt2')
            CCBforce.addPerBondParameter('kt3')
            CCBforce.addPerBondParameter('kt4')
            CCBforce.addPerBondParameter('theta1')
            CCBforce.addPerBondParameter('theta2')
            CCBforce.addPerBondParameter('theta3')
            CCBforce.addPerBondParameter('theta4')
            CCBforce.addPerBondParameter('kp1')
            CCBforce.addPerBondParameter('kp2')
            CCBforce.addPerBondParameter('phi1')
            CCBforce.addPerBondParameter('phi2')
            #if p in ('GC', 'CG'):
            #    CCBforce.setCutoffDistance(cutoff_GC)
            #elif p in ('AU', 'UA'):
            #    CCBforce.setCutoffDistance(cutoff_AU)
            #elif p in ('GU', 'UG'):
            #    CCBforce.setCutoffDistance(cutoff_GU)
            CCBforce.setUsesPeriodicBoundaryConditions(ctrl.PBC)
            CCBforce.setForceGroup(totalforcegroup)

            para_list = [u0 * u_kcalmol,]
            if p in ('GC', 'CG'):
                para_list += para_list_GC
            elif p in ('AU', 'UA'):
                para_list += para_list_AU
            elif p in ('GU', 'UG'):
                para_list += para_list_GU
            else:
                print("Error: unknown pair. ", bp3, p)
                sys.exit(2)

            CCBforce.addBond([i, j, i-1, j-1, i+1, j+1], para_list)

            system.addForce(CCBforce)


        """ Make Hbforce for each pair """
        """
        bps = {}
        bps_u0 = {}
        for l in open(ctrl.infile_bpcoef):
            lsp = l.split()
            imp = int(lsp[1])
            jmp = int(lsp[2])
            imp3 = lsp[3][0:3]
            jmp3 = lsp[3][4:7]
            imp3_rev = imp3[::-1]
            jmp3_rev = jmp3[::-1]
            u0 = float(lsp[4])
            if (imp3, jmp3) in bps:
                bps[(imp3, jmp3)].append((imp, jmp))
                assert bps_u0[(imp3, jmp3)] == u0
            elif (jmp3_rev, imp3_rev) in bps:
                bps[(jmp3_rev, imp3_rev)].append((jmp, imp))
                assert bps_u0[(jmp3_rev, imp3_rev)] == u0
            else:
                bps[(imp3, jmp3)] = [(imp, jmp),]
                bps_u0[(imp3, jmp3)] = u0
        #print ('len(bps)', len(bps))
        #print ('len(bps_u0)', len(bps_u0))

        #energy_function = "step(-penalty) * Ubp0 * exp(penalty);"
        energy_function = "select(step(dcut - abs(r-r0)), Ubp0 * exp(-penalty), 0);"
        #energy_function = "select(step(dcut - abs(r-r0)), select(step(penalty), Ubp0 * exp(-penalty), 0), 0);"
        energy_function += "penalty = "
        energy_function += "kr *(r - r0)^2"
        energy_function += "+ kt1*(angle(a1, d1, d2) - theta1)^2"
        energy_function += "+ kt2*(angle(d1, a1, a2) - theta2)^2"
        energy_function += "+ kt3*(angle(a1, d1, d3) - theta3)^2"
        energy_function += "+ kt4*(angle(d1, a1, a3) - theta4)^2"
        energy_function += "+ kp1*(1. + cos(dihedral(d2, d1, a1, a2) + phi1))"
        energy_function += "+ kp2*(1. + cos(dihedral(d3, d1, a1, a3) + phi2));"
        energy_function += "r = distance(a1, d1)"
        #energy_function += "penalty = "
        #energy_function += "kr *(r - r0)^2"
        #energy_function += "+ kt1*(t1 - theta1)^2"
        #energy_function += "+ kt2*(t2 - theta2)^2"
        #energy_function += "+ kt3*(t3 - theta3)^2"
        #energy_function += "+ kt4*(t4 - theta4)^2"
        #energy_function += "+ select(max(0, pi-(max(t1,t2)+theta_zero)), kp1*(1. + cos(dihedral(d2, d1, a1, a2) + phi1)), 20)"
        #energy_function += "+ select(max(0, pi-(max(t3,t4)+theta_zero)), kp2*(1. + cos(dihedral(d3, d1, a1, a3) + phi2)), 20);"
        #energy_function += "r = distance(a1, d1);"
        #energy_function += "t1 = angle(a1, d1, d2);"
        #energy_function += "t2 = angle(d1, a1, a2);"
        #energy_function += "t3 = angle(a1, d1, d3);"
        #energy_function += "t4 = angle(d1, a1, a3);"
        #energy_function += "theta_zero = 0.001;"
        #energy_function += f"pi = {pi}"

        cutoff_ddist_GC = sqrt(log(abs(10.0/0.01)) / ff.GC_bond_k.value_in_unit(u_A**(-2))) * u_A
        cutoff_ddist_AU = sqrt(log(abs(10.0/0.01)) / ff.AU_bond_k.value_in_unit(u_A**(-2))) * u_A
        cutoff_ddist_GU = sqrt(log(abs(10.0/0.01)) / ff.GU_bond_k.value_in_unit(u_A**(-2))) * u_A
        cutoff_GC = ff.GC_bond_r + cutoff_ddist_GC
        cutoff_AU = ff.AU_bond_r + cutoff_ddist_AU
        cutoff_GU = ff.GU_bond_r + cutoff_ddist_GU
        print('cutoff_ddist_GC', cutoff_ddist_GC)
        print('cutoff_ddist_AU', cutoff_ddist_AU)
        print('cutoff_ddist_GU', cutoff_ddist_GU)
        print('cutoff_GC', cutoff_GC)
        print('cutoff_AU', cutoff_AU)
        print('cutoff_GU', cutoff_GU)
        para_list_GC = [cutoff_ddist_GC, ff.GC_bond_k, ff.GC_bond_r,
                        ff.GC_angl_k1, ff.GC_angl_k2, ff.GC_angl_k3, ff.GC_angl_k4,
                        ff.GC_angl_theta1, ff.GC_angl_theta2, ff.GC_angl_theta3, ff.GC_angl_theta4,
                        ff.GC_dihd_k1, ff.GC_dihd_k2, ff.GC_dihd_phi1, ff.GC_dihd_phi2]
        para_list_AU = [cutoff_ddist_AU, ff.AU_bond_k, ff.AU_bond_r,
                        ff.AU_angl_k1, ff.AU_angl_k2, ff.AU_angl_k3, ff.AU_angl_k4,
                        ff.AU_angl_theta1, ff.AU_angl_theta2, ff.AU_angl_theta3, ff.AU_angl_theta4,
                        ff.AU_dihd_k1, ff.AU_dihd_k2, ff.AU_dihd_phi1, ff.AU_dihd_phi2]
        para_list_GU = [cutoff_ddist_GU, ff.GU_bond_k, ff.GU_bond_r,
                        ff.GU_angl_k1, ff.GU_angl_k2, ff.GU_angl_k3, ff.GU_angl_k4,
                        ff.GU_angl_theta1, ff.GU_angl_theta2, ff.GU_angl_theta3, ff.GU_angl_theta4,
                        ff.GU_dihd_k1, ff.GU_dihd_k2, ff.GU_dihd_phi1, ff.GU_dihd_phi2]

        for bp3, bps_list in bps.items():
            #print (bp3, len(bps_list))
            p = bp3[0][1] + bp3[1][1]

            for idx, (imp, jmp) in enumerate(bps_list):
                i = imp - 1
                j = jmp - 1

                Hbforce = omm.CustomHbondForce(energy_function)
                Hbforce.addPerDonorParameter('Ubp0')
                Hbforce.addPerDonorParameter('dcut')
                Hbforce.addPerDonorParameter('kr')
                Hbforce.addPerDonorParameter('r0')
                Hbforce.addPerDonorParameter('kt1')
                Hbforce.addPerDonorParameter('kt2')
                Hbforce.addPerDonorParameter('kt3')
                Hbforce.addPerDonorParameter('kt4')
                Hbforce.addPerDonorParameter('theta1')
                Hbforce.addPerDonorParameter('theta2')
                Hbforce.addPerDonorParameter('theta3')
                Hbforce.addPerDonorParameter('theta4')
                Hbforce.addPerDonorParameter('kp1')
                Hbforce.addPerDonorParameter('kp2')
                Hbforce.addPerDonorParameter('phi1')
                Hbforce.addPerDonorParameter('phi2')
                if p in ('GC', 'CG'):
                    Hbforce.setCutoffDistance(cutoff_GC)
                elif p in ('AU', 'UA'):
                    Hbforce.setCutoffDistance(cutoff_AU)
                elif p in ('GU', 'UG'):
                    Hbforce.setCutoffDistance(cutoff_GU)
                if ctrl.PBC:
                    Hbforce.setNonbondedMethod(omm.CustomHbondForce.CutoffPeriodic)
                else:
                    Hbforce.setNonbondedMethod(omm.CustomHbondForce.CutoffNonPeriodic)
                Hbforce.setForceGroup(totalforcegroup)
        
                para_list = [bps_u0[bp3] * u_kcalmol,]
                if p in ('GC', 'CG'):
                    para_list += para_list_GC
                elif p in ('AU', 'UA'):
                    para_list += para_list_AU
                elif p in ('GU', 'UG'):
                    para_list += para_list_GU
                else:
                    print("Error: unknown pair. ", bp3, p)
                    sys.exit(2)
        
                #print (imp, jmp, para_list)
                Hbforce.addAcceptor(i, i-1, i+1, [])
                Hbforce.addDonor   (j, j-1, j+1, para_list)
        
                system.addForce(Hbforce)
        """


        """ Grouping by three nucleotide combinations """
        """
    ### save
        bps = {}
        bps_u0 = {}
        for l in open(ctrl.infile_bpcoef):
            lsp = l.split()
            imp = int(lsp[1])
            jmp = int(lsp[2])
            imp3 = lsp[3][0:3]
            jmp3 = lsp[3][4:7]
            imp3_rev = imp3[::-1]
            jmp3_rev = jmp3[::-1]
            u0 = float(lsp[4])
            if (imp3, jmp3) in bps:
                bps[(imp3, jmp3)].append((imp, jmp))
                assert bps_u0[(imp3, jmp3)] == u0
            elif (jmp3_rev, imp3_rev) in bps:
                bps[(jmp3_rev, imp3_rev)].append((jmp, imp))
                assert bps_u0[(jmp3_rev, imp3_rev)] == u0
            else:
                bps[(imp3, jmp3)] = [(imp, jmp),]
                bps_u0[(imp3, jmp3)] = u0
        #print ('len(bps)', len(bps))
        #print ('len(bps_u0)', len(bps_u0))

        #energy_function = "step(-penalty) * Ubp0 * exp(penalty);"
        #energy_function = "select(step(dcut - abs(r-r0)), Ubp0 * exp(-penalty), 0);"
        energy_function = "select(max(0, dcut - abs(r-r0)), select(step(penalty), Ubp0 * exp(-penalty), 0), 0);"
    #    energy_function += "penalty = "
    #    energy_function += "kr *(r - r0)^2"
    #    energy_function += "+ kt1*(angle(a1, d1, d2) - theta1)^2"
    #    energy_function += "+ kt2*(angle(d1, a1, a2) - theta2)^2"
    #    energy_function += "+ kt3*(angle(a1, d1, d3) - theta3)^2"
    #    energy_function += "+ kt4*(angle(d1, a1, a3) - theta4)^2"
    #    energy_function += "+ kp1*(1. + cos(dihedral(d2, d1, a1, a2) + phi1))"
    #    energy_function += "+ kp2*(1. + cos(dihedral(d3, d1, a1, a3) + phi2));"
    #    energy_function += "r = distance(a1, d1)"
        energy_function += "penalty = "
        energy_function += "kr *(r - r0)^2"
        energy_function += "+ kt1*(t1 - theta1)^2"
        energy_function += "+ kt2*(t2 - theta2)^2"
        energy_function += "+ kt3*(t3 - theta3)^2"
        energy_function += "+ kt4*(t4 - theta4)^2"
        energy_function += "+ select(max(0, pi-(max(t1,t2)+theta_zero)), kp1*(1. + cos(dihedral(d2, d1, a1, a2) + phi1)), 20)"
        energy_function += "+ select(max(0, pi-(max(t3,t4)+theta_zero)), kp2*(1. + cos(dihedral(d3, d1, a1, a3) + phi2)), 20);"
        energy_function += "r = distance(a1, d1);"
        energy_function += "t1 = angle(a1, d1, d2);"
        energy_function += "t2 = angle(d1, a1, a2);"
        energy_function += "t3 = angle(a1, d1, d3);"
        energy_function += "t4 = angle(d1, a1, a3);"
        energy_function += "theta_zero = 0.001;"
        energy_function += f"pi = {pi}"

        cutoff_ddist_GC = sqrt(log(abs(10.0/0.01)) / ff.GC_bond_k.value_in_unit(u_A**(-2))) * u_A
        cutoff_ddist_AU = sqrt(log(abs(10.0/0.01)) / ff.AU_bond_k.value_in_unit(u_A**(-2))) * u_A
        cutoff_ddist_GU = sqrt(log(abs(10.0/0.01)) / ff.GU_bond_k.value_in_unit(u_A**(-2))) * u_A
        cutoff_GC = ff.GC_bond_r + cutoff_ddist_GC
        cutoff_AU = ff.AU_bond_r + cutoff_ddist_AU
        cutoff_GU = ff.GU_bond_r + cutoff_ddist_GU
        print('cutoff_ddist_GC', cutoff_ddist_GC)
        print('cutoff_ddist_AU', cutoff_ddist_AU)
        print('cutoff_ddist_GU', cutoff_ddist_GU)
        print('cutoff_GC', cutoff_GC)
        print('cutoff_AU', cutoff_AU)
        print('cutoff_GU', cutoff_GU)
        para_list_GC = [cutoff_ddist_GC, ff.GC_bond_k, ff.GC_bond_r,
                        ff.GC_angl_k1, ff.GC_angl_k2, ff.GC_angl_k3, ff.GC_angl_k4,
                        ff.GC_angl_theta1, ff.GC_angl_theta2, ff.GC_angl_theta3, ff.GC_angl_theta4,
                        ff.GC_dihd_k1, ff.GC_dihd_k2, ff.GC_dihd_phi1, ff.GC_dihd_phi2]
        para_list_AU = [cutoff_ddist_AU, ff.AU_bond_k, ff.AU_bond_r,
                        ff.AU_angl_k1, ff.AU_angl_k2, ff.AU_angl_k3, ff.AU_angl_k4,
                        ff.AU_angl_theta1, ff.AU_angl_theta2, ff.AU_angl_theta3, ff.AU_angl_theta4,
                        ff.AU_dihd_k1, ff.AU_dihd_k2, ff.AU_dihd_phi1, ff.AU_dihd_phi2]
        para_list_GU = [cutoff_ddist_GU, ff.GU_bond_k, ff.GU_bond_r,
                        ff.GU_angl_k1, ff.GU_angl_k2, ff.GU_angl_k3, ff.GU_angl_k4,
                        ff.GU_angl_theta1, ff.GU_angl_theta2, ff.GU_angl_theta3, ff.GU_angl_theta4,
                        ff.GU_dihd_k1, ff.GU_dihd_k2, ff.GU_dihd_phi1, ff.GU_dihd_phi2]

        for bp3, bps_list in bps.items():
            #print (bp3, len(bps_list))
            p = bp3[0][1] + bp3[1][1]

            Hbforce = omm.CustomHbondForce(energy_function)
            Hbforce.addPerDonorParameter('Ubp0')
            Hbforce.addPerDonorParameter('dcut')
            Hbforce.addPerDonorParameter('kr')
            Hbforce.addPerDonorParameter('r0')
            Hbforce.addPerDonorParameter('kt1')
            Hbforce.addPerDonorParameter('kt2')
            Hbforce.addPerDonorParameter('kt3')
            Hbforce.addPerDonorParameter('kt4')
            Hbforce.addPerDonorParameter('theta1')
            Hbforce.addPerDonorParameter('theta2')
            Hbforce.addPerDonorParameter('theta3')
            Hbforce.addPerDonorParameter('theta4')
            Hbforce.addPerDonorParameter('kp1')
            Hbforce.addPerDonorParameter('kp2')
            Hbforce.addPerDonorParameter('phi1')
            Hbforce.addPerDonorParameter('phi2')
            #if p in ('GC', 'CG'):
            #    Hbforce.setCutoffDistance(cutoff_GC)
            #elif p in ('AU', 'UA'):
            #    Hbforce.setCutoffDistance(cutoff_AU)
            #elif p in ('GU', 'UG'):
            #    Hbforce.setCutoffDistance(cutoff_GU)
            #if ctrl.PBC:
            #    Hbforce.setNonbondedMethod(omm.CustomHbondForce.CutoffPeriodic)
            #else:
            #    Hbforce.setNonbondedMethod(omm.CustomHbondForce.CutoffNonPeriodic)
            Hbforce.setNonbondedMethod(omm.CustomHbondForce.NoCutoff)
            Hbforce.setForceGroup(totalforcegroup)

            para_list = [bps_u0[bp3] * u_kcalmol,]
            if p in ('GC', 'CG'):
                para_list += para_list_GC
            elif p in ('AU', 'UA'):
                para_list += para_list_AU
            elif p in ('GU', 'UG'):
                para_list += para_list_GU
            else:
                print("Error: unknown pair. ", bp3, p)
                sys.exit(2)

            list_i_Acc = {}
            list_j_Don = {}
            for idx, (imp, jmp) in enumerate(bps_list):
                i = imp - 1
                j = jmp - 1
                #print (imp, jmp, para_list)
                Hbforce.addAcceptor(i, i-1, i+1, [])
                Hbforce.addDonor   (j, j-1, j+1, para_list)
                list_i_Acc[i] = idx
                list_j_Don[j] = idx

            for i, Acc in list_i_Acc.items():
                for j, Don in list_j_Don.items():
                    if abs(j-i) < 5:
                        Hbforce.addExclusion(Don, Acc)
                        print('Exclude', i, j)

            system.addForce(Hbforce)

        """

    ########## WCA
    if ff.wca:
        totalforcegroup += 1
        forcegroup['WCA'] = totalforcegroup
        print(f"    {totalforcegroup:2d}:    WCA")
        groupnames.append("Uwca")

        #energy_function =  'step(sig-r) * ep * ((R6 - 2)*R6 + 1);'
        #energy_function =  'WCAflag1*WCAflag2*step(sig-r) * ep * ((R6 - 2)*R6 + 1);'
        energy_function =  'select(WCAflag1*WCAflag2*step(sig-r), ep * ((R6 - 2)*R6 + 1), 0);'
        energy_function += 'R6=(sig/r)^6;'

        WCAforce = omm.CustomNonbondedForce(energy_function)
        WCAforce.addGlobalParameter('ep',  ff.wca_epsilon)
        WCAforce.addGlobalParameter('sig', ff.wca_sigma)
        WCAforce.addPerParticleParameter('WCAflag')

        for residue in topology.residues():
            #WCAforce.addParticle([])
            if residue.name[1:2] == 'D':
                WCAforce.addParticle([0,])
            else:
                WCAforce.addParticle([1,])

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
        WCAforce.setForceGroup(totalforcegroup)
        if ctrl.PBC:
            WCAforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffPeriodic)
        else:
            WCAforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffNonPeriodic)
        system.addForce(WCAforce)

    ########## Debye-Huckel
    if ctrl.ele:
        totalforcegroup += 1
        forcegroup['Ele'] = totalforcegroup
        print(f"    {totalforcegroup:2d}:    Ele")
        groupnames.append("Uele")

        DHforce = omm.CustomNonbondedForce("DHscale*Zp1*Zp2*exp(-kappa*r)/r")
        DHforce.addGlobalParameter("kappa", ele_kappa)
        DHforce.addGlobalParameter("DHscale", ele_scale)
        DHforce.addPerParticleParameter('Zp')

        for residue in topology.residues():
            if residue.name[1:2] == 'D':
                DHforce.addParticle([0,])
            elif atm_index(residue)+1 in ctrl.ele_no_charge:
                DHforce.addParticle([0,])
            else:
                DHforce.addParticle([ele_Zp,])

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
        DHforce.setForceGroup(totalforcegroup)
        if ctrl.PBC:
            DHforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffPeriodic)
        else:
            DHforce.setNonbondedMethod(omm.CustomNonbondedForce.CutoffNonPeriodic)
        system.addForce(DHforce)

    ########## NNP
    dir_torchmdnet = None
    githash_torchmdnet = None

    if ctrl.use_NNP:

        # Validate embeddings
        if len(ctrl.NNP_emblist) != nnt:
            print('Error: The length of embeddings is not consistent with the sequence.')
            print(f'       Expected {nnt}, got {len(ctrl.NNP_emblist)}')
            sys.exit(2)

        totalforcegroup += 1
        forcegroup['NNP'] = totalforcegroup
        groupnames.append("Unn")
        print(f"    {totalforcegroup:2d}:    NNP")

        """
        This section is to run OpenMM-Torch for a machine learning potential trained by TorchMD-Net.
        references:
            https://github.com/openmm/openmm-torch/blob/master/tutorials/openmm-torch-nnpops.ipynb
            https://github.com/openmm/openmm-torch/issues/135
        """
        import torch
        from openmmtorch import TorchForce
        import torchmdnet
        from torchmdnet.models.model import load_model

        try:
            dir_torchmdnet = os.path.dirname(os.path.abspath(torchmdnet.__file__))
        except:
            pass

        if dir_torchmdnet is not None:
            try:
                import subprocess
                githash_torchmdnet = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                                              cwd=dir_torchmdnet,
                                                              stderr=subprocess.DEVNULL).decode('ascii').strip()
            except:
                pass

        #torch.set_default_dtype(torch.float32)
        #torch.set_default_dtype(torch.float64)

        #model = load_model(ctrl.NNP_model, derivative=False)
        #torch.jit.script(model).save('model.pt')

        if ctrl.NNP_modelforce:
            class ForceModule(torch.nn.Module):
                def __init__(self, epochfile, embeddings):
                    super(ForceModule, self).__init__()

                    #self.model = torch.jit.load('model.pt')
                    self.model = torch.jit.script(load_model(epochfile, derivative=True))

                    self.model.eval()

                    if ctrl.device == 'CUDA':
                        self.embeddings = embeddings.cuda()
                        #self.batch = torch.arange(1).repeat_interleave(embeddings.size(0)).cuda()
                    else:
                        self.embeddings = embeddings
                        #self.batch = torch.arange(1).repeat_interleave(embeddings.size(0))
                    self.batch = None
                    self.box = None

                def forward(self, positions):
                    #positions_A = positions.to(torch.float32)*10  # nm -> angstrom
                    positions_A = positions*10  # nm -> angstrom
                    energy, force = self.model(self.embeddings, positions_A, self.batch, self.box)
                    return energy*4.184, force*41.84

        else:
            class ForceModule(torch.nn.Module):
                def __init__(self, epochfile, embeddings):
                    super(ForceModule, self).__init__()

                    self.model = torch.jit.script(load_model(epochfile, derivative=False))
                    '''derivative=False : Let OpenMM calculate force by back-propagation.'''

                    self.model.eval()

                    if ctrl.device == 'CUDA':
                        self.embeddings = embeddings.cuda()
                    else:
                        self.embeddings = embeddings

                    self.batch = None
                    self.box = None

                def forward(self, positions):
                    #positions_A = positions.to(torch.float32)*10  # nm -> angstrom
                    positions_A = positions*10  # nm -> angstrom
                    energy, _ = self.model(self.embeddings, positions_A, self.batch, self.box)
                    return energy*4.184

        embeddings = torch.tensor(ctrl.NNP_emblist, dtype=torch.long)

        module = torch.jit.script(ForceModule(ctrl.NNP_model, embeddings))
        torch_force = TorchForce(module)
        if ctrl.NNP_modelforce:
            torch_force.setOutputsForces(True)
        torch_force.setForceGroup(totalforcegroup)
        system.addForce(torch_force)

    print('')

    if ctrl.use_NNP:
        print(f"TorchMD-Net:")
        print(f"    Module directory: {dir_torchmdnet}")
        print(f"    Git hash: {githash_torchmdnet}")
        print(f'')

    ################################################
    #             Simulation set up
    ################################################

    class EnergyReporter(object):
        def __init__ (self, file, reportInterval, simulation=None, state=None):
            self._out = open(file, 'w')
            self._reportInterval = reportInterval
            self._out.write('#     1:Step    2:T        3:Ekin      4:Utotal')
                            #123456789012 123456 1234567890123 1234567890123'
            icol = 4
            if ff.bond:
                icol += 1
                self._out.write(' %13s' % f'{icol}:Ubond')
            if ff.angle:
                icol += 1
                self._out.write(' %13s' % f'{icol}:Uangl')
            if ff.angle_ReB:
                icol += 1
                self._out.write(' %13s' % f'{icol}:Uangl')
            if ff.dihexp:
                icol += 1
                self._out.write(' %13s' % f'{icol}:Udih')
            if ff.bp and ctrl.BP_model > 0:
                icol += 1
                self._out.write(' %13s' % f'{icol}:Ubp')
            if ff.wca:
                icol += 1
                self._out.write(' %13s' % f'{icol}:Uwca')
            if ctrl.ele:
                icol += 1
                self._out.write(' %13s' % f'{icol}:Uele')
            if ctrl.use_NNP:
                icol += 1
                self._out.write(' %13s' % f'{icol}:Unn')

            self._out.write("\n")

            # Output if this is the first step (not restarting)
            if simulation is not None and state is not None:
                self.report(simulation, state)
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
            #state = simulation.context.getState(getEnergy=True)  # This shouldn't be needed
            energy = state.getKineticEnergy().value_in_unit(u_kcalmol)
            self._out.write(f" {energy:13.6g}")
            energy = state.getPotentialEnergy().value_in_unit(u_kcalmol)
            self._out.write(f" {energy:13.6g}")
            for i in range(1, totalforcegroup + 1):
                state = simulation.context.getState(getEnergy=True, groups={i})
                energy = state.getPotentialEnergy().value_in_unit(u_kcalmol)
                self._out.write(f" {energy:13.6g}")
            self._out.write("\n")
            self._out.flush()

    class MyDCDReporter(app.DCDReporter):
        def __init__(self, file, reportInterval, simulation=None, state=None):
            super().__init__(file, reportInterval)
            # Output if this is the first step (not restarting)
            if simulation is not None and state is not None:
                self.report(simulation, state)

    #integrator = omm.LangevinIntegrator(ctrl.LD_temp, ctrl.LD_gamma, ctrl.LD_dt)
    integrator = omm.LangevinMiddleIntegrator(ctrl.LD_temp, ctrl.LD_gamma, ctrl.LD_dt)
    integrator.setRandomNumberSeed(ctrl.LD_seed)

    platform = None
    properties = None

    if ctrl.device is not None:
        platform = omm.Platform.getPlatformByName(ctrl.device)

    if ctrl.device == 'CUDA':
        #properties = {'Precision': 'double'}
        properties = {'Precision': 'single'}

    elif ctrl.device == 'CPU':
        if ctrl.nthreads is not None:
            properties = {'Threads': f'{ctrl.nthreads}'}

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

    simulation = app.Simulation(topology, system, integrator, platform, properties)

    platform = simulation.context.getPlatform()
    print(f'Platform selected: {platform.getName()}')
    print(f'Speed enhancement estimate: {platform.getSpeed()}')
    print(f'Properties:')
    for name in platform.getPropertyNames():
        print(f'    {name}: {platform.getPropertyValue(simulation.context, name)}')
    print()
    sys.stdout.flush()
    sys.stderr.flush()

    if ctrl.restart:
        assert (ctrl.chk_file is not None) or (ctrl.xml_file is not None)

        if ctrl.chk_file is not None:
            print(f'Loading checkpoint file: {ctrl.chk_file}\n')
            simulation.loadCheckpoint(ctrl.chk_file)

        elif ctrl.xml_file is not None:
            print(f'Loading state file: {ctrl.xml_file}\n')
            simulation.loadState(ctrl.xml_file)

    else:
        if ctrl.PBC:
            boxvector = diag(ctrl.PBC_size) * u_A
            simulation.context.setPeriodicBoxVectors(*boxvector)

        simulation.context.setPositions(positions)

        ## Write PDB before minimization
        #state = simulation.context.getState(getPositions=True)
        #app.PDBFile.writeFile(topology, state.getPositions(), open("before_minimize.pdb", "w"), keepIds=True)

        if ctrl.job_type == 'MD' and ctrl.minimization:
            print('Minimizing energy ...')
            #simulation.minimizeEnergy(ctrl.minimization_tolerance, ctrl.minimization_max_iter)
            simulation.minimizeEnergy(5.0*unit.kilojoule_per_mole/unit.nanometer, ctrl.minimization_max_iter)

            ## Write PDB after minimization
            #state = simulation.context.getState(getPositions=True)
            #app.PDBFile.writeFile(topology, state.getPositions(), open("after_minimize.pdb", "w"), keepIds=True)

        if ctrl.job_type == 'MD':
            print(f"Setting the initial velocities, T = {ctrl.temp}, seed = {ctrl.velo_seed}\n")
            simulation.context.setVelocitiesToTemperature(ctrl.temp, ctrl.velo_seed)


    ################################################
    #               MD
    ################################################
    if ctrl.job_type == 'MD':
        print('Simulation starting ...')

        # Only Nstep_next
        if ctrl.Nstep is None:
            nstep = ctrl.Nstep_next

        # Only Nstep
        elif ctrl.Nstep_next is None:
            nstep = ctrl.Nstep - simulation.currentStep

        # Both Nstep and Nstep_next
        else:
            if ctrl.Nstep < simulation.currentStep + ctrl.Nstep_next:
                nstep = ctrl.Nstep - simulation.currentStep
            else:
                nstep = ctrl.Nstep_next

        simulation.reporters.append(app.StateDataReporter(ctrl.outfile_log, ctrl.Nstep_log, 
                                    step=True, potentialEnergy=True, temperature=True, 
                                    remainingTime=True, totalSteps=nstep, separator='  '))

        if ctrl.restart: # Not recording the initial state
            simulation.reporters.append(EnergyReporter(ctrl.outfile_out, ctrl.Nstep_out))
            simulation.reporters.append(MyDCDReporter(ctrl.outfile_dcd, ctrl.Nstep_out))

        else: # Recording the initial state (step = 0)
            state = simulation.context.getState(getEnergy=True, getPositions=True)
            simulation.reporters.append(EnergyReporter(ctrl.outfile_out, ctrl.Nstep_out, simulation, state))
            simulation.reporters.append(MyDCDReporter(ctrl.outfile_dcd, ctrl.Nstep_out, simulation, state))

        simulation.reporters.append(app.CheckpointReporter(ctrl.outfile_chk, ctrl.Nstep_rst))
        #simulation.reporters.append(app.CheckpointReporter(ctrl.outfile_xml, ctrl.Nstep_rst, writeState=True))
        # State XML file will be written only at the end of the simulation.

        print(f"    Current step: {simulation.currentStep}")
        print(f"    Running for: {nstep}\n")
        sys.stdout.flush()
        sys.stderr.flush()

        t1 = time.time()
        simulation.step(nstep)
        #simulation.runForClockTime(time, checkpointFile=None, stateFile=None, checkpointInterval=None)

        t = time.time()
        totaltime = t - t0
        runtime = t - t1
        print(f"Job summary:")
        print(f"    Elapsed time: {totaltime:.1f} secs = {totaltime/3600:.2f} hrs")
        print(f"    Run time: {runtime:.1f} secs = {runtime/3600:.2f} hrs")
        print(f"    Speed: {86400*nstep/runtime:.2e} steps/day")

        simulation.saveCheckpoint(ctrl.outfile_chk)
        simulation.saveState(ctrl.outfile_xml)

    ################################################
    #            DCD / DCD_FORCE
    ################################################
    elif ctrl.job_type in ('DCD', 'DCD_FORCE'):

        from dcd import DcdFile
        dcd = DcdFile(ctrl.infile_dcd)
        dcd.open_to_read()
        dcd.read_header()

        energy_file = open(ctrl.outfile_out, 'w')
        energy_file.write('#    1:Frame      2:Utotal')
                          #123456789012 1234567890123'
        icol = 2
        if ff.bond:
            icol += 1
            energy_file.write(' %13s' % f'{icol}:Ubond')
        if ff.angle:
            icol += 1
            energy_file.write(' %13s' % f'{icol}:Uangl')
        if ff.angle_ReB:
            icol += 1
            energy_file.write(' %13s' % f'{icol}:Uangl')
        if ff.dihexp:
            icol += 1
            energy_file.write(' %13s' % f'{icol}:Udih')
        if ff.bp and ctrl.BP_model > 0:
            icol += 1
            energy_file.write(' %13s' % f'{icol}:Ubp')
        if ff.wca:
            icol += 1
            energy_file.write(' %13s' % f'{icol}:Uwca')
        if ctrl.ele:
            icol += 1
            energy_file.write(' %13s' % f'{icol}:Uele')
        if ctrl.use_NNP:
            icol += 1
            energy_file.write(' %13s' % f'{icol}:Unn')
        energy_file.write("\n")

        if ctrl.job_type == 'DCD_FORCE':
            force_files = {}
            for s in ('Bond', 'Angle', 'ReB', 'DihExp', 'BP', 'WCA', 'Ele', 'NNP'):
                if s not in forcegroup:
                    continue
                if s == 'Bond':
                    force_files[s] = open(ctrl.outfile_prefix + f'_bond.out', 'w')
                elif s in ('Angle', 'ReB'):
                    force_files[s] = open(ctrl.outfile_prefix + f'_angl.out', 'w')
                elif s in ('DihExp', ):
                    force_files[s] = open(ctrl.outfile_prefix + f'_dihe.out', 'w')
                elif s  == 'BP':
                    force_files[s] = open(ctrl.outfile_prefix + f'_bp.out', 'w')
                elif s  == 'Ele':
                    force_files[s] = open(ctrl.outfile_prefix + f'_bp.out', 'w')
                elif s  == 'WCA':
                    force_files[s] = open(ctrl.outfile_prefix + f'_exv.out', 'w')
                elif s  == 'NNP':
                    force_files[s] = open(ctrl.outfile_prefix + f'_nn.out', 'w')

        iframe = 0
        while dcd.has_more_data():
            iframe += 1
            positions = dcd.read_onestep_omm()
            simulation.context.setPositions(positions)

            energies = {}
            for s in ('Bond', 'Angle', 'ReB', 'DihExp', 'BP', 'WCA', 'Ele', 'NNP'):
                if s not in forcegroup:
                    continue

                i = forcegroup[s]
                #print(system.getForce(i).getName())

                if ctrl.job_type == 'DCD':
                    state = simulation.context.getState(getEnergy=True, groups={i})
                elif ctrl.job_type == 'DCD_FORCE':
                    state = simulation.context.getState(getEnergy=True, getForces=True, groups={i})

                energies[s] = state.getPotentialEnergy().value_in_unit(u_kcalmol)

                if ctrl.job_type == 'DCD_FORCE':
                    forces = state.getForces()
                    for force in forces:
                        x = force[0].value_in_unit(u_kcalmol/u_A)
                        y = force[1].value_in_unit(u_kcalmol/u_A)
                        z = force[2].value_in_unit(u_kcalmol/u_A)
                        force_files[s].write(f' {x:10.4e} {y:10.4e} {z:10.4e}')
                    force_files[s].write('\n')

            energy_file.write(f"{iframe:12d}")
            energy_file.write(f" {sum(energies.values()):13.6g}")  # Utotal
            for s in ('Bond', 'Angle', 'ReB', 'DihExp', 'BP', 'WCA', 'Ele', 'NNP'):
                if s not in forcegroup:
                    continue
                energy_file.write(f" {energies[s]:13.6g}")
            energy_file.write("\n")
            #energy_file.flush()

        energy_file.close()

        if ctrl.job_type == 'DCD_FORCE':
            for f in force_files.values():
                f.close()

if __name__ == "__main__":
    main()
