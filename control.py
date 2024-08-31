from dataclasses import dataclass, field
from typing import List, Dict
from simtk import unit
from simtk.unit import Quantity

################################################
#          Control object
################################################
@dataclass
class Control:    ### structure to group all simulation parameter
    device: str = ''
    restart: bool = False
    restart_file: str = None

    xml:         str = None
    ff:          str = None
    infile_pdb:  str = None
    infile_xyz:  str = None
    infile_bpcoef:str = None
    outfile_log: str = './md.log'
    outfile_out: str = './md.out'
    outfile_dcd: str = './md.dcd'
    outfile_rst: str = './md.rst'

    Nstep: int        = 10
    Nstep_out:  int   = 1
    Nstep_log:  int   = 1000000
    Nstep_rst:  int   = 1000000

    temp: Quantity     = field(default_factory=lambda: Quantity(300.0, unit.kelvin))
    velo_seed: int = 0
    LD_temp: Quantity  = field(default_factory=lambda: Quantity(300.0, unit.kelvin))
    LD_gamma: Quantity = field(default_factory=lambda: Quantity(0.5, unit.picosecond**(-1)))
    LD_dt: Quantity    = field(default_factory=lambda: Quantity(50, unit.femtoseconds))
    LD_seed: int = 0

    minimization: bool = False
    minimization_max_iter: int = None
    minimization_tolerance: float = 1.0

    PBC: bool = False
    PBC_size: List = field(default_factory=lambda: [0.0, 0.0, 0.0])

    BP_model: int = 5
    BP_min_loop: int = 3

    ele: bool = False
    ele_ionic_strength: float = 0.15
    ele_cutoff_type: int = 1
    ele_cutoff_factor: float = 50.0
    ele_no_charge: List = field(default_factory=lambda: [])
    ele_length_per_charge: Quantity = field(default_factory=lambda: Quantity(4.38178046, unit.angstrom))
    ele_exclusions: Dict = field(default_factory=lambda: {'1-2': True, '1-3': True})

    use_NNP: bool = False
    NNP_model: str = ''
    NNP_emblist: List = None
        #field(default_factory=lambda: 
        #   [5,2,3,4,1,4,2,1,2,2,4,3,1,4,1,3,1,4,3,2,4,3,1,4,1,2,3,1,5])
    NNP_modelforce: bool = True

    def __str__(self):
        return (f"Control:\n"
              + f"    device: {self.device}\n"
              + f"    restart: {self.restart}\n"
              + f"    restart_file: {self.restart_file}\n"
              + f"    xml: {self.xml}\n"
              + f"    ff: {self.ff}\n"
              + f"    infile_pdb: {self.infile_pdb}\n"
              + f"    infile_xyz: {self.infile_xyz}\n"
              + f"    infile_bpcoef: {self.infile_bpcoef}\n"
              + f"    outfile_log: {self.outfile_log}\n"
              + f"    outfile_out: {self.outfile_out}\n"
              + f"    outfile_dcd: {self.outfile_dcd}\n"
              + f"    outfile_rst: {self.outfile_rst}\n"
              + f"    Nstep: {self.Nstep}\n"
              + f"    Nstep_out: {self.Nstep_out}\n"
              + f"    Nstep_log: {self.Nstep_log}\n"
              + f"    Nstep_rst: {self.Nstep_rst}\n"
              + f"    temp: {self.temp}\n"
              + f"    velo_seed: {self.velo_seed}\n"
              + f"    LD_temp: {self.LD_temp}\n"
              + f"    LD_gamma: {self.LD_gamma}\n"
              + f"    LD_dt: {self.LD_dt}\n"
              + f"    LD_seed: {self.LD_seed}\n"
              + f"    minimization: {self.minimization}\n"
              + f"    minimization_max_iter: {self.minimization_max_iter}\n"
              + f"    minimization_tolerance: {self.minimization_tolerance}\n"
              + f"    PBC: {self.PBC}\n"
              + f"    PBC_size: {self.PBC_size}\n"
              + f"    BP_model: {self.BP_model}\n"
              + f"    BP_min_loop: {self.BP_min_loop}\n"
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
              + f"    NNP_modelforce: {self.NNP_modelforce}\n"
                )

    def load_toml(self, tm):

        if 'xml' in tm['Files']['In']:
            self.xml = tm['Files']['In']['xml']
        if 'ff' in tm['Files']['In']:
            self.ff = tm['Files']['In']['ff']

        self.fasta = tm['Files']['In']['fasta']
        if 'pdb_ini' in tm['Files']['In']:
            self.infile_pdb   = tm['Files']['In']['pdb_ini']
        if 'xyz_ini' in tm['Files']['In']:
            self.infile_xyz   = tm['Files']['In']['xyz_ini']
        if 'bpcoef' in tm['Files']['In']:
            self.infile_bpcoef   = tm['Files']['In']['bpcoef']
        self.outfile_dcd  = tm['Files']['Out']['prefix'] + '.dcd'
        self.outfile_log  = tm['Files']['Out']['prefix'] + '.log'
        self.outfile_out  = tm['Files']['Out']['prefix'] + '.out'
        self.outfile_rst  = tm['Files']['Out']['prefix'] + '.rst'

        self.velo_seed    = tm['Condition']['rng_seed']
        self.temp         = tm['Condition']['tempK'] * unit.kelvin
        self.LD_temp      = tm['Condition']['tempK'] * unit.kelvin

        self.Nstep        = tm['MD']['nstep']
        self.Nstep_out    = tm['MD']['nstep_save']
        if 'nstep_save_rst' in tm['MD']:
            self.Nstep_rst    = tm['MD']['nstep_save_rst']
        if 'friction' in tm['MD']:
            self.LD_gamma = tm['MD']['friction'] / unit.picosecond
        elif 'viscosity_Pas' in tm['MD']:
            pass

        if 'dt_fs' in tm['MD']:
            self.LD_dt = tm['MD']['dt_fs'] * unit.femtoseconds
        elif 'dt' in tm['MD']:
            self.LD_dt = tm['MD']['dt'] * 50 * unit.femtoseconds

        self.LD_seed      = tm['Condition']['rng_seed']

        if 'Electrostatic' in tm:
            self.ele = True
            self.ele_ionic_strength = tm['Electrostatic']['ionic_strength']
            self.ele_cutoff_type    = tm['Electrostatic']['cutoff_type']
            self.ele_cutoff_factor  = tm['Electrostatic']['cutoff']
            if 'no_charge' in tm['Electrostatic']:
                self.ele_no_charge      = tm['Electrostatic']['no_charge']
            self.ele_length_per_charge = tm['Electrostatic']['length_per_charge'] * unit.angstrom
            if 'exclude_covalent_bond_pairs' in tm['Electrostatic']:
                self.ele_exclusions['1-2'] = tm['Electrostatic']['exclude_covalent_bond_pairs']

        if 'NNP' in tm:
            self.use_NNP      = True
            self.NNP_model    = tm['Files']['In']['TMnet_ckpt']
            self.NNP_modelforce = tm['NNP']['model_force']
            #self.NNP_emblist  = tm['external']['embeddings']

        if 'PBC_box' in tm:
            self.PBC = True
            self.PBC_size = tm['PBC_box']['size']

        if 'Progress' in tm:
            if 'step' in tm['Progress']:
                self.Nstep_log    = tm['Progress']['step']

        if 'minimization' in tm:
            if self.use_NNP:
                print ('Warning: minimization cannot be done for NNP. Continue without minimization')
            else:
                self.minimization = True
                if 'max_iteration' in tm['minimization']:
                    self.minimization_max_iter = tm['minimization']['max_iteration']
                if 'tolerance' in tm['minimization']:
                    self.minimization_tolerance = tm['minimization']['tolerance']

    def load_TorchMDyaml(self, yml):
        self.set_from_TorchMDyaml(tmyaml)
        self.infile_pdb   = yml['structure']
        self.Nstep        = yml['steps']
        self.Nstep_out    = yml['output_period']
        self.Nstep_log    = yml['save_period']
        self.Nstep_rst    = yml['save_period']
        self.outfile_dcd  = yml['output'] + '.dcd'
        self.outfile_log  = yml['output'] + '.log'
        self.outfile_out  = yml['output'] + '.out'
        self.outfile_rst  = yml['output'] + '.rst'
        self.temp         = yml['temperature'] * unit.kelvin
        self.LD_temp      = yml['langevin_temperature'] * unit.kelvin
        self.LD_gamma     = yml['langevin_gamma'] / unit.picosecond
        self.LD_dt        = yml['timestep'] * unit.femtoseconds
        self.NNP_model    = yml['external']['file']
        self.NNP_emblist  = yml['external']['embeddings']
        self.use_NNP      = True
