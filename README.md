# OpenMM scripts to run SIS RNA simulations

## OpenMM environment

OpenMM can be installed by `mamba`/`conda`.

```
(torchmd) $ mamba install openmm
```

It is also pre-installed on the Pharmacy HPC.

```
[bluto:] $ module load openmm
```

## How to run simulations with a neural-network potential (NNP)

### Install OpenMM-Torch

To use NNP trained by TorchMD-net, we use [OpenMM-Torch](https://github.com/openmm/openmm-torch).

Again, this package can be easily installed by `mamba`.

```
(torchmd) $ mamba install openmm-torch
```

### Simulation set up

To run simulations, you need
- model file (e.g. `epoch=299-val_loss...`)
- force field file (e.g. `htv23_nnp_dihexp.ff`)
- simulation input YAML file (e.g. `simulate.yamll`)
- PDB file for the initial structure (e.g. `T2HP_unfolded.pdb`)

Example set up are in [examples/TMnet_T2HP/](https://github.com/Hori-Lab/sismm/tree/main/examples/TMnet_T2HP).

Note that, in the PDB file, residue names have to be `RA`, `RU`, `RC`, `RG`, and `RD` (terminal dummy).

For the input YAML file, you can use almost the same input file as for the TorchMD input. You just have to add the following line for the initial structure PDB (not DCD).

```
structure: ./T2HP_unfolded.pd
```

Relevant lines for the OpenMM script are:

```
timestep: 50
langevin_gamma: 0.5
langevin_temperature: 300
temperature: 300
output: md
output_period: 1000
save_period: 1000
seed: 10001
steps: 5000000
structure: ./T2HP_unfolded.pdb
external:
  embeddings: [5, 2, 3, 4, 1, 4, 2, 1, 2, 2, 4, 3, 1, 4, 1, 3, 1, 4, 3, 2, 4, 3, 1, 4, 1, 2, 3, 1, 5]
  file: ./epoch=213-val_loss=0.0335-test_loss=0.0690.ckpt
```

All other lines will be ignored in OpenMM job. (See `# ignored` comments in [simulate.yaml](https://github.com/Hori-Lab/sismm/tree/main/examples/TMnet_T2HP/simulate.yaml).

### Running the simulations

Submit your job by the following command.

(For GPU)

```
sis-torch.py  --tmyaml simulate.yaml --ff htv23_nnp_dihexp.ff --cuda
```

(Otherwise)

```
sis-torch.py  --tmyaml simulate.yaml --ff htv23_nnp_dihexp.ff
```

An example HTCondor script is [condor.txt](https://github.com/Hori-Lab/sismm/tree/main/examples/TMnet_T2HP/condor.txt).
