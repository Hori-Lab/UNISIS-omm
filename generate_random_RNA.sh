#!/bin/bash

#SBATCH --account=su006-044
#SBATCH --job-name=R_C0.200
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=42
#SBATCH --gres=gpu:ampere_a100:1
#SBATCH --time=48:00:00
#SBATCH -o random_RNA.sout
#SBATCH -e random_RNA.serr

echo "Running on `hostname`"

echo "CUDA visible devices ${CUDA_VISIBLE_DEVICES}"


###i=0   1   2   3    4    5    6   7     8
Ns=(64 128 256 512 1024 2048 4096 8192 16384)
steps=(1000000 2000000 4000000 8000000 16000000 32000000 64000000 128000000 256000000)
saves=(10000 20000 40000 80000 100000 100000 100000 100000 100000)

for ((i=0;i<${#Ns[@]};++i)); do
    N=${Ns[i]}
    step=${steps[i]}
    save=${saves[i]}

    echo "############### N = $N, step=$step, save=$save"
    echo "Start at `date`"

    if [ $i -eq 0 ]; then
        initial="../sismm/linear_64.xyz"
    else
        initial="N${Ns[$((i-1))]}_N$N.xyz"
    fi

    echo "pre-run without ReB & dihexp"
    time ../sismm/sisrna_BPfree.py --inixyz $initial \
                 --step 100000 --temperature 300.0 \
                 --angle \
                 --ionic_strength 0.200 --cutoff 200.0 \
                 --traj N${N}_pre.dcd --energy N${N}_pre.energy --output N${N}_pre.out --frequency 10000 \
                 --finalxyz N${N}_pre.xyz --res_file N${N}_pre.chk \
                 1> N${N}_pre.log 2> N${N}_pre.err

    echo "main run"
    time ../sismm/sisrna_BPfree.py --inixyz N${N}_pre.xyz \
                 --step $step --temperature 300.0 \
                 --ReB --dihexp \
                 --ionic_strength 0.200 --cutoff 200.0 \
                 --traj N${N}.dcd --energy N${N}.energy --output N${N}.out --frequency $save \
                 --finalxyz N${N}.xyz --res_file N${N}.chk \
                 1> N${N}.log 2> N${N}.err

    if [ $N -ne 8 ]; then
        echo "fine graining"
        ../sismm/fine_graining.py N${N}.xyz N${N}_N${Ns[$((i+1))]}.xyz
    fi

    echo "Finished at `date`"
done

