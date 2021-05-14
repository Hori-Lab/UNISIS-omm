#!/usr/bin/env python

import os
import subprocess

n_trials = (50, 50, 50, 50, 20, 20)
sigmas = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
step_saves = (1000, 1000, 1000, 1000, 1000, 2000)

for sigma, n_trial, step_save in zip(sigmas, n_trials, step_saves):

    for n in range(n_trial):

        dirname = f"CAG31_400_last_sigma{sigma:03.0f}_{n:03d}"
    
        if not os.path.isdir(dirname):
            os.mkdir(dirname)
    
        os.chdir(dirname)

        if not os.path.exists('rna_cg_tracer.xml'):
            os.symlink('../rna_cg_tracer.xml', 'rna_cg_tracer.xml')
    
        cmd = ['../droplet_tracer.py', '--cgpdb', '../CAG31_400_last.pdb',
               '-x', f"{step_save:d}",
               '-n', '5000000',
               '-t', 'md.dcd',
               '-o', 'md.out',
               '-H', '1.67',
               '--tp_eps', '2.0',
               '--tp_sig', f"{sigma:4.1f}",
               '--tp_initrange', '0.1',
               '--tp_terminate']

               # Augusta GPU
               #'--platform', 'CUDA',
               #'--CUDAdevice', "'%s'" % os.environ['CUDA_VISIBLE_DEVICES'],
    
        fout = open('out', 'w')
        ferr = open('err', 'w')
        subprocess.call(cmd, stdout=fout, stderr=ferr)
        fout.close()
        ferr.close()
    
        os.chdir("../")
    
