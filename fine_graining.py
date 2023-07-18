#!/usr/bin/env python

import sys
import numpy as np
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('input_xyz', help='input xyz file')
parser.add_argument('output_xyz', help='output xyz file')

args = parser.parse_args()

n = 0
seq = []
xyz = []

for il, l in enumerate(open(args.input_xyz)):
    if il == 0:
        n = int(l)
    elif il == 1:
        continue
    else:
        lsp = l.split()
        seq.append(lsp[0])
        x = float(lsp[1])
        y = float(lsp[2])
        z = float(lsp[3])
        xyz.append(np.array((x,y,z)))

if n != len(xyz):
    print('Error: n /= len(xyz)')
    sys.exit(2)


######### Generate new coordinates ################
xyz_new = []
seq_new = []

# The first bead
xyz_new.append(xyz[0] - 0.25 * (xyz[1] - xyz[0]))
seq_new.append(seq[0])

# all the middle beads
for i in range(n-1):
    d = xyz[i+1] - xyz[i]
    xyz_new.append(xyz[i] + 0.25*d)
    xyz_new.append(xyz[i] + 0.75*d)
    seq_new.append(seq[i])
    seq_new.append(seq[i+1])

# The last bead
xyz_new.append(xyz[n-1] + 0.25 * (xyz[n-1] - xyz[n-2]))
seq_new.append(seq[n-1])
###################################################


# Enlarge the new coordinates
for i in range(2*n):
    xyz_new[i] *= 2.0

# COM of the original
com = np.array([0.0, 0.0, 0.0])
for i in range(n):
    com += xyz[i]
com /= n

# COM of the new
com_new = np.array([0.0, 0.0, 0.0])
for i in range(2*n):
    com_new += xyz_new[i]
com_new /= 2*n

# Move the COM of the new one to the original COM
for i in range(2*n):
    xyz_new[i] = xyz_new[i] - com_new + com

# Write
fout = open(args.output_xyz, 'w')
fout.write('%i\n' % (2 * n,))
fout.write('\n')

for i in range(2*n):
    fout.write(seq_new[i])
    fout.write("  %8.3f  %8.3f  %8.3f\n" % tuple(xyz_new[i]))

fout.close()
