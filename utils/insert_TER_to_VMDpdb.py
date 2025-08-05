#!/usr/bin/env python

import sys

if len(sys.argv) != 4:
    print("Usage: SCRIPT [input pdb] [# repeat] [output pdb]")
    sys.exit(2)

n_repeat = int(sys.argv[2])

f = open(sys.argv[-1],'w')

count = 0
for l in open(sys.argv[1]):

    f.write(l)

    if l[0:4] == "ATOM":
        count += 1
        if (count == 3*n_repeat):
            f.write("TER\n")
            count = 0

f.close()
