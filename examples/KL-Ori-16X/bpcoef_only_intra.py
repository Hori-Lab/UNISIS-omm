#!/usr/bin/env python

fout = open('../KL-Ori-16X_onlyintra.bpcoef', 'w')

Nnt_per_chain = 63
Nchain_per_mol = 4
Nmol = 16
#Nchain = Nchain_per_mol * Nmol

for l in open('./KL-Ori-16X.bpcoef'):
    lsp = l.split()
    imp = int(lsp[1])
    jmp = int(lsp[2])
    bp6 = lsp[3]
    u0str = lsp[4]

    ichain = (imp-1) // Nnt_per_chain + 1
    jchain = (jmp-1) // Nnt_per_chain + 1

    imol = (ichain-1) // Nchain_per_mol + 1
    jmol = (jchain-1) // Nchain_per_mol + 1

    print (f"{imp:6d} {jmp:6d} {ichain:4d} {jchain:4d} {imol:4d} {jmol:4d}")
    # only intra molecule (within one X)
    if imol == jmol:
        fout.write(l)

fout.close()



