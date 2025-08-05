################################################
#         Utility functions
################################################

import itertools as it

def prev_and_next(iterable):
    prevs, items, nexts = it.tee(iterable, 3)
    prevs = it.chain([None], prevs)
    nexts = it.chain(it.islice(nexts, 1, None), [None])
    return zip(prevs, items, nexts)
    # For N, this will geneerate [None,0,1], [0,1,2], ..., [N-3,N-2,N-1], [N-2,N-1,None]

def fours(iterable):
    a, b, c, d = it.tee(iterable, 4)
    next(b, None)
    next(c, None); next(c, None)
    next(d, None); next(d, None); next(d, None)
    return zip(a, b, c, d)
    # For N, this will geneerate [0,1,2,3], [1,2,3,4], ..., [N-4,N-3,N-2,N-1] (no None)

def atm_index(res):
    #return res.atoms[0].index
    for atom in res.atoms():
        return atom.index

def get_atom(res):
    for atom in res.atoms():
        return atom

