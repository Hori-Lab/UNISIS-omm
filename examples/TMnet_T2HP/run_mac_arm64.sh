rm -f md.*
rm -f run1/*
rm -f run2/*

mkdir -p run1
../../sis-torch.py --tmyaml simulate.yaml --ff htv23_nnp_dihexp.ff 1> run1/out 2> run1/err
mv md.* run1

mkdir -p run2
../../sis-torch.py --tmyaml simulate.yaml --ff htv23_nnp_dihexp.ff -r run1/md.rst 1> run2/out 2> run2/err
mv md.* run2

mkdir -p run1_dcd
../../sis-torch.py --tmyaml dcd.yaml --ff htv23_nnp_dihexp.ff 1> dcd.out 2> dcd.err
mv dcd.out dcd.err force.out force_*.out run1_dcd

