rm -f md.*
rm -f run1/*
rm -f run2/*

mkdir -p run1
../../sis-torch.py input_T2HP_go.toml 1> run1/out 2> run1/err
mv md.* run1

mkdir -p sis_dcd
~/sis/sis input_sis.toml 1> sis_dcd/out 2> sis_dcd/err
mv md.* sis_dcd
