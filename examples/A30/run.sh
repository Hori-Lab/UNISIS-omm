rm -f md.*
rm -f run1/*
rm -f run2/*

mkdir -p run1
../../sis-torch.py input_A30.toml 1> run1/out 2> run1/err
mv md.* run1

mkdir -p run2
../../sis-torch.py input_A30.toml -r run1/md.rst 1> run2/out 2> run2/err
mv md.* run2
