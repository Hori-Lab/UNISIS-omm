rm -f md.*
rm -f run1/*
rm -f sis_dcd/*
rm -f omm_dcd/*
rm -f run2_xml/*
rm -f run2_xhk/*
rm -f run_long/*

mkdir -p run1
../../sis-torch.py input_T2HP_go.toml 1> run1/out 2> run1/err
mv md.* run1

mkdir -p sis_dcd
~/sis/sis input_sis.toml 1> sis_dcd/out 2> sis_dcd/err
mv md.* sis_dcd

mkdir -p omm_dcd
../../sis-torch.py input_T2HP_go_dcd.toml 1> omm_dcd/out 2> omm_dcd/err
mv md.* omm_dcd

# Restart by the state XML file
mkdir -p run2_xml
../../sis-torch.py input_T2HP_go.toml -x ./run1/md.xml 1> run2_xml/out 2> run2_xml/err
mv md.* run2_xml

# Restart by the checkpoint file
mkdir -p run2_chk
../../sis-torch.py input_T2HP_go.toml -c ./run1/md.chk 1> run2_chk/out 2> run2_chk/err
mv md.* run2_chk

mkdir -p run_long
../../sis-torch.py input_T2HP_go_long.toml 1> run_long/out 2> run_long/err
mv md.* run_long
