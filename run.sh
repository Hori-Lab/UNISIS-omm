#./droplet.py -f "(CAG)31" -C 20 -x 100 -n 1000000 -t md.dcd -o md.out -H 1.67
#./droplet.py -f "(CAG)31" -C 20 -x 1 -n 10 -t md.dcd -o md.out -H 1.67 -i test.pdb


./droplet_tracer.py --cgpdb CAG47_200_fit_droplet0_last_cls0.pdb -x 100 -n 1000000 -t md.dcd -o md.out -H 1.67 \
                    --tp_eps 2.0 \
                    --tp_sig 10.0

