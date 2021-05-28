#./droplet.py -f "(CAG)31" -C 20 -x 100 -n 1000000 -t md.dcd -o md.out -H 1.67
#./droplet.py -f "(CAG)31" -C 20 -x 1 -n 10 -t md.dcd -o md.out -H 1.67 -i test.pdb


#./droplet_tracer.py --cgpdb CAG31_400_last.pdb -x 10 -n 1000 -t md.dcd -o md.out -H 1.67 \
#                    --tp_eps 2.0 \
#                    --tp_sig 30.0 \
#                    --tp_initrange 0.1\
#                    --tp_terminate \
#                     1> out 2> err
#                    #--platform 'CUDA'\
#                    #--CUDAdevice '0'\

./droplet.py -f "(CAG)47" -C 200 \
             -x 10000 -n 100000000 \
             -t md.dcd -o md.out \
             -H 1.67 \
             -i CAG47_200_md30_last6704.pdb \
             -K 150.0 --cutoff 50.0 \
             1> out 2> err
