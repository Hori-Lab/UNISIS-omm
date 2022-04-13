

output = open('5SAnnealing.temp', 'w')
starttemp = 380.15
Nstep = 100000000
temp = 0
annealstep = 1
annealing = True
s = 0

for i in range(0, 80, 10):
   
    temp = starttemp - i
   
    g = 0
    step = i * 1000000 
    print('i', i)
    print('step', step)
    
    output.write(str(step) + ' ' +str(temp) + '\n')
        
