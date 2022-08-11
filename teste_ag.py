from math import trunc
from AlgoritmoGenetico import GeneticAlgorithm
from utils import *
import time

starting_time = time.time()
starting_cpu_time = time.process_time()

ag = GeneticAlgorithm(20, 49)
i = 0
while(ag.no_changes_best_fit < 5000):
    ag.new_generation()
    i += 1

    '''
    nn = ag.last_gen_best_nn

    if (ag.no_changes_best_fit == 0):
        print(f"Gen {i}: {round(compute_error(nn.feed_forward(train_X)), 4)}")
    elif (i % 1000 == 0): 
        print(f"Gen {i}: ...")
'''

ending_time = time.time()
elapsed_time = ending_time - starting_time

print(f"Número total de gerações: {i}")
print('Tempo de execução:', round(elapsed_time, 3), 'segundos.\n')

nn = ag.last_gen_best_nn
print("MSE: " + str(compute_error(ag.last_gen_best_nn.A_2)))
dif = abs(train_Y - nn.A_2)
print("Diferenças:")
print(str(np.round(dif, 4)))

print("\nFim do algoritmo genético.")