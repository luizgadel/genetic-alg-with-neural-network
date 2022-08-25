from AlgoritmoGenetico import GeneticAlgorithm
from utils import *
import time

starting_time = time.time()

ag = GeneticAlgorithm(20, 61)
n_gens = 10000
for i in range(n_gens):
    ag.new_generation()

ending_time = time.time()
elapsed_time = ending_time - starting_time

print(f"Número total de gerações: {n_gens}")
print('Tempo de execução:', round(elapsed_time, 3), 'segundos.\n')

nn = ag.last_gen_best_nn
print("MSE: " + str(compute_error(nn.A_2)))
dif = abs(train_Y - nn.A_2)
print("Diferenças:")
print(str(np.round(dif, 4)))

print("\nFim do algoritmo genético.")