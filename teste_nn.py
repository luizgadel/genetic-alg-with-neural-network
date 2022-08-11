import numpy as np
from RedeNeuralLuiz import NeuralNetwork
from RedeNeuralCaio import NNC
from utils import *
import time

starting_time = time.time()

nn = NeuralNetwork(2, 7, 5, random_2D_list(2, 7), random_2D_list(7, 5))
n_gens = 1000000
train_neural_network(nn, n_gens)

ending_time = time.time()
elapsed_time = ending_time - starting_time

print(f"Número total de gerações: {n_gens}")
print('Tempo de execução:', round(elapsed_time, 3), 'segundos.\n')

print("MSE: " + str(compute_error(nn.A_2)))
dif = abs(train_Y - nn.A_2)
print("Diferenças:")
print(str(np.round(dif, 4)))

print("\nFim da rede neural.")
