from RedeNeuralLuiz import NeuralNetwork
from network import Network
from utils import *

weights = [random_2D_list(2, 7), random_2D_list(7, 5)]
biases = [random_2D_list(1, 7), random_2D_list(1, 5)]

nn = Network(2, 7, 5, weights, biases)
n_gens = 3_000
train_neural_network(nn, n_gens)