from AlgoritmoGenetico import GeneticAlgorithm
from utils import train_genetic_algorithm

ag = GeneticAlgorithm(20, 61)
n_gens = 200
train_genetic_algorithm(ag, n_gens)