from AlgoritmoGenetico import GeneticAlgorithm
from utils import *

ag = GeneticAlgorithm(20, 49)

print(ag.last_gen)

print(f"Geração 0:")
last_gen_best_ind = ag.last_gen[ag.last_gen_best_fit_pos]
print(f"{last_gen_best_ind} ---> {ag.last_gen_best_fit}")

i = 0
while(ag.no_changes_best_fit < 5000):
    ag.new_generation()
    i += 1

    if (ag.no_changes_best_fit == 0):
        print(f"Geração {i}:")
        last_gen_best_ind = ag.last_gen[ag.last_gen_best_fit_pos]
        print(f"{last_gen_best_ind[:5]} ---> {1 / ag.last_gen_best_fit}")

print(f"Fim do algoritmo genético. Número de gerações: {i}")
print(f"Melhor fitness:\n {ag.last_gen_fitness[:14]} \n {ag.last_gen_fitness[14:]}")

'''
Comparar tabela de perda do melhor indivíduo com o obtido pelo backpropag
'''