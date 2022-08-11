from AlgoritmoGenetico import GeneticAlgorithm

ag = GeneticAlgorithm(5, 5)

print(ag.last_gen)

print(f"Geração 0:")
print(f"{ag.last_gen[ag.last_gen_best_fit_pos]} ---> {ag.last_gen_best_fit}")

i = 0
while(ag.no_changes_best_fit < 500):
    ag.new_generation()
    i += 1

    if (ag.no_changes_best_fit == 0):
        print(f"Geração {i}:")
        print(f"{ag.last_gen[ag.last_gen_best_fit_pos]} ---> {ag.last_gen_best_fit}")

print(f"Fim do algoritmo genético. Número de gerações: {i}")