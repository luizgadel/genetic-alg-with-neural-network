from AlgoritmoGenetico import GeneticAlgorithm

ag = GeneticAlgorithm(5, 5)

print(ag.last_gen)

print(f"Melhor fitness da geração 0:")
print(max(ag.last_gen_fitness))

for i in range(1, 500):
    ag.new_generation()
    print(f"Melhor fitness da geração {i}:")
    print(max(ag.last_gen_fitness))