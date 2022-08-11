from typing import List
import numpy as np
import random


class GeneticAlgorithm:
    def __init__(
        self, number_of_individuals, number_genes,
        crossover_prob=0.8, mutation_prob=0.03
    ):
        self.n_ind = number_of_individuals
        self.n_genes = number_genes
        self.p_crossover = crossover_prob
        self.p_mutation = mutation_prob

        self.first_generation()

    def obj_f(self, ind):
        return sum(ind)

    def get_roulette_wheel(self):
        arr = self.last_gen_fitness / \
            np.sum(self.last_gen_fitness)
        acumulador = []
        anterior = 0
        for i in range(self.n_ind):
            atual = round(arr[i]*10000000)
            nova_pos = anterior + atual
            acumulador.append(nova_pos)
            anterior = nova_pos
        return acumulador

    def get_last_gen_fitness(self):
        fitness_array = []
        for ind in self.last_gen:
            fitness_array.append(self.obj_f(ind))

        self.last_gen_fitness = fitness_array

        self.roulette_wheel = self.get_roulette_wheel()

    def first_generation(self):
        gen_1 = np.random.rand(self.n_ind, self.n_genes)
        self.last_gen = np.array(gen_1)
        self.get_last_gen_fitness()

    def random_parents(self, n_parents=2):
        limite = self.roulette_wheel[self.n_ind - 1]

        parent_array = []
        for i in range(n_parents):
            new_parent = random.randint(0, limite)

            parent_position = 0
            while (new_parent > self.roulette_wheel[parent_position]):
                parent_position += 1

            parent_array.append(parent_position)
        return parent_array

    # def apply_elitism(self):

    def new_generation(self):
        new_gen = []
        for i in range(self.n_ind):
            parents = self.random_parents()

            p = random.random()

            ''' crossover '''
            if (p <= self.p_crossover):
                crossover = random.randint(1, self.n_genes-1)
                child = np.concatenate(
                    (self.last_gen[parents[0]][0:crossover], self.last_gen[parents[1]][crossover:self.n_genes]), axis=None)
            else:
                y = random.randint(0, 1)
                child = self.last_gen[parents[y]][:]

            ''' mutação '''
            for j in range(self.n_genes):
                p = random.random()
                if (p <= self.p_mutation):
                    random_gene = np.random.rand()
                    child[j] = random_gene

            new_gen.append(child)

        self.last_gen = new_gen
        # self.apply_elitism()
        '''Vou precisar alterar a função que calcula fitness pois do jeito atual ela sobrescreve 
        o fitness da geração anterior com o da nova geração. No entanto, para aplicar elitismo, 
        vou precisar ter os dois fitness ao mesmo tempo.'''
        self.get_last_gen_fitness()
