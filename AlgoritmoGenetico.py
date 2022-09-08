import numpy as np
import random as rand
from utils import *
from RedeNeuralLuiz import NeuralNetwork
from network import Network


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

    def objective_function(self, ind):
        il_size = 2
        hl_size = 7
        ol_size = 5

        w1_size = il_size * hl_size
        wb1_size = w1_size + hl_size
        w2_size = hl_size * ol_size

        wb1 = np.array(ind[:wb1_size])
        wb2 = np.array(ind[wb1_size:])
        weights_1, biases_1 = split_weights_and_biases(wb1, il_size, hl_size)
        weights_2, biases_2 = split_weights_and_biases(wb2, hl_size, ol_size)

        weights = [weights_1, weights_2]
        biases = [biases_1, biases_2]

        nn = Network(il_size, hl_size, ol_size, weights, biases)
        nn.update_mini_batch(train_X, train_Y, 1)

        return (1 / nn.cost()), nn

    def get_gen_fitness(self, gen):
        fitness_array = []
        nn_array = []
        for ind in gen:
            fitness, nn = self.objective_function(ind)
            fitness_array.append(fitness)
            nn_array.append(nn)

        return fitness_array, nn_array

    def first_generation(self):
        gen_1 = np.random.randn(self.n_ind, self.n_genes)
        gen_1_fitness, gen_1_nn = self.get_gen_fitness(gen_1)

        self.last_gen = np.array(gen_1)
        self.last_gen_fitness = gen_1_fitness
        self.last_gen_best_fit_pos = np.argmax(gen_1_fitness)
        self.last_gen_best_fit = gen_1_fitness[self.last_gen_best_fit_pos]
        self.last_gen_best_nn = gen_1_nn[self.last_gen_best_fit_pos]
        self.no_changes_best_fit = 0

    def random_parents(self, n_parents=2):
        roulette_wheel = get_roulette_wheel(self.last_gen_fitness)
        limite = max(roulette_wheel)

        parent_array = []
        for i in range(n_parents):
            new_parent = rand.randint(0, limite)

            parent_position = 0
            while (new_parent > roulette_wheel[parent_position]):
                parent_position += 1

            parent_array.append(parent_position)
        return parent_array

    def apply_elitism(self, gen, gen_fitness):
        # descobre a posição do indivíduo com o pior fitness da nova geração
        worst_fit_pos = np.argmin(gen_fitness)

        # substitui esse indivíduo pelo melhor da geração passada
        gen[worst_fit_pos] = self.last_gen[self.last_gen_best_fit_pos]
        # atualiza o fitness do novo indivíduo
        gen_fitness[worst_fit_pos] = self.last_gen_best_fit

    def new_generation(self):
        new_gen = []
        for i in range(self.n_ind):
            parents = self.random_parents()

            p = np.random.rand()

            ''' crossover '''
            if (p <= self.p_crossover):
                crossover = rand.randint(1, self.n_genes-1)
                child = np.concatenate(
                    (self.last_gen[parents[0]][:crossover], self.last_gen[parents[1]][crossover:]), axis=None)
            else:
                y = rand.randint(0, 1)
                child = self.last_gen[parents[y]][:]

            ''' mutação '''
            for j in range(self.n_genes):
                p = np.random.rand()
                if (p <= self.p_mutation):
                    random_gene = np.random.randn()
                    child[j] = random_gene

            new_gen.append(child)

        new_gen_fitness, new_gen_nn = self.get_gen_fitness(new_gen)
        self.apply_elitism(new_gen, new_gen_fitness)

        # descobre quem é o indivíduo com o melhor fitness da nova geração
        new_gen_best_fit_pos = np.argmax(new_gen_fitness)
        # guarda o valor de fitness desse indivíduo
        new_gen_best_fit = new_gen_fitness[new_gen_best_fit_pos]
        # se o melhor fitness da nova geração é igual ao melhor fitness da geração passada
        if (new_gen_best_fit == self.last_gen_best_fit):
            # incrementa 1 no contador que indica falta de mudanças no melhor fitness
            self.no_changes_best_fit += 1
        else:  # senão
            self.no_changes_best_fit = 0  # zera o contador para indicar que houve uma mudança
            self.last_gen_best_fit = new_gen_best_fit  # atualiza o valor do melhor fitness
            # atualiza a posição do indivíduo com melhor fitness
            self.last_gen_best_fit_pos = new_gen_best_fit_pos
            self.last_gen_best_nn = new_gen_nn[new_gen_best_fit_pos]

        self.last_gen = new_gen
        self.last_gen_fitness = new_gen_fitness
