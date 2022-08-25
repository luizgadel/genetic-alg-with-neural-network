import numpy as np
import random
from utils import *
from RedeNeuralLuiz import NeuralNetwork

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
        input_layer_size = 2
        hidden_layer_size = 7
        output_layer_size = 5

        weights_1_arr_size = (input_layer_size+1)*hidden_layer_size

        weights_1_arr = np.array(ind[:weights_1_arr_size])
        weights_2_arr = np.array(ind[weights_1_arr_size:])

        weights_1_mat = weights_1_arr.reshape((input_layer_size+1, hidden_layer_size))
        weights_2_mat = weights_2_arr.reshape((hidden_layer_size+1, output_layer_size))

        nn = NeuralNetwork(input_layer_size, hidden_layer_size, output_layer_size, weights_1_mat, weights_2_mat)
        output = nn.feed_forward(train_X)

        return (1 / compute_error(output)), nn

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
            new_parent = random.randint(0, limite)

            parent_position = 0
            while (new_parent > roulette_wheel[parent_position]):
                parent_position += 1

            parent_array.append(parent_position)
        return parent_array

    def apply_elitism(self, gen, gen_fitness):
        worst_fit_pos = np.argmin(gen_fitness) # descobre a posição do indivíduo com o pior fitness da nova geração

        gen[worst_fit_pos] = self.last_gen[self.last_gen_best_fit_pos] # substitui esse indivíduo pelo melhor da geração passada
        gen_fitness[worst_fit_pos] = self.last_gen_best_fit # atualiza o fitness do novo indivíduo

    def new_generation(self):
        new_gen = []
        for i in range(self.n_ind):
            parents = self.random_parents()

            p = np.random.rand()

            ''' crossover '''
            if (p <= self.p_crossover):
                crossover = random.randint(1, self.n_genes-1)
                child = np.concatenate(
                    (self.last_gen[parents[0]][:crossover], self.last_gen[parents[1]][crossover:]), axis=None)
            else:
                y = random.randint(0, 1)
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

        new_gen_best_fit_pos = np.argmax(new_gen_fitness) # descobre quem é o indivíduo com o melhor fitness da nova geração
        new_gen_best_fit = new_gen_fitness[new_gen_best_fit_pos] # guarda o valor de fitness desse indivíduo
        if (new_gen_best_fit == self.last_gen_best_fit): # se o melhor fitness da nova geração é igual ao melhor fitness da geração passada
            self.no_changes_best_fit += 1 # incrementa 1 no contador que indica falta de mudanças no melhor fitness
        else: # senão
            self.no_changes_best_fit = 0 # zera o contador para indicar que houve uma mudança
            self.last_gen_best_fit = new_gen_best_fit # atualiza o valor do melhor fitness
            self.last_gen_best_fit_pos = new_gen_best_fit_pos # atualiza a posição do indivíduo com melhor fitness
            self.last_gen_best_nn = new_gen_nn[new_gen_best_fit_pos]

        self.last_gen = new_gen
        self.last_gen_fitness = new_gen_fitness