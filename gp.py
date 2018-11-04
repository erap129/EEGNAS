"""Genetic programming algorithms."""
from __future__ import absolute_import

import random
import numpy as np
from operator import itemgetter
from models_generation import print_structure
import copy


class TournamentOptimizer:
    """Define a tournament play selection process."""

    def __init__(self, population_sz, init_fn, mutate_fn, naive_nas):
        """
        Initialize optimizer.

            params::

                init_fn: initialize a model
                mutate_fn: mutate function - mutates a model
                eval_fn: evaluate function - evaluates the fitness of a model
                nb_workers: number of workers
        """

        self.init_fn = init_fn
        self.mutate_fn = mutate_fn
        self.naive_nas = naive_nas
        self.n_chans = naive_nas.n_chans
        self.input_time_len = naive_nas.input_time_len

        # population
        self.population_sz = population_sz
        self.population = [init_fn(self.n_chans, self.input_time_len) for i in range(population_sz)]
        self.evaluations = np.zeros(population_sz)
        self.finalized_models = None

        # book keeping
        self.elite = []
        self.stats = []
        self.history = []

    def step(self):
        """Tournament evolution step."""
        print('\nPopulation sample:')
        for i in range(0, self.population_sz, 3):
            print_structure(self.population[i].layer_collection)
        self.evaluate()
        children = []
        print('\nPopulation mean:{} max:{}'.format(
            np.mean(self.evaluations), np.max(self.evaluations)))
        n_elite = 2
        sorted_pop = np.argsort(self.evaluations)[::-1]
        elite = sorted_pop[:n_elite]
        top_model = self.finalized_models[elite[0]]
        # print top@n_elite scores
        # elites always included in the next population
        self.elite = []
        print('\nTop performers:')
        for i, e in enumerate(elite):
            self.elite.append((self.evaluations[e], self.population[e]))
            print("{}-score:{}".format(str(i), self.evaluations[e]))
            self.finalized_models[e].summary()
            children.append(self.population[e])
        # tournament probabilities:
        # first p
        # second p*(1-p)
        # third p*((1-p)^2)
        # etc...
        p = 0.85  # winner probability
        tournament_size = 3
        probs = [p * ((1 - p) ** i) for i in range(tournament_size - 1)]
        # a little trick to certify that probs is adding up to 1.0
        probs.append(1 - np.sum(probs))

        while len(children) < self.population_sz:
            pop = range(len(self.population))
            sel_k = random.sample(pop, k=tournament_size)
            fitness_k = list(np.array(self.evaluations)[sel_k])
            selected = zip(sel_k, fitness_k)
            rank = sorted(selected, key=itemgetter(1), reverse=True)
            pick = np.random.choice(tournament_size, size=1, p=probs)[0]
            best = rank[pick][0]
            model = self.mutate_fn(self.population[best])
            children.append(model)

        self.population = children
        return top_model
        # if we want to do a completely completely random search per epoch
        # self.population = [randomize_network(bounded=False) for i in range(self.population_sz) ]

    def evaluate(self):
        """evaluate the models."""
        returns = []
        models = []
        for model in self.population:
            res, finalized_model = self.naive_nas.evaluate_model(model)
            returns.append(res)
            models.append(finalized_model)

        self.evaluations = returns
        self.finalized_models = models
        self.stats.append(copy.deepcopy(returns))
