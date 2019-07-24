import pickle

import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

import global_vars
from model_generation.model_similarity import network_similarity
from utilities.config_utils import set_default_config
import matplotlib
import numpy as np
import logging
import sys
log = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)s : %(message)s',
                    level=logging.DEBUG, stream=sys.stdout)

def sim_affinity(X):
    return pairwise_distances(X, metric=network_similarity)


set_default_config('configurations/config.ini')

population_file = 'weighted_populations/747_1_SO_pure_cross_subject_BCI_IV_2a_num_layers_30_pretrained.p'
population = pickle.load(open(population_file, 'rb'))

print(network_similarity(population[0]['model'], population[1]['model']))
similarity_matrix = np.zeros((global_vars.get('pop_size'), global_vars.get('pop_size')))
for i in range(similarity_matrix.shape[0]):
    for j in range(similarity_matrix.shape[1]):
        similarity_matrix[i,j] = network_similarity(population[i]['model'], population[j]['model'])
print(similarity_matrix)
cluster = AgglomerativeClustering(n_clusters=5, affinity='precomputed', linkage='average')
print(cluster.fit_predict(similarity_matrix))




