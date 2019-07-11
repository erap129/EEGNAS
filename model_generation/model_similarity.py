import random
from collections import defaultdict
from model_generation.abstract_layers import *
from Bio import pairwise2

from model_generation.abstract_layers import ConvLayer, PoolingLayer


def string_representation(layer_collection):
    translation = {FlattenLayer: 'f',
                   DropoutLayer: 'd',
                   BatchNormLayer: 'b',
                   ConvLayer: 'c',
                   PoolingLayer: 'p',
                   ActivationLayer: 'a',
                   IdentityLayer: 'i'}
    rep = ''
    for layer in layer_collection:
        rep += translation[type(layer)]
    return rep


def get_layer(layer_collection, layer_type, order):
    count = 0
    for layer in layer_collection:
        if type(layer) == layer_type:
            count += 1
            if count == order:
                return layer
    print(f"searched for layer {str(layer_type)} but didn't find.\n"
          f"the layer collection is: {str(layer_collection)}\n"
          f"the order is: {order}")


def layer_comparison(layer_type, layer1_order, layer2_order, layer_collection1, layer_collection2, attrs, output):
    score = 0
    layer1 = get_layer(layer_collection1, layer_type, layer1_order)
    layer2 = get_layer(layer_collection2, layer_type, layer2_order)
    for attr in attrs:
        added_value = 1 / (abs(getattr(layer1, attr) - getattr(layer2, attr)) + 1) * 5
        score += added_value
        output.append(f"{layer_type.__name__}_{layer1_order}_{layer2_order}"
                      f" with attribute {attr}, added value : 1 / abs({getattr(layer1, attr)}"
                      f" - {getattr(layer2, attr)}) + 1 * 5 = {added_value:.3f}")
    return score


def network_similarity(layer_collection1, layer_collection2, return_output=False):
    str1 = string_representation(layer_collection1)
    str2 = string_representation(layer_collection2)
    alignment = pairwise2.align.globalms(str1, str2, 2, -1, -.5, -.1)[0]
    output = ['-' * 50]
    output.append(format_alignment(*alignment))
    score = alignment[2]
    str1_orders = defaultdict(lambda:0)
    str2_orders = defaultdict(lambda:0)
    for x,y in (zip(alignment[0], alignment[1])):
        str1_orders[x] += 1
        str2_orders[y] += 1
        if x == y == 'c':
            score += layer_comparison(ConvLayer, str1_orders['c'], str2_orders['c'],
                                      layer_collection1, layer_collection2,
                                      ['kernel_eeg_chan', 'filter_num', 'kernel_time'], output)
        if x == y == 'p':
            score += layer_comparison(PoolingLayer, str1_orders['p'], str2_orders['p'],
                                      layer_collection1, layer_collection2,
                                      ['pool_time', 'stride_time'], output)
    output.append(f"final similarity: {score:.3f}")
    output.append('-' * 50)
    if return_output:
        return score, '\n'.join(output)
    else:
        return score


def calculate_population_similarity(layer_collections, evolution_file, sim_count):
    sim = 0
    to_output = 3
    for i in range(sim_count):
        idxs = random.sample(range(len(layer_collections)), 2)
        score, output = network_similarity(layer_collections[idxs[0]], layer_collections[idxs[1]], return_output=True)
        sim += score
        if to_output > 0:
            with open(evolution_file, "a") as text_file:
                print(output, file=text_file)
            to_output -= 1
    return sim / sim_count


def calculate_one_similarity(layer_collection, other_layer_collections):
    sim = 0
    for other_layer_collection in other_layer_collections:
        score, output = network_similarity(layer_collection, other_layer_collection)
        sim += score
    return sim / len(other_layer_collections)
