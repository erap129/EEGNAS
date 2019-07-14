import random
from collections import defaultdict
import utilities.monitors
from utilities.misc import chunks
import globals


def calc_ensembles_fitness(ensembles, pop_fitnesses, weighted_population):
    for ensemble in ensembles:
        ensemble_fit = one_ensemble_fitness(weighted_population, ensemble)
        for pop in ensemble:
            pop_fitnesses[pop].append(ensemble_fit)
    return ensemble_fit


def ensemble_fitness(weighted_population):
    pop_fitnesses = defaultdict(list)
    for iteration in range(globals.get('ensemble_iterations')):
        pop_indices = list(range(globals.get('pop_size')))
        random.shuffle(pop_indices)
        ensembles = list(chunks(pop_indices, globals.get('ensemble_size')))
        calc_ensembles_fitness(ensembles, pop_fitnesses, weighted_population)
    for pop_fitness in pop_fitnesses.items():
        weighted_population[pop_fitness[0]]['fitness'] = np.average(pop_fitness[1])


def one_subject_one_ensemble_fitness(weighted_population, ensemble, str_prefix=''):
    ensemble_preds = np.mean([weighted_population[i][f'{str_prefix}val_raw'] for i in ensemble], axis=0)
    pred_labels = np.argmax(ensemble_preds, axis=1).squeeze()
    ensemble_targets = weighted_population[ensemble[0]][f'{str_prefix}val_target']
    ensemble_fit = getattr(utilities.monitors, f'{globals.get("ga_objective")}_func')(pred_labels, ensemble_targets)
    return ensemble_fit


def one_ensemble_fitness(weighted_population, ensemble):
    if globals.get('cross_subject'):
        ensemble_fit = 0
        for subject in globals.get('subjects_to_check'):
            ensemble_fit += one_subject_one_ensemble_fitness(weighted_population, ensemble, str_prefix=f'{subject}_')
        return ensemble_fit / len(globals.get('subjects_to_check'))
    else:
        return one_subject_one_ensemble_fitness(weighted_population, ensemble)


def calculate_ensemble_fitness(weighted_population, ensemble):
    if globals.get('cross_subject'):
        ensemble_fit = 0
        for subject in range(1, globals.get('num_subjects') + 1):
            ensemble_fit += one_ensemble_fitness(weighted_population, ensemble)
        return ensemble_fit / globals.get('num_subjects')
    else:
        return one_ensemble_fitness(weighted_population, ensemble)


def permanent_ensemble_fitness(weighted_population):
    pop_indices = list(range(globals.get('pop_size')))
    ensembles = list(chunks(pop_indices, globals.get('ensemble_size')))
    perm_ensemble_fitnesses = []
    for i, ensemble in enumerate(ensembles):
        ensemble_fit = calculate_ensemble_fitness(weighted_population, ensemble)
        ensemble_fit_dict = {'group_id': i, 'fitness': ensemble_fit}
        perm_ensemble_fitnesses.append(ensemble_fit_dict)
        for pop_index in ensemble:
            weighted_population[pop_index]['fitness'] = ensemble_fit_dict['fitness']
            weighted_population[pop_index]['group_id'] = ensemble_fit_dict['group_id']
    perm_ensemble_fitnesses.sort(reverse=True, key=lambda x: x['fitness'])
    weighted_population.append(perm_ensemble_fitnesses)


def cross_subject_shared_fitness(weighted_population):
    for item in weighted_population:
        fitness = item[f'val_{globals.get("ga_objective")}']
        subject_array = range(1, globals.get('num_subjects')+1)
        fitness_vector = [item[f'{i}_val_{globals.get("ga_objective")}'] for i in subject_array]
        denominator = 1
        dists = []
        for pop in weighted_population:
            pop_fitness = [pop[f'{i}_val_{globals.get("ga_objective")}'] for i in range(1, globals.get('num_subjects')+1)]
            dists.append(pdist([fitness_vector, pop_fitness])[0])
        dists_norm = [float(i)/max(dists) for i in dists]
        for dist in dists_norm:
            if dist < globals.get('min_dist'):
                denominator += 1-dist
        item['fitness'] = fitness / denominator


def normal_fitness(weighted_population):
    for pop in weighted_population:
        pop['fitness'] = pop[f'val_{globals.get("ga_objective")}']
