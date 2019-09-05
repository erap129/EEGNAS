
def add_parent_child_relations(weighted_population, stats):
    avg_ratio = 0
    avg_count = 0
    for pop in weighted_population:
        if 'parents' in pop:
            parent_fitness = (pop['parents'][0]['fitness'] + pop['parents'][1]['fitness']) / 2
            avg_ratio += pop['fitness'] / parent_fitness
            avg_count += 1
    if avg_count != 0:
        stats['parent_child_ratio'] = avg_ratio / avg_count


def mark_perm_ensembles(weighted_population):
    for i, pop in enumerate(weighted_population):
        pop['perm_ensemble_role'] = i % globals.get('ensemble_size')
        pop['perm_ensemble_id'] = int(i / globals.get('ensemble_size'))
