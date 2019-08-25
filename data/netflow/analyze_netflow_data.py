
if __name__ == '__main__':
    global_vars.init_config('configurations/config.ini')
    set_default_config('../../configurations/config.ini')
    global_vars.set('dataset', 'netflow_asflow')
    set_params_by_dataset('../../configurations/dataset_params.ini')
    global_vars.set('data_folder', '../../data/')
    global_vars.set('input_height', 24)
    global_vars.set('steps_ahead', 24)
    global_vars.set('no_shuffle', True)
    global_vars.set('jumps', 24)
    global_vars.set('start_point', 3)
    dataset = get_dataset('all')
    dataset = unify_dataset(dataset)
    for threshold in range(0, 100000, 1000):
        num_overflows = count_overflows_in_data(dataset, threshold)
        num_overflows_15_19 = count_overflows_in_data(dataset, threshold, 15, 24)
        print(f'num overflows for threshold {threshold}: {num_overflows}/{len(dataset.y)}')
        print(f'num overflows for threshold {threshold} between 15:00 and 19:00: {num_overflows_15_19}/{len(dataset.y)}')