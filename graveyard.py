def find_best_model_evolution(self):
    nb_evolution_steps = 10
    tournament = \
        gp.TournamentOptimizer(
            population_sz=10,
            init_fn=random_model,
            mutate_fn=mutate_net,
            naive_nas=self)

    for i in range(nb_evolution_steps):
        print('\nEvolution step:{}'.format(i))
        print('================')
        top_model = tournament.step()
        # keep track of the experiment results & corresponding architectures
        name = "tourney_{}".format(i)
        res, _ = self.evaluate_model(top_model, test=True)
    return res


def find_best_model_bb(self, folder_name, time_limit=12 * 60 * 60):
    curr_models = queue.Queue()
    operations = [self.factor_filters, self.add_conv_maxpool_block]
    initial_model = self.base_model()
    curr_models.put(initial_model)
    res = self.evaluate_model(initial_model)
    total_results = []
    result_tree = treelib.Tree()
    result_tree.create_node(data=res, identifier=initial_model.name)
    start_time = time.time()
    while not curr_models.empty() and time.time() - start_time < time_limit:
        curr_model = curr_models.get_nowait()
        for op in operations:
            model = op(curr_model)
            res = self.evaluate_model(model)
            print('node name is:', model.name)
            result_tree.create_node(data=res, identifier=model.name, parent=curr_model.name)
            result_tree.show()
            print('model accuracy:', res[1] * 100)
            total_results.append(res[1])
            curr_models.put(model)
    print(total_results)
    return total_results.max()


def find_best_model_simanneal(self, folder_name, experiment=None, time_limit=12 * 60 * 60):
    curr_model = self.base_model()
    earlystopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=5)
    mcp = ModelCheckpoint('best_keras_model.hdf5', save_best_only=True, monitor='val_acc', mode='max')
    start_time = time.time()
    curr_acc = 0
    num_of_ops = 0
    temperature = 10
    coolingRate = 0.003
    operations = [self.factor_filters, self.add_conv_maxpool_block]
    total_time = 0
    if experiment == 'filter_experiment':
        results = pd.DataFrame(columns=['conv1 filters', 'conv2 filters', 'conv3 filters',
                                        'accuracy', 'runtime', 'switch probability', 'temperature'])
    while time.time() - start_time < time_limit and not self.finalize_flag:
        K.clear_session()
        op_index = random.randint(0, len(operations) - 1)
        num_of_ops += 1
        mcp = ModelCheckpoint('keras_models/best_keras_model' + str(num_of_ops) + '.hdf5',
                              save_best_only=True, monitor='val_acc', mode='max', save_weights_only=True)
        model = operations[op_index](curr_model)
        finalized_model = self.finalize_model(model.layer_collection)
        if self.cropping:
            finalized_model.model = convert_to_dilated(model.model)
        start = time.time()
        finalized_model.model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                                  callbacks=[earlystopping, mcp])
        finalized_model.model.load_weights('keras_models/best_keras_model' + str(num_of_ops) + '.hdf5')
        end = time.time()
        total_time += end - start
        res = finalized_model.model.evaluate(self.X_test, self.y_test)[1] * 100
        curr_model = model
        if res >= curr_acc:
            curr_model = model
            curr_acc = res
            probability = -1
        else:
            probability = np.exp((res - curr_acc) / temperature)
            rand = np.random.choice(a=[1, 0], p=[probability, 1 - probability])
            if rand == 1:
                curr_model = model
        temperature *= (1 - coolingRate)
        print('train time in seconds:', end - start)
        print('model accuracy:', res * 100)
        if experiment == 'filter_experiment':
            results.loc[num_of_ops - 1] = np.array([int(finalized_model.model.get_layer('conv1').filters),
                                                    int(finalized_model.model.get_layer('conv2').filters),
                                                    int(finalized_model.model.get_layer('conv3').filters),
                                                    res,
                                                    str(end - start),
                                                    str(probability),
                                                    str(temperature)])
            print(results)
    final_model = self.finalize_model(curr_model.layer_collection)
    final_model.model.fit(self.X_train, self.y_train, epochs=50, validation_data=(self.X_valid, self.y_valid),
                          callbacks=[earlystopping])
    res = final_model.model.evaluate(self.X_test, self.y_test)[1] * 100
    print('model accuracy:', res)
    print('average train time per model', total_time / num_of_ops)
    now = str(datetime.datetime.now()).replace(":", "-")
    if experiment == 'filter_experiment':
        if not os.path.isdir('results/' + folder_name):
            createFolder('results/' + folder_name)
        results.to_csv('results/' + folder_name + '/subject_' + str(self.subject_id) + experiment + '_' + now + '.csv',
                       mode='a')


def grid_search_filters(self, lo, hi, jumps):
    model = target_model()
    start_time = time.time()
    num_of_ops = 0
    total_time = 0
    results = pd.DataFrame(columns=['conv1 filters', 'conv2 filters', 'conv3 filters',
                                    'test acc', 'val acc', 'train acc', 'runtime'])
    for first_filt in range(lo, hi, jumps):
        for second_filt in range(lo, hi, jumps):
            for third_filt in range(lo, hi, jumps):
                K.clear_session()
                num_of_ops += 1
                model = set_target_model_filters(model, first_filt, second_filt, third_filt)
                run_time, res_test, res_val, res_train, _ = self.evaluate_model(model)
                total_time += run_time
                print('train time in seconds:', run_time)
                print('model accuracy:', res_test)
                results.loc[num_of_ops - 1] = np.array([int(model.model.get_layer('conv1').filters),
                                                        int(model.model.get_layer('conv2').filters),
                                                        int(model.model.get_layer('conv3').filters),
                                                        res_test,
                                                        res_val,
                                                        res_train,
                                                        str(run_time)])
                print(results)

    total_time = time.time() - start_time
    print('average train time per model', total_time / num_of_ops)
    now = str(datetime.datetime.now()).replace(":", "-")
    results.to_csv('results/filter_gridsearch_' + now + '.csv', mode='a')


def grid_search_kernel_size(self, lo, hi, jumps):
    model = target_model()
    start_time = time.time()
    num_of_ops = 0
    total_time = 0
    results = pd.DataFrame(columns=['conv1 kernel size', 'conv2 kernel size', 'conv3 kernel size',
                                    'accuracy', 'train acc', 'runtime'])
    for first_size in range(lo, hi, jumps):
        for second_size in range(lo, hi, jumps):
            for third_size in range(lo, hi, jumps):
                K.clear_session()
                num_of_ops += 1
                model = set_target_model_kernel_sizes(model, first_size, second_size, third_size)
                run_time, res, res_train = self.run_one_model(model)
                total_time += time
                print('train time in seconds:', time)
                print('model accuracy:', res)
                results.loc[num_of_ops - 1] = np.array([int(model.model.get_layer('conv1').kernel_size[1]),
                                                        int(model.model.get_layer('conv2').kernel_size[1]),
                                                        int(model.model.get_layer('conv3').kernel_size[1]),
                                                        res_train,
                                                        str(run_time)])
                print(results)

    total_time = time.time() - start_time
    print('average train time per model', total_time / num_of_ops)
    now = str(datetime.datetime.now()).replace(":", "-")
    results.to_csv('results/kernel_size_gridsearch_' + now + '.csv', mode='a')




# def folder_renamer():
#     if not globals.config['DEFAULT'].getboolean('success'):
#         new_exp_folder = exp_folder + '_fail'
#     else:
#         new_exp_folder = exp_folder
#     os.rename(exp_folder, new_exp_folder)
#
#
# atexit.register(folder_renamer())



def add_conv_maxpool_block(layer_collection, conv_width=10, conv_filter_num=50, dropout=False,
                           pool_width=3, pool_stride=3, conv_layer_name=None, random_values=True):
    layer_collection = copy.deepcopy(layer_collection)
    if random_values:
        conv_time = random.randint(5, 10)
        conv_filter_num = random.randint(0, 50)
        pool_time = 2
        pool_stride = 2

    topo_layers = create_topo_layers(layer_collection.values())
    last_layer_id = topo_layers[-1]
    if dropout:
        dropout = DropoutLayer()
        layer_collection[dropout.id] = dropout
        layer_collection[last_layer_id].make_connection(dropout)
        last_layer_id = dropout.id
    conv_layer = ConvLayer(kernel_time=conv_width, kernel_eeg_chan=1,
                           filter_num=conv_filter_num, name=conv_layer_name)
    layer_collection[conv_layer.id] = conv_layer
    layer_collection[last_layer_id].make_connection(conv_layer)
    batchnorm_layer = BatchNormLayer()
    layer_collection[batchnorm_layer.id] = batchnorm_layer
    conv_layer.make_connection(batchnorm_layer)
    activation_layer = ActivationLayer()
    layer_collection[activation_layer.id] = activation_layer
    batchnorm_layer.make_connection(activation_layer)
    maxpool_layer = PoolingLayer(pool_time=pool_width, stride_time=pool_stride, mode='MAX')
    layer_collection[maxpool_layer.id] = maxpool_layer
    activation_layer.make_connection(maxpool_layer)
    return layer_collection
    # return MyModel.new_model_from_structure(layer_collection, name=model.name + '->add_conv_maxpool')

def add_skip_connection_concat(model):
    topo_layers = create_topo_layers(model.layer_collection.values())
    to_concat = random.sample(model.layer_collection.keys(), 2)  # choose 2 random layer id's
    # first_layer_index = topo_layers.index(np.min(to_concat))
    # second_layer_index = topo_layers.index(np.max(to_concat))
    # first_layer_index = topo_layers[first_layer_index]
    # second_layer_index = topo_layers[second_layer_index]
    first_layer_index = np.min(to_concat)
    second_layer_index = np.max(to_concat)
    first_shape = model.model.get_layer(str(first_layer_index)).output.shape
    second_shape = model.model.get_layer(str(second_layer_index)).output.shape
    print('first layer shape is:', first_shape)
    print('second layer shape is:', second_shape)

    height_diff = int(first_shape[1]) - int(second_shape[1])
    width_diff = int(first_shape[2]) - int(second_shape[2])
    height_crop_top = height_crop_bottom = np.abs(int(height_diff / 2))
    width_crop_left = width_crop_right = np.abs(int(width_diff / 2))
    if height_diff % 2 == 1:
        height_crop_top += 1
    if width_diff % 2 == 1:
        width_crop_left += 1
    if height_diff < 0:
        ChosenHeightClass = ZeroPadLayer
    else:
        ChosenHeightClass = CroppingLayer
    if width_diff < 0:
        ChosenWidthClass = ZeroPadLayer
    else:
        ChosenWidthClass = CroppingLayer
    first_layer = model.layer_collection[first_layer_index]
    second_layer = model.layer_collection[second_layer_index]
    next_layer = first_layer
    if height_diff != 0:
        heightChanger = ChosenHeightClass(height_crop_top, height_crop_bottom, 0, 0)
        model.layer_collection[heightChanger.id] = heightChanger
        first_layer.make_connection(heightChanger)
        next_layer = heightChanger
    if width_diff != 0:
        widthChanger = ChosenWidthClass(0, 0, width_crop_left, width_crop_right)
        model.layer_collection[widthChanger.id] = widthChanger
        next_layer.make_connection(widthChanger)
        next_layer = widthChanger
    concat = ConcatLayer(next_layer.id, second_layer_index)
    model.layer_collection[concat.id] = concat
    next_layer.connections.append(concat)
    for lay in second_layer.connections:
        concat.connections.append(lay)
        if not isinstance(lay, ConcatLayer):
            lay.parent = concat
        else:
            if lay.second_layer_index == second_layer_index:
                lay.second_layer_index = concat.id
            if lay.first_layer_index == second_layer_index:
                lay.first_layer_index = concat.id
    second_layer.connections = []
    second_layer.connections.append(concat)
    return model

def edit_conv_layer(model, mode):
    layer_collection = copy.deepcopy(model.layer_collection)
    conv_indices = [layer.id for layer in layer_collection.values() if isinstance(layer, ConvLayer)]
    try:
        random_conv_index = random.randint(2, len(conv_indices) - 2) # don't include last conv
    except ValueError:
        return model
    factor = 1 + random.uniform(0,1)
    if mode == 'filters':
        layer_collection[conv_indices[random_conv_index]].filter_num = \
            int(layer_collection[conv_indices[random_conv_index]].filter_num * factor)
    elif mode == 'kernels':
        layer_collection[conv_indices[random_conv_index]].kernel_width = \
            int(layer_collection[conv_indices[random_conv_index]].kernel_width * factor)
    return MyModel.new_model_from_structure(layer_collection, name=model.name + '->factor_filters')

def factor_filters(model):
    return edit_conv_layer(model, mode='filters')

def factor_kernels(model):
    return edit_conv_layer(model, mode='kernels')

def set_target_model_filters(model, filt1, filt2, filt3):
    conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
    conv_indices = conv_indices[2:len(conv_indices)]  # take only relevant indices
    model.layer_collection[conv_indices[0]].filter_num = filt1
    model.layer_collection[conv_indices[1]].filter_num = filt2
    model.layer_collection[conv_indices[2]].filter_num = filt3
    return model.new_model_from_structure(model.layer_collection)

def set_target_model_kernel_sizes(model, size1, size2, size3):
    conv_indices = [layer.id for layer in model.layer_collection.values() if isinstance(layer, ConvLayer)]
    conv_indices = conv_indices[2:len(conv_indices)]  # take only relevant indices
    model.layer_collection[conv_indices[0]].kernel_width = size1
    model.layer_collection[conv_indices[1]].kernel_width = size2
    model.layer_collection[conv_indices[2]].kernel_width = size3
    return model.new_model_from_structure(model.layer_collection)

def mutate_net(model):
    operations = [add_conv_maxpool_block, factor_filters, factor_kernels, add_skip_connection_concat]
    op_index = random.randint(0, len(operations) - 1)
    try:
        model = operations[op_index](model)
    except ValueError as e:
        print('exception occured while performing ', operations[op_index].__name__)
        print('error message: ', str(e))
        print('trying another mutation...')
        return mutate_net(model)
    return model



# class CroppingLayer(Layer):
#     @initializer
#     def __init__(self, height_crop_top, height_crop_bottom, width_crop_left, width_crop_right):
#         Layer.__init__(self)
