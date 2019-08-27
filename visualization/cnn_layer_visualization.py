"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import torch
from torch.optim import Adam
from braindecode.torch_ext.util import np_to_var
from models_generation import target_model
import global_vars
import numpy as np
from data_preprocessing import get_train_val_test
from scipy.io import savemat
from EEGNAS_experiment import get_normal_settings, set_params_by_dataset

class CNNLayerVisualization():
    """
        Produces an image that minimizes the loss of a convolution
        operation for a specific layer and filter
    """
    def __init__(self, model, selected_layer, selected_filter):
        self.model = model
        self.model.eval()
        self.selected_layer = selected_layer
        self.selected_filter = selected_filter
        self.conv_output = 0
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def hook_layer(self):
        def hook_function(module, grad_in, grad_out):
            # Gets the conv output of the selected filter (from selected layer)
            self.conv_output = grad_out[0, self.selected_filter]
        # Hook the selected layer
        self.model[self.selected_layer].register_forward_hook(hook_function)

    def visualise_layer_with_hooks(self, steps='max'):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.float32(np.random.uniform(-1, 1, (1, global_vars.get('eeg_chans'),
                                            global_vars.get('input_height'), global_vars.get('input_width'))))
        # Process image and return variable
        processed_image = torch.tensor(np_to_var(random_image), requires_grad=True, device='cuda')
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        curr_step = 0
        prev_loss = float("inf")
        while True:
            optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            x = processed_image
            for index, layer in enumerate(self.model):
                # Forward pass layer by layer
                # x is not used after this point because it is only needed to trigger
                # the forward hook function
                x = layer(x)
                # Only need to forward until the selected layer is reached
                if index == self.selected_layer:
                    # (forward hook function triggered)
                    break
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self.conv_output)
            if curr_step % 50 == 0:
                print('Iteration:', str(curr_step), 'Loss:', "{0:.2f}".format(loss.data.cpu().numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            curr_step += 1
            if curr_step == steps:
                break
            elif steps == 'max':
                if loss >= prev_loss:
                    break
                prev_loss = loss

        return processed_image.detach().cpu().numpy().squeeze()[None, :, :]
        # savemat(f'dataset_{globals.get("dataset")}_model_{model_selection}_layer_'
        #         f'{self.selected_layer}_filter_{self.selected_filter}_steps_{steps}.mat',
        #         {'X': processed_image.detach().cpu().numpy().squeeze()[None, :, :]})


def dataset_to_mat(dataset_name, data_sets_X, data_sets_y, class_ids):
    data_X = np.concatenate(data_sets_X)
    data_y = np.concatenate(data_sets_y)
    data_X = data_X[[y in class_ids for y in data_y]]
    savemat(f'{dataset_name}_classes_{class_ids}.mat', {'X': data_X})


if __name__ == '__main__':
    global_vars.set_dummy_config()
    global_vars.set('input_time_len', 1125)
    global_vars.set('n_classes', 4)
    global_vars.set('eeg_chans', 22)
    global_vars.set('valid_set_fraction', 0.2)
    global_vars.set('dataset', 'BCI_IV_2b')
    global_vars.set('batch_size', 60)
    global_vars.set('do_early_stop', True)
    global_vars.set('remember_best', True)
    global_vars.set('max_epochs', 50)
    global_vars.set('max_increase_epochs', 3)
    global_vars.set('final_max_epochs', 800)
    global_vars.set('final_max_increase_epochs', 80)
    global_vars.set('cuda', True)
    global_vars.set('data_folder', '../../data/')
    global_vars.set('low_cut_hz', 0)
    global_vars.set('dataset', 'BCI_IV_2b')
    set_params_by_dataset()
    model_selection = 'deep4'
    cnn_layer = {'evolution': 10, 'deep4': 25}
    filter_pos = {'evolution': 0, 'deep4': 0}
    model_dir = '91_x_BCI_IV_2b'
    model_name = 'best_model_5_1_8_7_9_2_3_4_6.th'
    model = {'evolution': torch.load(f'../../models/{model_dir}/{model_name}'),
                        'deep4': target_model('deep')}
    subject_id = 1
    train_set = {}
    val_set = {}
    test_set = {}
    train_set[subject_id], val_set[subject_id], test_set[subject_id] = \
        get_train_val_test(global_vars.get('data_folder'), subject_id)
    stop_criterion, iterator, loss_function, monitors = get_normal_settings()
    naiveNAS = NaiveNAS(iterator=iterator, exp_folder=None, exp_name=None,
                        train_set=train_set, val_set=val_set, test_set=test_set,
                        stop_criterion=stop_criterion, monitors=monitors, loss_function=loss_function,
                        config=global_vars.config, subject_id=subject_id, fieldnames=None, strategy='cross_subject',
                        evolution_file=None, csv_file=None)
    _, _, pretrained_model, _, _ = naiveNAS.evaluate_model(model[model_selection], final_evaluation=True)
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer[model_selection], filter_pos[model_selection])
    layer_vis.visualise_layer_with_hooks()

    # dataset_to_mat('BCI_IV_2b', [train_set[subject_id].X, val_set[subject_id].X, test_set[subject_id].X],
    #                [train_set[subject_id].y, val_set[subject_id].y, test_set[subject_id].y], [1])