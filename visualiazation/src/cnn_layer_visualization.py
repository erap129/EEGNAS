"""
Created on Sat Nov 18 23:12:08 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os

import mne
import numpy as np

import torch
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import models
from braindecode.torch_ext.util import np_to_var
from misc_functions import preprocess_image, recreate_image, save_image, create_mne
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly
from plotly import tools
from plotly.graph_objs import Layout, YAxis, Scatter, Annotation, Annotations, Data, Figure, Marker, Font


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

    def visualise_layer_with_hooks(self):
        # Hook the selected layer
        self.hook_layer()
        # Generate a random image
        random_image = np.float32(np.random.uniform(-1, 1, (1, 22, 1125, 1)))
        # Process image and return variable
        # processed_image = preprocess_image(random_image, False)
        processed_image = Variable(np_to_var(random_image), requires_grad=True)
        # Define optimizer for the image
        optimizer = Adam([processed_image], lr=0.1, weight_decay=1e-6)
        for i in range(1, 31):
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
            print('Iteration:', str(i), 'Loss:', "{0:.2f}".format(loss.data.numpy()))
            # Backward
            loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = create_mne(processed_image)

            if i % 5 == 0:
                # self.plotly_stacked()
                self.create_all_spectrograms(processed_image.detach().numpy().squeeze())

    def show_spectrogram(self, data):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(256 / 96, 256 / 96)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.specgram(data, NFFT=256, Fs=250)
        fig.canvas.draw()
        plt.show()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    def create_all_spectrograms(self, data, im_size=256):
        for j, channel in enumerate(data[:1]):
            channel *= 1000
            fig = plt.figure(frameon=False)
            fig.set_size_inches((im_size - 10) / 96, (im_size - 10) / 96)
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.plot(list(range(len(channel))), channel)
            ax2.specgram(channel, NFFT=256, Fs=250, noverlap=255)
            fig.canvas.draw()
            plt.show()
            plt.close(fig)

    def plotly_stacked(self):
        plotly.tools.set_credentials_file(username='erap129', api_key='P73jnmo843LIkBI4ouwy')
        picks = mne.pick_types(self.created_image.info, eeg=True, exclude=[])
        start, stop = self.created_image.time_as_index([0, 4.5])

        n_channels = 20
        data, times = self.created_image[picks[:n_channels], start:stop]
        ch_names = [self.created_image.info['ch_names'][p] for p in picks[:n_channels]]
        step = 1. / n_channels
        kwargs = dict(domain=[1 - step, 1], showticklabels=False, zeroline=False, showgrid=False)

        # create objects for layout and traces
        layout = Layout(yaxis=YAxis(kwargs), showlegend=False)
        traces = [Scatter(x=times, y=data.T[:, 0])]

        # loop over the channels
        for ii in range(1, n_channels):
            kwargs.update(domain=[1 - (ii + 1) * step, 1 - ii * step])
            layout.update({'yaxis%d' % (ii + 1): YAxis(kwargs), 'showlegend': False})
            traces.append(Scatter(x=times, y=data.T[:, ii], yaxis='y%d' % (ii + 1)))

        # add channel names using Annotations
        annotations = Annotations([Annotation(x=-0.06, y=0, xref='paper', yref='y%d' % (ii + 1),
                                              text=ch_name, font=Font(size=9), showarrow=False)
                                   for ii, ch_name in enumerate(ch_names)])
        layout.update(annotations=annotations)

        # set the size of the figure and plot it
        layout.update(autosize=False, width=1000, height=600)
        fig = Figure(data=Data(traces), layout=layout)
        py.iplot(fig, filename='shared xaxis')

    def plot(self):
        import mne
        import numpy as np
        from mne.datasets import sample
        data_path = sample.data_path()
        raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
        raw = mne.io.Raw(raw_fname, preload=False)
        print(raw)

        start, stop = raw.time_as_index([100, 115])  # 100 s to 115 s data segment
        data, times = raw[:306, start:stop]
        print(data.shape)
        print(times.shape)
        print(times.min(), times.max())

        picks = mne.pick_types(raw.info, meg='mag', exclude=[])
        data, times = raw[picks[:10], start:stop]

        import matplotlib.pyplot as plt
        import plotly.plotly as py
        import plotly
        plotly.tools.set_credentials_file(username='erap129', api_key='P73jnmo843LIkBI4ouwy')

        plt.plot(times, data.T)
        plt.xlabel('time (s)')
        plt.ylabel('MEG data (T)')

        update = dict(layout=dict(showlegend=True), data=[dict(name=raw.info['ch_names'][p]) for p in picks[:10]])
        py.iplot_mpl(plt.gcf(), update=update)


if __name__ == '__main__':
    cnn_layer = 6
    filter_pos = 5
    # Fully connected layer is not needed
    # pretrained_model = models.vgg16(pretrained=True).features
    pretrained_model = torch.load('../models/best_model_9_8_6_7_2_1_3_4_5.th')
    layer_vis = CNNLayerVisualization(pretrained_model, cnn_layer, filter_pos)

    # Layer visualization with pytorch hooks
    layer_vis.visualise_layer_with_hooks()

    # Layer visualization without pytorch hooks
    # layer_vis.visualise_layer_without_hooks()
