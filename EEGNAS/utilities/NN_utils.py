import numpy as np
import torch


def get_intermediate_layer_value(model, x, layer_idx):
    model.eval()
    for idx, layer in enumerate(list(model.children())[:layer_idx+1]):
        x = layer(x)
    return x


def get_class_distribution(model, X):
    model.eval()
    with torch.no_grad():
        preds = model(X)
        preds = preds.cpu().data.numpy()
        pred_labels = np.argmax(preds, axis=1).squeeze()
    return np.bincount(pred_labels)
