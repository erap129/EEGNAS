import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
from sklearn import preprocessing
from EEGNAS.data_preprocessing import get_nyse_train_val_test


train_set, val_set, test_set = get_nyse_train_val_test('..')
X_test = test_set.X
X_train = train_set.X
nsamples, nx, ny = X_test.shape
d2_X_test = X_test.reshape((nsamples, nx * ny))
d2_X_test = preprocessing.normalize(d2_X_test)
nsamples, nx, ny = X_train.shape
d2_X_train = X_train.reshape((nsamples, nx * ny))
d2_X_train = preprocessing.normalize(d2_X_train)
clf = RandomForestClassifier()
clf.fit(d2_X_train, train_set.y)
pred = clf.predict(d2_X_test)
acc = metrics.accuracy_score(test_set.y, pred)
print(f'acc for NYSE is {acc}')
