import numpy as np
from moabb.datasets import Cho2017, BNCI2014001, BNCI2014004
from moabb.paradigms import (LeftRightImagery, MotorImagery,
                             FilterBankMotorImagery)
from moabb.evaluations import CrossSessionEvaluation, WithinSessionEvaluation
from moabb.datasets import utils

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from moabb.pipelines.features import LogVariance
import numpy as np

import moabb
# moabb.set_log_level('info')
#
# pipelines = {}
# pipelines['AM + LDA'] = make_pipeline(LogVariance(),
#                                       LDA())
# parameters = {'C': np.logspace(-2, 2, 10)}
# clf = GridSearchCV(SVC(kernel='linear'), parameters)
# pipe = make_pipeline(LogVariance(), clf)
#
# pipelines['AM + SVM'] = pipe
#
# datasets = [Cho2017()]
#
paradigm = LeftRightImagery()
#
# evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets,
#                                     suffix='examples', overwrite=False)
# results = evaluation.process(pipelines)
#
# results.to_csv('moab_cho.csv')

dataset = Cho2017()
dataset.download(path='/home/cluster/users/eladr/mne_data')
subjects = [5]

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
pass