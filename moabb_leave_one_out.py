"""=======================
Statistical Analysis
=======================

The MOABB codebase comes with convenience plotting utilities and some
statistical testing. This tutorial focuses on what those exactly are and how
they can be used.

"""
# Authors: Vinay Jayaram <vinayjayaram13@gmail.com>
#
# License: BSD (3-clause)

import moabb
import matplotlib.pyplot as plt
import moabb.analysis.plotting as moabb_plt
import torch
from moabb.analysis.meta_analysis import (
    find_significant_differences, compute_dataset_statistics)  # noqa: E501

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from moabb.datasets import BNCI2014001
from moabb.paradigms import LeftRightImagery, MotorImagery
from moabb.evaluations import CrossSessionEvaluation, CrossSubjectEvaluation
from skorch import NeuralNetClassifier

moabb.set_log_level('info')

print(__doc__)

###############################################################################
# Results Generation
# ---------------------
#
# First we need to set up a paradigm, dataset list, and some pipelines to
# test. This is explored more in the examples -- we choose a left vs right
# imagery paradigm with a single bandpass. There is only one dataset here but
# any number can be added without changing this workflow.
#
# Create pipelines
# ----------------
#
# Pipelines must be a dict of sklearn pipeline transformer.
#
# The csp implementation from MNE is used. We selected 8 CSP components, as
# usually done in the litterature.
#
# The riemannian geometry pipeline consists in covariance estimation, tangent
# space mapping and finaly a logistic regression for the classification.

pipelines = {}
model_dir = '128_2_BNCI2014001'
model_name = 'BNCI2014001.th'
model = torch.load(f'models/{model_dir}/{model_name}')
net = NeuralNetClassifier(
            model,
            max_epochs=50,
            lr=1e-3,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
        )

pipelines['CSP + LDA'] = make_pipeline(CSP(n_components=8), LDA())

pipelines['RG + LR'] = make_pipeline(Covariances(), TangentSpace(),
                                     LogisticRegression())

pipelines['CSP + LR'] = make_pipeline(
    CSP(n_components=8), LogisticRegression())

pipelines['RG + LDA'] = make_pipeline(Covariances(), TangentSpace(), LDA())

##############################################################################
# Evaluation
# ----------
#
# We define the paradigm (LeftRightImagery) and the dataset (BNCI2014001).
# The evaluation will return a dataframe containing a single AUC score for
# each subject / session of the dataset, and for each pipeline.
#
# Results are saved into the database, so that if you add a new pipeline, it
# will not run again the evaluation unless a parameter has changed. Results can
# be overwritten if necessary.

paradigm = MotorImagery()
datasets = paradigm.datasets[:2]
overwrite = False  # set to True if we want to overwrite cached results
evaluation = CrossSubjectEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    suffix='examples',
    overwrite=overwrite)

results = evaluation.process(pipelines)
print
