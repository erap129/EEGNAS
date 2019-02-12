import numpy as np
from moabb.datasets import Cho2017
from moabb.paradigms import (LeftRightImagery, MotorImagery,
                             FilterBankMotorImagery)

paradigm = LeftRightImagery()

dataset = Cho2017()
dataset.download()
subjects = [1]

X, y, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
pass