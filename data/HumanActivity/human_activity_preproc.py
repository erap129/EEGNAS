import pandas as pd
import numpy as np

num_activities = 8
samples_per_activity_train = 50
samples_per_activity_test = 10

def get_human_activity(data_folder, subject_id):
    global num_activities, samples_per_activity_train, samples_per_activity_test
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for activity in range(1, num_activities+1):
        for sample in range(1, samples_per_activity_train+1):
            y_train.append(activity)
            x_data = pd.read_csv(f'{data_folder}/a{subject_id:02d}/p{activity}/s{sample:02d}.txt').values
            X_train.append(x_data)
        for sample in range(samples_per_activity_train, samples_per_activity_train+samples_per_activity_test+1):
            y_test.append(activity)
            x_data = pd.read_csv(f'{data_folder}/a{subject_id:02d}/p{activity}/s{sample:02d}.txt').values
            X_test.append(x_data)
    X_train = np.array(X_train)
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.array(X_test)
    X_test = np.swapaxes(X_test, 1, 2)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    X_train = X_train[s]
    y_train = y_train[s]
    s = np.arange(X_test.shape[0])
    np.random.shuffle(s)
    X_test = X_test[s]
    y_test = y_test[s]

    return X_train, y_train, X_test, y_test
