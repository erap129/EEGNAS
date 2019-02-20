import pandas as pd
import numpy as np

num_activities = 8
samples_per_activity = 60

def get_human_activity(data_folder, subject_id):
    global num_activities, samples_per_activity
    np.random.seed(42)
    X_train = []
    y_train = []
    for activity in range(1, num_activities+1):
        for sample in range(1, samples_per_activity+1):
            y_train.append(activity-1)
            x_data = pd.read_csv(f'{data_folder}/a{subject_id:02d}/p{activity}/s{sample:02d}.txt').values
            X_train.append(x_data)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_train = np.swapaxes(X_train, 1, 2)
    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    X_train = X_train[s]
    y_train = y_train[s]

    X_test = X_train[8*50:]
    y_test = y_train[8*50:]
    X_train = X_train[:8*50]
    y_train = y_train[:8*50]

    return X_train, y_train, X_test, y_test
