import numpy as np
import os

###################################################


def load_data(data_dir):
    """
    Loads data from data directory.
    """
    X = np.load(os.path.join(data_dir, 'X.npy'))
    y = np.load(os.path.join(data_dir, 'y.npy'))
    return X, y

