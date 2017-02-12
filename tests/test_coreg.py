from coreg.source import data_utils, coreg
import numpy as np
import os
import unittest

###################################################

data_dir = '../data/skillcraft'

X, y = data_utils.load_data(data_dir)

def test_load_data():
    X_check = np.load(os.path.join(data_dir, 'X.npy'))
    y_check = np.load(os.path.join(data_dir, 'y.npy'))
    assert np.allclose(X, X_check)
    assert np.allclose(y, y_check)

###################################################

