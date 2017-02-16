from coreg.source import data_utils, coreg
import numpy as np
import os
import unittest

###################################################

data_dir = '../data/skillcraft'
k1 = 3
k2 = 3
p1 = 2
p2 = 5
max_iters = 100
pool_size = 100
verbose = True
random_state = -1
num_labeled = 100
num_test = 1000

cr = coreg.Coreg(k1, k2, p1, p2, max_iters, pool_size)

###################################################

cr.add_data(data_dir)
cr.split_data(random_state, num_labeled, num_test)

def test_add_data():
    X_check = np.load(os.path.join(data_dir, 'X.npy'))
    y_check = np.load(os.path.join(data_dir, 'y.npy'))
    assert np.allclose(cr.X, X_check)
    assert np.allclose(cr.y, y_check)

###################################################

cr.train(verbose=verbose)