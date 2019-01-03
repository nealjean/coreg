from coreg import Coreg
import numpy as np


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

cr = Coreg(k1, k2, p1, p2, max_iters, pool_size)

###################################################

cr.add_data(data_dir)

def test_add_data():
    assert cr.X.shape == (3325, 18)
    assert cr.y.shape == (3325, 1)

####

cr._split_data(random_state, num_labeled, num_test)

def test_split_data():
    assert np.allclose(cr.L1_X, cr.X_labeled)

####

cr._fit_and_evaluate(verbose=False)
cr._get_pool()

def test_iteration():
    stop_training = cr._run_iteration(0, 0, False, False)
    assert stop_training in [True, False]