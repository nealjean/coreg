from coreg import Coreg
import numpy as np


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
cr.add_data(data_dir)

# Run training
num_train = 100
trials = 3
verbose = True
cr.run_trials(num_train, trials, verbose)
