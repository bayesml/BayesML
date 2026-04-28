import pytest
import os
import sys

# Add the parent directory to sys.path
# NOTE: This is a workaround for the import error when running the test file directly.
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from bayesml import linearregression

import numpy as np

SEED = 1

@pytest.fixture
def lr_sample_data(): # linear regression sample data
    gen_params = {
        'c_degree': 2,
        'seed': SEED
    }
    gen_model = linearregression.GenModel(**gen_params)
    gen_model.gen_params()
    x,y = gen_model.gen_sample(sample_size=1000)
    return {
        'x': x,
        'y': y
    }

def test_batch_p_params(lr_sample_data):
    x = lr_sample_data['x']
    y = lr_sample_data['y']

    learn_model = linearregression.LearnModel(c_degree=2)
    learn_model.update_posterior(x,y)
    learn_model.calc_pred_dist(x)
    p_params_batch = learn_model.get_p_params()

    p_params_seq = []
    for i in range(10):
        learn_model.calc_pred_dist(x[i])
        assert np.isclose(learn_model.get_p_params()['p_ms'], p_params_batch['p_ms'][i], atol=1e-10)
        assert np.isclose(learn_model.get_p_params()['p_lambdas'], p_params_batch['p_lambdas'][i], atol=1e-10)
        assert np.isclose(learn_model.get_p_params()['p_nus'], p_params_batch['p_nus'][i], atol=1e-10)

if __name__ == "__main__":
    pytest.main()