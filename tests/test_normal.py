import pytest
import os
import sys

# Add the parent directory to sys.path
# NOTE: This is a workaround for the import error when running the test file directly.
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from bayesml import normal

import numpy as np

RNG = np.random.default_rng(12345)

@pytest.fixture
def random_sample_data():
    n = 10
    x_continuous = RNG.random((n,5))
    x_categorical = RNG.choice([0,1], size=(n,5))
    y_continuous = RNG.random(n)
    y_categorical = RNG.choice([0,1], size=n)
    
    return {
        'x_continuous': x_continuous,
        'x_categorical': x_categorical,
        'y_continuous': y_continuous,
        'y_categorical': y_categorical
    }

@pytest.fixture
def standard_normal_sample_data():
    gen_params = {
        'mu': 0,
        'tau': 1,
        'seed': 12345
    } # standard normal distribution
    genmodel = normal.GenModel(**gen_params)
    num_samples = 10000
    return genmodel.gen_sample(num_samples)


# def test_null(random_sample_data):
#     # do nothing, just check if the test runs
#     pass

def test_posterior_mean(standard_normal_sample_data):
    # Test the posterior distribution
    learn_params = {
        'h0_m': 0.,
        'h0_kappa': 1.,
        'h0_alpha': 1.,
        'h0_beta': 1.,
    }
    learnmodel = normal.LearnModel(**learn_params)
    learnmodel.update_posterior(standard_normal_sample_data)
    assert np.isclose(learnmodel.get_hn_params()['hn_m'],standard_normal_sample_data.mean(), atol=1/standard_normal_sample_data.shape[0])
    # NOTE: Both values are not always close when the sample size is small, due to the nature of the posterior distribution.
    # NOTE: Absolute tolerance (atol) is set to 1/sample_size, which is not a too small value.

if __name__ == "__main__":
    pytest.main()