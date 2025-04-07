import pytest
import os
import sys

# Add the parent directory to sys.path
# NOTE: This is a workaround for the import error when running the test file directly.
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from bayesml import metatree
from bayesml import linearregression as submodel

import numpy as np

SEED = 123
rng = np.random.default_rng(SEED)

@pytest.fixture
def metatree_sample_data(): # linear regression sample data
    n = 10
    x_continuous = rng.random((n,3))
    x_categorical = rng.choice([0,1], size=(n,2))
    y_continuous = rng.random(n)
    y_categorical = rng.choice([0,1], size=n)
    return {
        'x_continuous': x_continuous,
        'x_categorical': x_categorical,
        'y_continuous': y_continuous,
        'y_categorical': y_categorical
    }

def test_metatree_batch_pred(metatree_sample_data):
    x_continuous = metatree_sample_data['x_continuous']
    x_categorical = metatree_sample_data['x_categorical']
    y_continuous = metatree_sample_data['y_continuous']
    y_categorical = metatree_sample_data['y_categorical']

    # initialise the model
    model = metatree.LearnModel(
        c_dim_continuous=3,
        c_dim_categorical=2,
        SubModel=submodel,
        sub_constants={'c_degree':3},
        sub_h0_params={'h0_alpha':1.1},
    )
    # update the posterior distribution
    model.update_posterior(
        x_continuous=x_continuous,
        x_categorical=x_categorical,
        y=y_continuous,
        random_state=123,
    )
    # calculate the predictive distribution
    model.calc_pred_dist(
        x_continuous=x_continuous,
        x_categorical=x_categorical,
    )
    
    ##############################
    # test on prediction values
    ##############################
    # calculate the prediction values
    pred_values = model.make_prediction(loss='squared')
    # desired prediction values
    # the values have been calculated using the same model and parameters, but as sequential predictions
    desireble_pred_values = np.array(
        [0.15029685, 0.24191882, 0.36074221, 0.26618345, 0.31980923,
        0.2772317 , 0.2354977 , 0.22170278, 0.13868499, 0.38301837])
    # check if the prediction values are close to the desired values
    assert np.all(np.isclose(pred_values, desireble_pred_values)), f"Prediction values are incorrect: {pred_values} != {desireble_pred_values}"

    ##############################
    # test on prediction variances
    ##############################
    # calculate the prediction variances
    pred_vars = model.calc_pred_var()
    # desired prediction variances
    desireble_pred_vars = np.array(
        [0.36318155, 0.35075593, 0.38577622, 0.36629843, 0.33658361,
        0.36846702, 0.35938511, 0.37930525, 0.31264912, 0.35395258])
    # check if the prediction variances are close to the desired values
    assert np.all(np.isclose(pred_vars, desireble_pred_vars)), f"Prediction variances are incorrect: {pred_vars} != {desireble_pred_vars}"

    ##############################
    # test on prediction densities
    ##############################
    # calculate the prediction densities
    pred_densities = model.calc_pred_density(np.arange(2)[:,np.newaxis])
    # desired prediction densities
    desireble_pred_densities = np.array(
        [[0.72017802, 0.65954781, 0.56689135, 0.64847739, 0.60924181,
        0.62572473, 0.66837509, 0.67467456, 0.75183963, 0.55488958],
        [0.19983381, 0.26276965, 0.35004522, 0.27428951, 0.31711274,
        0.29176254, 0.25486811, 0.24659016, 0.18370341, 0.36756198]])
    # check if the prediction densities are close to the desired values
    assert np.all(np.isclose(pred_densities, desireble_pred_densities)), f"Prediction densities are incorrect: {pred_densities} != {desireble_pred_densities}"

if __name__ == "__main__":
    pytest.main()