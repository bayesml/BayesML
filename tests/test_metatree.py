import pytest
import os
import sys

# Add the parent directory to sys.path
# NOTE: This is a workaround for the import error when running the test file directly.
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from bayesml import metatree
from bayesml import normal, linearregression, poisson, exponential # REG models
from bayesml import bernoulli, categorical # CLF models

import numpy as np

SEED = 123

@pytest.fixture
def metatree_sample_data():
    rng = np.random.default_rng(SEED)
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


def test_metatree_normal_batch_pred(metatree_sample_data):
    x_continuous = metatree_sample_data['x_continuous']
    x_categorical = metatree_sample_data['x_categorical']
    y_continuous = metatree_sample_data['y_continuous']
    y_categorical = metatree_sample_data['y_categorical']

    # initialise the model
    model = metatree.LearnModel(
        c_dim_continuous=3,
        c_dim_categorical=2,
        SubModel=normal,
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
        [0.27641338, 0.28833937, 0.28857635, 0.28985178, 0.28706087,
         0.2855081 , 0.27751605, 0.27984023, 0.28821349, 0.28831634])
    # check if the prediction values are close to the desired values
    assert np.all(np.isclose(pred_values, desireble_pred_values)), f"Prediction values are incorrect: {pred_values} != {desireble_pred_values}"

    ##############################
    # test on prediction variances
    ##############################
    # calculate the prediction variances
    pred_vars = model.calc_pred_var()
    # desired prediction variances
    desireble_pred_vars = np.array(
        [0.35755206, 0.31119767, 0.33593725, 0.33574303, 0.30342827,
         0.3154599 , 0.34116193, 0.36740548, 0.31259223, 0.30591997])
    # check if the prediction variances are close to the desired values
    assert np.all(np.isclose(pred_vars, desireble_pred_vars)), f"Prediction variances are incorrect: {pred_vars} != {desireble_pred_vars}"

    ##############################
    # test on prediction densities
    ##############################
    # calculate the prediction densities
    pred_densities = model.calc_pred_density(np.arange(2)[:,np.newaxis])
    # desired prediction densities
    desireble_pred_densities = np.array(
        [[0.6500694 , 0.65388057, 0.64928667, 0.64943002, 0.65552694,
          0.65364597, 0.65171061, 0.64985625, 0.65385478, 0.65467585],
        [0.27694262, 0.28162349, 0.28181434, 0.28212561, 0.28105075,
         0.28043262, 0.27731163, 0.27854053, 0.28156627, 0.28163985]])
    # check if the prediction densities are close to the desired values
    assert np.all(np.isclose(pred_densities, desireble_pred_densities)), f"Prediction densities are incorrect: {pred_densities} != {desireble_pred_densities}"

def test_metatree_linearregression_batch_pred(metatree_sample_data):
    x_continuous = metatree_sample_data['x_continuous']
    x_categorical = metatree_sample_data['x_categorical']
    y_continuous = metatree_sample_data['y_continuous']
    y_categorical = metatree_sample_data['y_categorical']

    # initialise the model
    model = metatree.LearnModel(
        c_dim_continuous=3,
        c_dim_categorical=2,
        SubModel=linearregression,
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

def test_metatree_bernoulli_batch_pred(metatree_sample_data):
    x_continuous = metatree_sample_data['x_continuous']
    x_categorical = metatree_sample_data['x_categorical']
    y_continuous = metatree_sample_data['y_continuous']
    y_categorical = metatree_sample_data['y_categorical']

    # initialise the model
    model = metatree.LearnModel(
        c_dim_continuous=3,
        c_dim_categorical=2,
        SubModel=bernoulli,
    )
    # update the posterior distribution
    model.update_posterior(
        x_continuous=x_continuous,
        x_categorical=x_categorical,
        y=y_categorical,
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
    pred_values = model.make_prediction(loss='0-1')
    # desired prediction values
    # the values have been calculated using the same model and parameters, but as sequential predictions
    desireble_pred_values = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    # check if the prediction values are close to the desired values
    assert np.all(np.isclose(pred_values, desireble_pred_values)), f"Prediction values are incorrect: {pred_values} != {desireble_pred_values}"

    # calculate the prediction values
    pred_values = model.make_prediction(loss='KL')
    # desired prediction values
    # the values have been calculated using the same model and parameters, but as sequential predictions
    desireble_pred_values = np.array(
        [[0.33298576, 0.66701424],
        [0.69952726, 0.30047274],
        [0.84431809, 0.15568191],
        [0.84643166, 0.15356834],
        [0.8445064 , 0.1554936 ],
        [0.62956863, 0.37043137],
        [0.84574869, 0.15425131],
        [0.82346626, 0.17653374],
        [0.3595937 , 0.6404063 ],
        [0.59751374, 0.40248626]])
    # check if the prediction values are close to the desired values
    assert np.all(np.isclose(pred_values, desireble_pred_values)), f"Prediction values are incorrect: {pred_values} != {desireble_pred_values}"

    ##############################
    # test on prediction densities
    ##############################
    # calculate the prediction densities
    pred_densities = model.calc_pred_density(np.arange(2)[:,np.newaxis])
    # desired prediction densities
    desireble_pred_densities = np.array(
        [[0.33298576, 0.66701424],
        [0.69952726, 0.30047274],
        [0.84431809, 0.15568191],
        [0.84643166, 0.15356834],
        [0.8445064 , 0.1554936 ],
        [0.62956863, 0.37043137],
        [0.84574869, 0.15425131],
        [0.82346626, 0.17653374],
        [0.3595937 , 0.6404063 ],
        [0.59751374, 0.40248626]]).T
    # check if the prediction densities are close to the desired values
    assert np.all(np.isclose(pred_densities, desireble_pred_densities)), f"Prediction densities are incorrect: {pred_densities} != {desireble_pred_densities}"

def test_metatree_categorical_batch_pred(metatree_sample_data):
    x_continuous = metatree_sample_data['x_continuous']
    x_categorical = metatree_sample_data['x_categorical']
    y_continuous = metatree_sample_data['y_continuous']
    y_categorical = metatree_sample_data['y_categorical']

    # initialise the model
    model = metatree.LearnModel(
        c_dim_continuous=3,
        c_dim_categorical=2,
        SubModel=categorical,
        sub_constants={'c_degree':3},
    )
    # update the posterior distribution
    model.update_posterior(
        x_continuous=x_continuous,
        x_categorical=x_categorical,
        y=y_categorical,
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
    pred_values = model.make_prediction(loss='0-1')
    # desired prediction values
    # the values have been calculated using the same model and parameters, but as sequential predictions
    desireble_pred_values = np.array(
        [1, 0, 0, 0, 0, 0, 0, 0, 1, 0])
    # check if the prediction values are close to the desired values
    assert np.all(np.isclose(pred_values, desireble_pred_values)), f"Prediction values are incorrect: {pred_values} != {desireble_pred_values}"

    # calculate the prediction values
    pred_values = model.make_prediction(loss='KL')
    # desired prediction values
    # the values have been calculated using the same model and parameters, but as sequential predictions
    desireble_pred_values = np.array(
        [[0.35627564, 0.54269806, 0.1010263 ],
         [0.65728076, 0.27916022, 0.06355903],
         [0.76414301, 0.17634316, 0.05951382],
         [0.76537173, 0.1754697 , 0.05915857],
         [0.76452273, 0.17636313, 0.05911414],
         [0.61047234, 0.3179039 , 0.07162376],
         [0.76485687, 0.17597582, 0.0591673 ],
         [0.74948527, 0.19151863, 0.0589961 ],
         [0.37772062, 0.52622695, 0.09605243],
         [0.5943782 , 0.33592791, 0.06969389]])
    # check if the prediction values are close to the desired values
    assert np.all(np.isclose(pred_values, desireble_pred_values)), f"Prediction values are incorrect: {pred_values} != {desireble_pred_values}"

    ##############################
    # test on prediction densities
    ##############################
    # calculate the prediction densities
    pred_densities = model.calc_pred_density(np.arange(3)[:,np.newaxis])
    # desired prediction densities
    desireble_pred_densities = np.array(
        [[0.35627564, 0.54269806, 0.1010263 ],
         [0.65728076, 0.27916022, 0.06355903],
         [0.76414301, 0.17634316, 0.05951382],
         [0.76537173, 0.1754697 , 0.05915857],
         [0.76452273, 0.17636313, 0.05911414],
         [0.61047234, 0.3179039 , 0.07162376],
         [0.76485687, 0.17597582, 0.0591673 ],
         [0.74948527, 0.19151863, 0.0589961 ],
         [0.37772062, 0.52622695, 0.09605243],
         [0.5943782 , 0.33592791, 0.06969389]]).T
    # check if the prediction densities are close to the desired values
    assert np.all(np.isclose(pred_densities, desireble_pred_densities)), f"Prediction densities are incorrect: {pred_densities} != {desireble_pred_densities}"

def test_metatree_poisson_batch_pred(metatree_sample_data):
    x_continuous = metatree_sample_data['x_continuous']
    x_categorical = metatree_sample_data['x_categorical']
    y_continuous = metatree_sample_data['y_continuous']
    y_categorical = metatree_sample_data['y_categorical']

    # initialise the model
    model = metatree.LearnModel(
        c_dim_continuous=3,
        c_dim_categorical=2,
        SubModel=poisson,
    )
    # update the posterior distribution
    model.update_posterior(
        x_continuous=x_continuous,
        x_categorical=x_categorical,
        y=y_categorical,
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
        [0.67354393, 0.36882062, 0.27093895, 0.27142963, 0.27083067,
         0.40199871, 0.27170506, 0.28673035, 0.64816579, 0.42163253])
    # check if the prediction values are close to the desired values
    assert np.all(np.isclose(pred_values, desireble_pred_values)), f"Prediction values are incorrect: {pred_values} != {desireble_pred_values}"

    ##############################
    # test on prediction densities
    ##############################
    # calculate the prediction densities
    pred_densities = model.calc_pred_density(np.arange(2)[:,np.newaxis])
    # desired prediction densities
    desireble_pred_densities = np.array(
        [[0.56255316, 0.71529643, 0.77638916, 0.77599045, 0.77639383,
          0.69650307, 0.77585311, 0.76625645, 0.57362505, 0.68548729],
         [0.28051919, 0.2188778 , 0.18365569, 0.18394869, 0.18370441,
          0.22826409, 0.18399693, 0.18968977, 0.27729537, 0.23370997]])
    # check if the prediction densities are close to the desired values
    assert np.all(np.isclose(pred_densities, desireble_pred_densities)), f"Prediction densities are incorrect: {pred_densities} != {desireble_pred_densities}"

def test_metatree_exponential_batch_pred(metatree_sample_data):
    x_continuous = metatree_sample_data['x_continuous']
    x_categorical = metatree_sample_data['x_categorical']
    y_continuous = metatree_sample_data['y_continuous']
    y_categorical = metatree_sample_data['y_categorical']

    # initialise the model
    model = metatree.LearnModel(
        c_dim_continuous=3,
        c_dim_categorical=2,
        SubModel=exponential,
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
        [0.43923559, 0.4472961 , 0.473235  , 0.48113332, 0.42870636,
         0.44227565, 0.43145572, 0.44318417, 0.45067431, 0.43933846])
    # check if the prediction values are close to the desired values
    assert np.all(np.isclose(pred_values, desireble_pred_values)), f"Prediction values are incorrect: {pred_values} != {desireble_pred_values}"

    ##############################
    # test on prediction densities
    ##############################
    # calculate the prediction densities
    pred_densities = model.calc_pred_density(np.arange(1,3)[:,np.newaxis])
    # desired prediction densities
    desireble_pred_densities = np.array(
        [[0.20178678, 0.20684251, 0.21068319, 0.21083722, 0.20291791,
          0.20500958, 0.20031508, 0.20336086, 0.20712402, 0.20584713],
         [0.02684789, 0.02820515, 0.03105924, 0.03131836, 0.02607321,
          0.02764364, 0.02602114, 0.02710114, 0.02843944, 0.02747074]])
    # check if the prediction densities are close to the desired values
    assert np.all(np.isclose(pred_densities, desireble_pred_densities)), f"Prediction densities are incorrect: {pred_densities} != {desireble_pred_densities}"

if __name__ == "__main__":
    pytest.main()