# Code Author
# Kohei Horinouchi <horinochi_18@toki.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Kohei Horinouchi <horinochi_18@toki.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
import warnings
import numpy as np
from scipy.stats import dirichlet as ss_dirichlet
from scipy.special import gammaln
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import (
    ParameterFormatError,
    DataFormatError,
    CriteriaError,
    ResultWarning,
    ParameterFormatWarning
    )
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    c_degree : int
        a positive integer.
    theta_vec : numpy ndarray, optional
        a real vector in :math:`[0, 1]^d`, by default [1/d, 1/d, ... , 1/d]
    h_alpha_vec : numpy ndarray, optional
        a vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]. 
        If a single real number is input, it will be broadcasted.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(self,c_degree,theta_vec=None,h_alpha_vec=None,seed=None,):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.theta_vec = np.ones(self.c_degree) / self.c_degree

        # h_params
        self.h_alpha_vec = np.ones(self.c_degree) / 2.0

        self.set_params(theta_vec)
        self.set_h_params(h_alpha_vec)

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int}
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {'c_degree':self.c_degree}
    
    def set_h_params(self,h_alpha_vec=None):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_alpha_vec : numpy ndarray, optional
            a vector of positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        """
        if h_alpha_vec is not None:
            _check.pos_floats(h_alpha_vec,'h_alpha_vec',ParameterFormatError)
            self.h_alpha_vec[:] = h_alpha_vec

        return self

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : {str:numpy ndarray}
            ``{"h_alpha_vec": self.h_alpha_vec}``
        """
        return {"h_alpha_vec": self.h_alpha_vec}

    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.theta_vec``.
        """
        self.theta_vec[:] = self.rng.dirichlet(self.h_alpha_vec)
        return self

    def set_params(self, theta_vec=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        p : numpy ndarray, optional
            a real vector :math:`p \in [0, 1]^d`, by default None.
        """
        if theta_vec is not None:
            _check.float_vec_sum_1(theta_vec,'theta_vec',ParameterFormatError)
            _check.shape_consistency(
                theta_vec.shape[0],'theta_vec.shape[0]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
            )
            self.theta_vec[:] = theta_vec

        return self

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : {str:numpy ndarray}
            ``{"theta_vec":self.theta_vec}``
        """
        return {"theta_vec": self.theta_vec}

    def gen_sample(self, sample_size, onehot=True):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer
        onehot : bool, optional
            If True, a generated sample will be one-hot encoded, 
            by default True.

        Returns
        -------
        x : numpy ndarray
            An non-negative int array. If onehot option is True, its shape will be 
            ``(sample_size,c_degree)`` and each row will be a one-hot vector. 
            If onehot option is False, its shape will be ``(sample_size,)`` 
            and each element will be smaller than self.c_degree.
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        x = self.rng.choice(self.c_degree,sample_size,p=self.theta_vec)
        if onehot:
            return np.eye(self.c_degree,dtype=int)[x]
        else:
            return x

    def save_sample(self, filename, sample_size, onehot=True):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int
            A positive integer
        onehot : bool, optional
            If True, a generated sample will be one-hot encoded, 
            by default True.
        
        See Also
        --------
        numpy.savez_compressed
        """
        np.savez_compressed(filename,x=self.gen_sample(sample_size,onehot))

    def visualize_model(self, sample_size=20, sample_num=5):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 20
        sample_num : int, optional
            A positive integer, by default 5
        
        Examples
        --------
        >>> from bayesml import categorical
        >>> model = categorical.GenModel(3)
        >>> model.visualize_model()
        theta_vec:[0.33333333 0.33333333 0.33333333]

        .. image:: ./images/categorical_example.png
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        _check.pos_int(sample_num,'sample_num',DataFormatError)
        cmap = plt.get_cmap("hsv")
        print(f"theta_vec:{self.theta_vec}",)
        fig, ax = plt.subplots(2,1,figsize=(5, sample_num+1),gridspec_kw={'height_ratios': [1,sample_num]})
        ax[0].set_title("True distribution")
        for j in range(self.c_degree):
            ax[0].barh(0,self.theta_vec[j],left=self.theta_vec[:j].sum(),color=cmap(j / self.c_degree))
        ax[0].tick_params(labelleft=False)
        ax[0].tick_params(left=False)
        ax[1].set_title("Generated sample")
        for i in range(sample_num):
            x = self.gen_sample(sample_size)
            # print(f"x{i}:{x}")
            tmp_sum = 0
            if i == 0:
                for j in range(self.c_degree):
                    count = x[:,j].sum()
                    ax[1].barh(
                        i,
                        count,
                        left=tmp_sum,
                        label=f'{j}',
                        color=cmap(j / self.c_degree),
                    )
                    tmp_sum += count
            else:
                for j in range(self.c_degree):
                    count = x[:,j].sum()
                    ax[1].barh(
                        i,
                        count,
                        left=tmp_sum,
                        color=cmap(j / self.c_degree),
                    )
                    tmp_sum += count
        ax[1].legend()
        ax[1].set_xlabel("Number of occurrences")
        plt.show()

class LearnModel(base.Posterior, base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_degree : int
        a positive integer.
    h0_alpha_vec : numpy.ndarray, optional
        a vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]. 
        If a single real number is input, it will be broadcasted.

    Attributes
    ----------
    hn_alpha_vec : numpy.ndarray
        a vector of positive real numbers
    p_theta_vec : numpy.ndarray
        a real vector in :math:`[0, 1]^d`
    """
    def __init__(self,c_degree,h0_alpha_vec=None):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)

        # h0_params
        self.h0_alpha_vec = np.ones(self.c_degree) / 2.0

        # hn_params
        self.hn_alpha_vec = np.ones(self.c_degree) / 2.0

        # p_params
        self.p_theta_vec = np.ones(self.c_degree) / self.c_degree

        self.set_h0_params(h0_alpha_vec)

    def get_constants(self):
        """Get constants of LearnModel.

        Returns
        -------
        constants : dict of {str: int}
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {'c_degree':self.c_degree}

    def set_h0_params(self,h0_alpha_vec=None):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h0_alpha_vec : numpy ndarray, optional
            a vector of positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        """
        if h0_alpha_vec is not None:
            _check.pos_floats(h0_alpha_vec,'h0_alpha_vec',ParameterFormatError)
            self.h0_alpha_vec[:] = h0_alpha_vec
        self.reset_hn_params()
        return self

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: float, numpy.ndarray}
            ``"h0_alpha_vec"`` : The value of ``self.h0_alpha_vec``
        """
        return {"h0_alpha_vec": self.h0_alpha_vec}

    def set_hn_params(self,hn_alpha_vec=None):
        """Set updated values of the hyperparameter of the posterior distribution.
        
        Parameters
        ----------
        hn_alpha_vec : numpy ndarray, optional
            a vector of positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        """
        if hn_alpha_vec is not None:
            _check.pos_floats(hn_alpha_vec,'hn_alpha_vec',ParameterFormatError)
            self.hn_alpha_vec[:] = hn_alpha_vec
        self.calc_pred_dist()
        return self

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: numpy.ndarray}
            ``"hn_alpha_vec"`` : The value of ``self.hn_alpha_vec``
        """
        return {"hn_alpha_vec": self.hn_alpha_vec}

    # default onehot option is False because it is used in metatree
    def _check_sample(self,x,onehot=False):
        if onehot:
            _check.onehot_vecs(x,'x',DataFormatError)
            if x.shape[-1] != self.c_degree:
                raise(DataFormatError(f"x.shape[-1] must be c_degree:{self.c_degree}"))
            return x.reshape(-1,self.c_degree)
        else:
            _check.nonneg_ints(x,'x',DataFormatError)
            if np.max(x) >= self.c_degree:
                raise(DataFormatError(
                    'np.max(x) must be smaller than self.c_degree: '
                    +f'np.max(x) = {np.max(x)}, self.c_degree = {self.c_degree}'
                ))
            return x

    def update_posterior(self,x,onehot=True):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            A non-negative int array. If onehot option is True, 
            its shape must be ``(sample_size,c_degree)`` and 
            each row must be a one-hot vector. If onehot option is False, 
            its shape must be ``(sample_size,)`` and each element must be 
            smaller than ``self.c_degree``.
        onehot : bool, optional
            If True, the input sample must be one-hot encoded, 
            by default True.
        """
        x = self._check_sample(x,onehot)
        if onehot:
            self.hn_alpha_vec[:] += x.sum(axis=0)
        else:
            for k in range(self.c_degree):
                self.hn_alpha_vec[k] += np.count_nonzero(x==k)
        return self

    def _update_posterior(self,x):
        """Update opsterior without input check."""
        for k in range(self.c_degree):
            self.hn_alpha_vec[k] += np.count_nonzero(x==k)
        return self

    def estimate_params(self, loss="squared",dict_out=False):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".
        dict_out : bool, optional
            If ``True``, output will be a dict, by default ``False``.

        Returns
        -------
        estimates : {numpy ndarray, float, None, or rv_frozen}
            The estimated values under the given loss function. If it is not exist, `None` will be returned.
            If the loss function is \"KL\", the posterior distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        if loss == "squared":
            if dict_out:
                return {'theta_vec':self.hn_alpha_vec / np.sum(self.hn_alpha_vec)}
            else:
                return self.hn_alpha_vec / np.sum(self.hn_alpha_vec)
        elif loss == "0-1":
            if np.all(self.hn_alpha_vec > 1):
                if dict_out:
                    return {'theta_vec':(self.hn_alpha_vec - 1) / (np.sum(self.hn_alpha_vec) - self.c_degree)}
                else:
                    return (self.hn_alpha_vec - 1) / (np.sum(self.hn_alpha_vec) - self.c_degree)
            else:
                warnings.warn("MAP estimate of theta_vec doesn't exist for the current hn_alpha_vec.",ResultWarning)
                if dict_out:
                    return {'theta_vec':None}
                else:
                    return None
        elif loss == "KL":
            return ss_dirichlet(alpha=self.hn_alpha_vec)
        else:
            raise (
                CriteriaError(
                    "Unsupported loss function! "
                    'This function supports "squared", "0-1", and "KL".'
                )
            )

    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import categorical
        >>> gen_model = categorical.GenModel(3)
        >>> x = gen_model.gen_sample(20)
        >>> learn_model = categorical.LearnModel()
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()
        hn_alpha_vec:[6.5 8.5 6.5]

        .. image:: ./images/categorical_posterior.png
        """
        print(f'hn_alpha_vec:{self.hn_alpha_vec}')
        theta_vec_pdf = self.estimate_params(loss='KL')
        fig, axes = plt.subplots()
        step_num = 200
        if self.c_degree == 3:
            theta1_range = np.linspace(0.01, 0.99, step_num)
            theta2_range = np.linspace(0.01, 0.99, step_num)
            X, Y = np.meshgrid(theta1_range, theta2_range)
            Z = np.zeros([step_num,step_num])
            for i in range(step_num):
                for j in range(step_num-i-1):
                    Z[i,j] = theta_vec_pdf.logpdf(np.array([X[i,j],
                                                            Y[i,j],
                                                            1.0-X[i,j]-Y[i,j]]))
            Z_min = Z.min()
            for i in range(step_num):
                Z[i,step_num-i-1:] = Z_min
            axes.contourf(X,Y,Z,cmap='Blues')
            theta_hat = self.estimate_params(loss='0-1')
            if theta_hat is not None:
                axes.scatter(theta_hat[0],theta_hat[1],marker="x",color='red',label='MAP estimate')
                plt.legend()
            axes.set_xlabel("theta_vec[0]")
            axes.set_ylabel("theta_vec[1]")

            plt.title(f"Log PDF for theta_vec")
            plt.show()

        elif self.c_degree == 2:
            theta1_range = np.linspace(0.01, 0.99, step_num)
            Z = np.zeros(step_num)
            for i in range(step_num):
                Z[i] = theta_vec_pdf.pdf(np.array([theta1_range[i],
                                                   1.0-theta1_range[i]]))
            axes.plot(theta1_range,Z)
            plt.show()

        else:
            raise(ParameterFormatError("if c_degree != 2 or c_degree != 3, it is impossible to visualize the model by this function."))

    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy.ndarray}
            ``"p_theta_vec"`` : The value of ``self.p_theta_vec``
        """
        return {"p_theta_vec": self.p_theta_vec}

    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_theta_vec[:] = self.hn_alpha_vec / self.hn_alpha_vec.sum()
        return self
    
    def _calc_pred_density(self,x):
        return self.p_theta_vec[x]

    def make_prediction(self,loss="squared",onehot=True):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".
        onehot : bool, optional
            If True, predected value under \"0-1\" loss will be one-hot encoded, 
            by default True.

        Returns
        -------
        Predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution will be returned
            as 1-dimensional numpy.ndarray that consists of occurence probabilities.
        """
        if loss == "squared":
            return self.p_theta_vec
        elif loss == "0-1":
            if onehot:
                tmp = np.zeros(self.c_degree,dtype=int)
                tmp[np.argmax(self.p_theta_vec)] = 1
                return tmp
            else:
                return np.argmax(self.p_theta_vec)
        elif loss == "KL":
            return self.p_theta_vec
        else:
            raise (
                CriteriaError(
                    "Unsupported loss function! "
                    'This function supports "squared", "0-1", and "KL".'
                )
            )

    def pred_and_update(self, x, loss="squared",onehot=True):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : numpy.ndarray or int
            If onehot option is True, 1-dimensional array whose length is ``c_degree``. 
            If onehot option is False, a non-negative integer.
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".
        onehot : bool, optional
            If True, the input must be one-hot encoded and a predected value 
            under \"0-1\" loss will be one-hot encoded, by default True.

        Returns
        -------
        Predicted_value : {int, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        self.calc_pred_dist()
        prediction = self.make_prediction(loss,onehot)
        self.update_posterior(x,onehot)
        return prediction

    def calc_log_marginal_likelihood(self):
        """Calculate log marginal likelihood

        Returns
        -------
        log_marginal_likelihood : float
            The log marginal likelihood.
        """
        return (gammaln(self.h0_alpha_vec.sum())
                -gammaln(self.h0_alpha_vec).sum()
                -gammaln(self.hn_alpha_vec.sum())
                +gammaln(self.hn_alpha_vec).sum())
