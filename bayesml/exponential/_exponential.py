# Code Author
# Luyu Ruan <zzstarsound@gmail.com>
# Koshi Shimada <shimada.koshi.re@gmail.com>
# Yuji Iikubo <yuji-iikubo.8@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Yuji Iikubo <yuji-iikubo.8@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
import warnings
import numpy as np
from scipy.stats import expon as ss_expon, gamma as ss_gamma, lomax as ss_lomax
from scipy.special import gammaln
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution.

    Parameters
    ----------
    lambda_ : float, optional
        a positive real number, by default 1.0.
    h_alpha : float, optional
        a positive real number, by default 1.0. 
    h_beta : float, optional
        a positive real number, by default 1.0. 
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(self,lambda_=1.0,h_alpha=1.0,h_beta=1.0,seed=None):
        self.rng = np.random.default_rng(seed)

        # params
        self.lambda_ = 1.0

        # h_params
        self.h_alpha = 1.0
        self.h_beta = 1.0

        self.set_params(lambda_)
        self.set_h_params(h_alpha,h_beta)

    def get_constants(self):
        """Get constants of GenModel.

        This model does not have any constants. 
        Therefore, this function returns an emtpy dict ``{}``.
        
        Returns
        -------
        constants : an empty dict
        """
        return {}

    def set_h_params(self,h_alpha=None,h_beta=None):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h_alpha : float, optional
            a positive real number, by default None.
        h_beta : float, optional
            a positive real number, by default None.
        """
        if h_alpha is not None:
            self.h_alpha = _check.pos_float(h_alpha,'h_alpha',ParameterFormatError)
        if h_beta is not None:
            self.h_beta = _check.pos_float(h_beta,'h_beta',ParameterFormatError)
        return self

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float}
            * ``"h_alpha"`` : The value of ``self.h_alpha``
            * ``"h_beta"`` : The value of ``self.h_beta``
        """
        return {"h_alpha":self.h_alpha, "h_beta":self.h_beta}
    
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.lambda_``.
        """
        self.lambda_ = self.rng.gamma(self.h_alpha,1.0/self.h_beta)
        return self

    def set_params(self,lambda_=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        lambda_ : float, optional
            a positive real number, by default None.
        """
        if lambda_ is not None:
            self.lambda_ = _check.pos_float(lambda_, 'lambda_', ParameterFormatError)
        return self

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:float}
            * ``"lambda_"`` : The value of ``self.lambda_``.
        """
        return {"lambda_":self.lambda_}

    def gen_sample(self,sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            1 dimensional array whose size is ``sammple_size`` and elements are positive real numbers.
        """
        _check.pos_int(sample_size, 'sample_size', DataFormatError)
        return self.rng.exponential(1.0/self.lambda_, sample_size)
        
    def save_sample(self,filename,sample_size):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as aNpzFile with keyword: \"x\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int
            A positive integer
        
        See Also
        --------
        numpy.savez_compressed
        """
        np.savez_compressed(filename,x=self.gen_sample(sample_size))

    def visualize_model(self,sample_size=100,hist_bins=10):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 100.
        hist_bins : float, optional
            A positive float, by default 10.

        Examples
        --------
        >>> from bayesml import normal
        >>> model = normal.GenModel()
        >>> model.visualize_model()
        lambda_:1.0

        .. image:: ./images/exponential_example.png
        """
        _check.pos_int(sample_size, 'sample_size', DataFormatError)
        _check.pos_int(hist_bins, 'hist_bins', DataFormatError)
        sample = self.gen_sample(sample_size)

        print(f"lambda_:{self.lambda_}")

        fig, ax = plt.subplots()
        ax.hist(sample,density=True,label=f"normalized hist n={sample_size}",bins=hist_bins)

        x = np.linspace(1.0e-8,
                        sample.max()+(sample.max()-sample.min())*0.25,
                        100)
        y = ss_expon.pdf(x,scale=1.0/self.lambda_)
        plt.plot(x, y, label=f"Exponential PDF lambda_={self.lambda_}")

        ax.set_xlabel("Realization")
        ax.set_ylabel("Probability or frequency")
        plt.title(f"PDF and normalized histogram")
        plt.legend()
        plt.show()


class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    h0_alpha : float, optional
        a positive real number, by default 1.0.
    h0_beta : float, optional
        a positive real number, by default 1.0.

    Attributes
    ----------
    hn_alpha : float
        a positive real number
    hn_beta : float
        a positive real number
    p_kappa : float
        a positive real number
    p_lambda : float
        a positive real number
    """
    def __init__(self,h0_alpha=1.0, h0_beta=1.0):
        # h0_params
        self.h0_alpha = 1.0
        self.h0_beta = 1.0

        # hn_params
        self.hn_alpha = 1.0
        self.hn_beta = 1.0

        # p_params
        self.p_kappa = 1.0
        self.p_lambda = 1.0

        self.set_h0_params(h0_alpha,h0_beta)
    
    def get_constants(self):
        """Get constants of LearnModel.

        This model does not have any constants. 
        Therefore, this function returns an emtpy dict ``{}``.
        
        Returns
        -------
        constants : an empty dict
        """
        return {}

    def set_h0_params(self,h0_alpha=None,h0_beta=None):
        """Set initial values of the hyperparameter of the posterior distribution.

        Parameters
        ----------
        h0_alpha : float, optional
            a positive real number, by default None.
        h0_beta : float, optional
            a positive real number, by default None.
        """
        if h0_alpha is not None:
            self.h0_alpha = _check.pos_float(h0_alpha, 'h0_alpha', ParameterFormatError)
        if h0_beta is not None:
            self.h0_beta = _check.pos_float(h0_beta, 'h0_beta', ParameterFormatError)
        self.reset_hn_params()
        return self

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: float}
            * ``"h0_alpha"`` : The value of ``self.h0_alpha``
            * ``"h0_beta"`` : The value of ``self.h0_beta``
        """
        return {"h0_alpha":self.h0_alpha, "h0_beta": self.h0_beta}

    def set_hn_params(self,hn_alpha=None,hn_beta=None):
        """Set updated values of the hyperparameter of the posterior distribution.

        Parameters
        ----------
        hn_alpha : float, optional
            a positive real number, by default None.
        hn_beta : float, optional
            a positive real number, by default None.
        """
        if hn_alpha is not None:
            self.hn_alpha = _check.pos_float(hn_alpha, 'hn_alpha', ParameterFormatError)
        if hn_beta is not None:
            self.hn_beta = _check.pos_float(hn_beta, 'hn_beta', ParameterFormatError)
        self.calc_pred_dist()
        return self

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float}
            * ``"hn_alpha"`` : The value of ``self.hn_alpha``
            * ``"hn_beta"`` : The value of ``self.hn_beta``
        """
        return {"hn_alpha":self.hn_alpha, "hn_beta":self.hn_beta}

    def _check_sample(self,x):
        return _check.pos_floats(x, 'x', DataFormatError)

    def update_posterior(self,x):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            All the elements must be positive real numbers.
        """
        x = self._check_sample(x)
        try:
            self.hn_alpha += x.size
        except:
            self.hn_alpha += 1
        self.hn_beta += np.sum(x)
        return self

    def _update_posterior(self,x):
        """Update opsterior without input check."""
        self.hn_alpha += x.size
        self.hn_beta += x.sum()
        return self

    def estimate_params(self,loss="squared",dict_out=False):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        dict_out : bool, optional
            If ``True``, output will be a dict, by default ``False``.

        Returns
        -------
        estimator : {float, None, rv_frozen}
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
                return {'lambda_':self.hn_alpha / self.hn_beta}
            else:
                return self.hn_alpha / self.hn_beta
        elif loss == "0-1":
            if self.hn_alpha > 1.0 :
                if dict_out:
                    return {'lambda_':(self.hn_alpha - 1.0) / self.hn_beta}
                else:
                    return (self.hn_alpha - 1.0) / self.hn_beta
            else:
                if dict_out:
                    return {'lambda_':0.0}
                else:
                    return 0.0
        elif loss == "abs":
            if dict_out:
                return {'lambda_':ss_gamma.median(a=self.hn_alpha,scale=1/self.hn_beta)}
            else:
                return ss_gamma.median(a=self.hn_alpha,scale=1/self.hn_beta)
        elif loss == "KL":
            return ss_gamma(a=self.hn_alpha,scale=1/self.hn_beta)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))
    
    def estimate_interval(self,credibility=0.95):
        """Credible interval of the parameter.

        Parameters
        ----------
        credibility : float, optional
            A posterior probability that the interval conitans the paramter, by default 0.95.

        Returns
        -------
        lower, upper: float
            The lower and the upper bound of the interval
        """
        _check.float_in_closed01(credibility, 'credibility', CriteriaError)
        return ss_gamma.interval(credibility,a=self.hn_alpha,scale=1/self.hn_beta)
    
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.

        Examples
        --------
        >>> from bayesml import exponential
        >>> gen_model = exponential.GenModel(lambda_=2.0)
        >>> x = gen_model.gen_sample(20)
        >>> learn_model = exponential.LearnModel()
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()

        .. image:: ./images/exponential_posterior.png
        """
        x = np.linspace(max(1.0e-8,self.hn_alpha/self.hn_beta-4.0*np.sqrt(self.hn_alpha)/self.hn_beta),
                        self.hn_alpha/self.hn_beta+4.0*np.sqrt(self.hn_alpha)/self.hn_beta,
                        100)
        fig, ax = plt.subplots()
        ax.plot(x,self.estimate_params(loss="KL").pdf(x))
        ax.set_xlabel("lambda")
        ax.set_ylabel("posterior")
        plt.show()
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: float}
            * ``"p_kappa"`` : The value of ``self.p_kappa``
            * ``"p_lambda"`` : The value of ``self.p_lambda``
        """
        return {"p_kappa":self.p_kappa, "p_lambda":self.p_lambda}

    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_kappa = self.hn_alpha
        self.p_lambda = self.hn_beta
        return self

    def _calc_pred_density(self,x):
        return ss_lomax.pdf(x,c=self.p_kappa,scale=self.p_lambda)

    def make_prediction(self,loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".

        Returns
        -------
        Predicted_value : {int, numpy.ndarray}
            The predicted value under the given loss function.
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        if loss == "squared": # Mean of EP
            if self.p_kappa > 1:
                return self.p_lambda / (self.p_kappa-1)
            else:
                warnings.warn("Mean doesn't exist for the current p_kappa.",ResultWarning)
                return None
        elif loss == "0-1": # Mode of EP
            return 0
        elif loss == "abs": # Median of EP
            return self.p_lambda * (2.0**(1.0/self.p_kappa) - 1)
        elif loss == "KL": # EP
            return ss_lomax(c=self.p_kappa,scale=self.p_lambda)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def pred_and_update(self,x,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : float
            a positive real number
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".

        Returns
        -------
        Predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function.
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        _check.pos_float(x,'x',DataFormatError)
        self.calc_pred_dist()
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x)
        return prediction

    def calc_log_marginal_likelihood(self):
        """Calculate log marginal likelihood

        Returns
        -------
        log_marginal_likelihood : float
            The log marginal likelihood.
        """
        return (self.h0_alpha * np.log(self.h0_beta)
                - gammaln(self.h0_alpha)
                - self.hn_alpha * np.log(self.hn_beta)
                + gammaln(self.hn_alpha))