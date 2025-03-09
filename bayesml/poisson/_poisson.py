# Code Author
# Ryota Maniwa <r_maniwa115@fuji.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Ryota Maniwa <r_maniwa115@fuji.waseda.jp>
# Naoki Ichijo <1jonao@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
import numpy as np
from scipy.stats import poisson as ss_poisson
from scipy.stats import gamma as ss_gamma 
from scipy.stats import nbinom as ss_nbinom
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
        a positive real number, by default 1.0
    h_alpha : float, optional
        a positive real number, by default 1.0
    h_beta : float, optional
        a positive real number, by default 1.0
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
            a positive real number, by default None
        h_beta : float, optional
            a positive real number, by default None
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
        self.lambda_ = self.rng.gamma(shape=self.h_alpha,scale=1.0/self.h_beta)
        return self

    def set_params(self,lambda_=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        lambda_ : float, optional
            a positive real number, by default None
        """
        if lambda_ is not None:
            self.lambda_ = _check.pos_float(lambda_,'lambda_',ParameterFormatError)
        return self

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:int}
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
            1 dimensional array whose size is ``sammple_size`` and elements are natural number.
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        return self.rng.poisson(self.lambda_,sample_size)
        
    def save_sample(self,filename,sample_size):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\".

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

    def visualize_model(self,sample_size=20):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 20
        
        Examples
        --------
        >>> from bayesml import poisson
        >>> model = poisson.GenModel()
        >>> model.visualize_model()
        lambda:1.0
        x:[2 1 1 0 0 5 0 2 2 2 1 1 0 1 3 0 0 1 1 0]

        .. image:: ./images/poisson_example.png
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        print(f"lambda:{self.lambda_}")
        fig, ax = plt.subplots()
        x = self.gen_sample(sample_size)
        print(f"x:{x}")
        
        plt.title(f"PMF and normalized histogram")
        ax.hist(x,bins=x.max()-x.min()+1,range=(x.min()-0.5,x.max()+0.5),
                density=True,label=f"normalized hist n={sample_size}",rwidth=1.0)
        y = ss_poisson.pmf(np.arange(x.min(),x.max()+1),self.lambda_)
        plt.plot(np.arange(x.min(),x.max()+1), y, label=f"Poisson PMF λ={self.lambda_}",marker='o')
        ax.set_xlabel("Realization")
        ax.set_ylabel("Probability or frequency")
        
        plt.legend()
        plt.show()

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    h0_alpha : float, optional
        a positive real number, by default 1.0
    h0_beta : float, optional
        a positive real number, by default 1.0
    
    Attributes
    ----------
    hn_alpha : float
        a positive real number
    hn_beta : float
        a positive real number
    p_r : float
        a positive real number
    p_theta : float
        a real number in :math:`[0, 1]`
    """
    def __init__(self,h0_alpha=1.0,h0_beta=1.0):
        # h0_params
        self.h0_alpha = 1.0
        self.h0_beta = 1.0

        # hn_params
        self.hn_alpha = 1.0
        self.hn_beta = 1.0

        #p_params
        self.p_r = 1.0
        self.p_theta = 0.5

        #statistics
        self._sum_log_factorial = 0.0

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
            a positive real number, by default None
        h0_beta : float, optional
            a positive real number, by default None
        """
        if h0_alpha is not None:
            self.h0_alpha = _check.pos_float(h0_alpha,'h0_alpha',ParameterFormatError)
        if h0_beta is not None:
            self.h0_beta = _check.pos_float(h0_beta,'h0_beta',ParameterFormatError)
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
        return {"h0_alpha":self.h0_alpha, "h0_beta":self.h0_beta}

    def set_hn_params(self,hn_alpha=None,hn_beta=None):
        """Set updated values of the hyperparameter of the posterior distribution.
        
        Parameters
        ----------
        hn_alpha : float, optional
            a positive real number, by default None
        hn_beta : float, optional
            a positive real number, by default None
        """
        self._sum_log_factorial = 0.0
        if hn_alpha is not None:
            self.hn_alpha = _check.pos_float(hn_alpha,'hn_alpha',ParameterFormatError)
        if hn_beta is not None:
            self.hn_beta = _check.pos_float(hn_beta,'hn_beta',ParameterFormatError)
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
        return _check.nonneg_ints(x,'x',DataFormatError)

    def update_posterior(self,x):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            All the elements must be non-negative integer.
        """
        x = self._check_sample(x)
        self.hn_alpha += np.sum(x)
        try:
            self.hn_beta += x.size
        except:
            self.hn_beta += 1
        self._sum_log_factorial += gammaln(x+1).sum()
        return self

    def _update_posterior(self,x):
        """Update opsterior without input check."""
        self.hn_alpha += x.sum()
        self.hn_beta += x.size
        self._sum_log_factorial += gammaln(x+1).sum()
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
            A posterior probability that the interval conitans the paramter, by default 0.95

        Returns
        -------
        lower, upper: float
            The lower and the upper bound of the interval
        """
        _check.float_in_closed01(credibility,'credibility',CriteriaError)
        return ss_gamma.interval(credibility,self.hn_alpha,self.hn_beta)
    
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import poisson
        >>> gen_model = poisson.GenModel(lambda_=2.0)
        >>> x = gen_model.gen_sample(20)
        >>> print(x)
        [2 1 6 2 3 1 1 3 2 1 2 1 1 1 2 2 4 2 2 2]
        >>> learn_model = poisson.LearnModel()
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()
        
        .. image:: ./images/poisson_posterior.png
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
            * ``"p_r"`` : The value of ``self.p_r``
            * ``"p_theta"`` : The value of ``self.p_theta``
        """
        return {"p_r":self.p_r, "p_theta":self.p_theta}
    
    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_r = self.hn_alpha
        self.p_theta = 1.0 / (1.0+self.hn_beta)
        return self

    def _calc_pred_density(self,x):
        return ss_nbinom.pmf(x,n=self.p_r,p=(1.0-self.p_theta))

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
        if loss == "squared":
            return self.p_r * self.p_theta / (1.0-self.p_theta)
        elif loss == "0-1":
            if self.p_r > 1.0 :
                return np.floor((self.p_r-1.0) * self.p_theta / (1.0-self.p_theta))
            else:
                return 0
        elif loss == "abs":
            return ss_nbinom.median(n=self.p_r,p=(1.0-self.p_theta))
        elif loss == "KL":
            return ss_nbinom(n=self.p_r,p=(1.0-self.p_theta))
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))
    

    def pred_and_update(self,x,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : int
            It must be natural number
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
        _check.nonneg_int(x,'x',DataFormatError)
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
                + gammaln(self.hn_alpha)
                - self._sum_log_factorial)