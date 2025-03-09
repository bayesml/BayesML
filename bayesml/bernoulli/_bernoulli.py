# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
import warnings
import numpy as np
from scipy.stats import beta as ss_beta
from scipy.special import gammaln
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution.

    Parameters
    ----------
    theta : float, optional
        a real number in :math:`[0, 1]`, by default 0.5.
    h_alpha : float, optional
        a positive real number, by default 0.5.
    h_beta : float, optional
        a positive real number, by default 0.5.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None.
    """
    def __init__(self,theta=0.5,h_alpha=0.5,h_beta=0.5,seed=None):
        self.rng = np.random.default_rng(seed)

        # params
        self.theta = 0.5

        # h_params
        self.h_alpha = 0.5
        self.h_beta = 0.5

        self.set_params(theta)
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
        
        The generated vaule is set at ``self.theta``.
        """
        self.theta = self.rng.beta(self.h_alpha,self.h_beta)
        return self

    def set_params(self,theta=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        theta : float, optional
            a real number :math:`\theta \in [0, 1]`, by default None.
        """
        if theta is not None:
            self.theta = _check.float_in_closed01(theta,'theta',ParameterFormatError)
        return self

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:float}
            * ``"theta"`` : The value of ``self.theta``.
        """
        return {"theta":self.theta}

    def gen_sample(self,sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            1 dimensional array whose size is ``sammple_size`` and elements are 0 or 1.
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        return self.rng.binomial(1,self.theta,sample_size)
        
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

    def visualize_model(self,sample_size=20,sample_num=5):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 20.
        sample_num : int, optional
            A positive integer, by default 5.
        
        Examples
        --------
        >>> from bayesml import bernoulli
        >>> model = bernoulli.GenModel()
        >>> model.visualize_model()
        theta:0.5
        x0:[1 1 0 0 0 1 0 1 0 0 0 1 0 1 0 1 0 1 0 0]
        x1:[1 1 0 0 0 0 0 1 1 0 0 0 1 0 1 0 0 0 0 0]
        x2:[0 1 0 1 0 0 1 0 0 0 1 0 1 1 1 0 1 0 1 1]
        x3:[0 0 0 1 1 0 1 0 1 0 0 0 1 0 1 0 1 0 1 1]
        x4:[1 0 1 1 1 1 0 1 0 0 1 1 0 0 0 0 0 0 1 1]
        
        .. image:: ./images/bernoulli_example.png
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)
        _check.pos_int(sample_num,'sample_num',DataFormatError)
        print(f"theta:{self.theta}")
        fig, ax = plt.subplots(2,1,figsize=(5, sample_num+1),gridspec_kw={'height_ratios': [1,sample_num]})
        ax[0].set_title("True distribution")
        ax[0].barh(0,self.theta,label='1',color="C0")
        ax[0].barh(0,1.0-self.theta,left=self.theta,label='0',color="C1")
        ax[0].tick_params(labelleft=False)
        ax[0].tick_params(left=False)
        ax[0].legend()
        ax[1].set_title("Generated sample")
        for i in range(sample_num):
            x = self.gen_sample(sample_size)
            print(f"x{i}:{x}")
            if i == 0:
                ax[1].barh(i,x.sum(),label='1',color="C0")
                ax[1].barh(i,sample_size-x.sum(),left=x.sum(),label='0',color="C1")
            else:
                ax[1].barh(i,x.sum(),color="C0")
                ax[1].barh(i,sample_size-x.sum(),left=x.sum(),color="C1")
        ax[1].legend()
        ax[1].set_xlabel("Number of occurrences")
        plt.show()

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    h0_alpha : float, optional
        a positive real number, by default 0.5.
    h0_beta : float, optional
        a positive real number, by default 0.5.
    
    Attributes
    ----------
    hn_alpha : float
        a positive real number
    hn_beta : float
        a positive real number
    p_theta : float
        a real number :math:`\theta_\mathrm{p} \in [0, 1]`
    """
    def __init__(self,h0_alpha=0.5,h0_beta=0.5):
        # h0_params
        self.h0_alpha = 0.5
        self.h0_beta = 0.5

        # hn_params
        self.hn_alpha = 0.5
        self.hn_beta = 0.5

        # p_params
        self.p_theta = 0.5

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
        h0_beta : float, optionanl
            a positive real number, by default None.
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
            a positive real number, by default None.
        hn_beta : float, optional
            a positive real number, by default None.
        """
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
        return _check.ints_of_01(x,'x',DataFormatError)

    def update_posterior(self,x):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy.ndarray
            All the elements must be 0 or 1.
        """
        x = self._check_sample(x)
        self.hn_alpha += np.count_nonzero(x==1)
        self.hn_beta += np.count_nonzero(x==0)
        return self

    def _update_posterior(self,x):
        """Update opsterior without input check."""
        self.hn_alpha += np.count_nonzero(x==1)
        self.hn_beta += np.count_nonzero(x==0)
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
        estimator : {float, None, rv_frozen} or dict of {str : float, None}
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
                return {'theta':self.hn_alpha / (self.hn_alpha + self.hn_beta)}
            else:
                return self.hn_alpha / (self.hn_alpha + self.hn_beta)
        elif loss == "0-1":
            if self.hn_alpha > 1.0 and self.hn_beta > 1.0:
                if dict_out:
                    return {'theta':(self.hn_alpha - 1.0) / (self.hn_alpha + self.hn_beta - 2.0)}
                else:
                    return (self.hn_alpha - 1.0) / (self.hn_alpha + self.hn_beta - 2.0)
            elif self.hn_alpha > 1.0:
                if dict_out:
                    return {'theta':1.0}
                else:
                    return 1.0
            elif self.hn_beta > 1.0:
                if dict_out:
                    return {'theta':0.0}
                else:
                    return 0.0
            else:
                warnings.warn("MAP estimate doesn't exist for the current hn_alpha and hn_beta.",ResultWarning)
                if dict_out:
                    return {'theta':None}
                else:
                    return None
        elif loss == "abs":
            if dict_out:
                return {'theta':ss_beta.median(self.hn_alpha,self.hn_beta)}
            else:
                return ss_beta.median(self.hn_alpha,self.hn_beta)
        elif loss == "KL":
            return ss_beta(self.hn_alpha,self.hn_beta)
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
        _check.float_in_closed01(credibility,'credibility',CriteriaError)
        return ss_beta.interval(credibility,self.hn_alpha,self.hn_beta)
    
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import bernoulli
        >>> gen_model = bernoulli.GenModel()
        >>> x = gen_model.gen_sample(20)
        >>> print(x)
        [0 1 1 0 1 0 0 0 0 1 0 0 0 1 0 1 0 0 1 0]
        >>> learn_model = bernoulli.LearnModel()
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()
        
        .. image:: ./images/bernoulli_posterior.png
        """
        p_range = np.linspace(0,1,100,endpoint=False)
        fig, ax = plt.subplots()
        ax.plot(p_range,self.estimate_params(loss="KL").pdf(p_range))
        ax.set_xlabel("theta")
        ax.set_ylabel("posterior")
        plt.show()
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: float}
            * ``"p_theta"`` : The value of ``self.p_theta``
        """
        return {"p_theta":self.p_theta}
    
    def calc_pred_dist(self):
        """Calculate the parameters of the predictive distribution."""
        self.p_theta = self.hn_alpha / (self.hn_alpha + self.hn_beta)
        return self

    def _calc_pred_density(self,x):
        if x:
            return self.p_theta
        else:
            return 1.0-self.p_theta

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
            return self.p_theta
        elif loss == "0-1" or loss == "abs":
            if self.p_theta > 0.5:
                return 1
            else:
                return 0
        elif loss == "KL":
            return np.array((1.0 - self.p_theta,
                             self.p_theta))
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def pred_and_update(self,x,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : int
            It must be 0 or 1
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
        _check.int_of_01(x,'x',DataFormatError)
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
        return (gammaln(self.h0_alpha+self.h0_beta)
                -gammaln(self.h0_alpha)
                -gammaln(self.h0_beta)
                -gammaln(self.hn_alpha+self.hn_beta)
                +gammaln(self.hn_alpha)
                +gammaln(self.hn_beta))
