# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
import warnings
import numpy as np
from scipy.stats import gamma as ss_gamma
from scipy.stats import multivariate_t as ss_multivariate_t
from scipy.stats import t as ss_t
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution.

    Parameters
    ----------
    c_degree : int
        a positive integer.
    theta_vec : numpy ndarray, optional
        a vector of real numbers, which includs the constant term, 
        by default [0.0, 0.0, ... , 0.0]
    tau : float, optional
        a positive real number, by default 1.0
    h_mu_vec : numpy ndarray, optional
        a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
    h_lambda_mat : numpy ndarray, optional
        a positive definate matrix, by default the identity matrix
    h_alpha : float, optional
        a positive real number, by default 1.0
    h_beta : float, optional
        a positive real number, by default 1.0
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(
            self,
            c_degree,
            theta_vec=None,
            tau=1.0,
            h_mu_vec=None,
            h_lambda_mat=None,
            h_alpha=1.0,
            h_beta=1.0,
            seed=None
            ):
        # constants
        self.c_degree = _check.nonneg_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.theta_vec = np.zeros(self.c_degree+1)
        self.tau = 1.0

        # h_params
        self.h_mu_vec = np.zeros(self.c_degree+1)
        self.h_lambda_mat = np.eye(self.c_degree+1)
        self.h_alpha = 1.0
        self.h_beta = 1.0

        self.set_params(theta_vec,tau)
        self.set_h_params(h_mu_vec,h_lambda_mat,h_alpha,h_beta)

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int}
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {'c_degree':self.c_degree}

    def set_h_params(self,h_mu_vec=None,h_lambda_mat=None,h_alpha=None,h_beta=None):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h_mu_vec : numpy ndarray, optional
            a vector of real numbers, by default None.
        h_lambda_mat : numpy ndarray, optional
            a positive definate matrix, by default None.
        h_alpha : float, optional
            a positive real number, by default None.
        h_beta : float, optional
            a positive real number, by default None.
        """
        if h_mu_vec is not None:
            _check.float_vec(h_mu_vec,'h_mu_vec',ParameterFormatError)
            _check.shape_consistency(
                h_mu_vec.shape[0],'h_mu_vec.shape[0]',
                self.c_degree+1,'self.c_degree+1',
                ParameterFormatError
            )
            self.h_mu_vec[:] = h_mu_vec

        if h_lambda_mat is not None:
            _check.pos_def_sym_mat(h_lambda_mat,'h_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                h_lambda_mat.shape[0],'h_lambda_mat.shape[0] and h_lambda_mat.shape[1]',
                self.c_degree+1,'self.c_degree+1',
                ParameterFormatError
            )
            self.h_lambda_mat[:] = h_lambda_mat

        if h_alpha is not None:
            self.h_alpha = _check.pos_float(h_alpha,'h_alpha',ParameterFormatError)
        if h_beta is not None:
            self.h_beta = _check.pos_float(h_beta,'h_beta',ParameterFormatError)

        return self

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float or numpy ndarray}
            * ``"h_mu_vec"`` : The value of ``self.h_mu_vec``
            * ``"h_lambda_mat"`` : The value of ``self.h_lambda_mat``
            * ``"h_alpha"`` : The value of ``self.h_alpha``
            * ``"h_beta"`` : The value of ``self.h_beta``
        """
        return {"h_mu_vec":self.h_mu_vec, "h_lambda_mat":self.h_lambda_mat, "h_alpha":self.h_alpha, "h_beta":self.h_beta}
    
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.theta_vec`` and ``self.tau.
        """
        self.tau = self.rng.gamma(shape=self.h_alpha,scale=1.0/self.h_beta)
        self.theta_vec = self.rng.multivariate_normal(mean=self.h_mu_vec,cov=np.linalg.inv(self.tau*self.h_lambda_mat))
        return self
        
    def set_params(self,theta_vec=None, tau=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        theta_vec : numpy ndarray, optional
            a vector of real numbers, by default None
        tau : float, optional, optional
            a positive real number, by default None
        """
        if theta_vec is not None:
            _check.float_vec(theta_vec,'theta_vec',ParameterFormatError)
            _check.shape_consistency(
                theta_vec.shape[0],'theta_vec.shape[0]',
                self.c_degree+1,'self.c_degree+1',
                ParameterFormatError
            )
            self.theta_vec[:] = theta_vec

        if tau is not None:
            self.tau = _check.pos_float(tau,'tau',ParameterFormatError)

        return self

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str: float or numpy ndarray}
            * ``"theta_vec"`` : The value of ``self.theta_vec``.
            * ``"tau"`` : The value of ``self.tau``.
        """
        return {"theta_vec":self.theta_vec, "tau":self.tau}

    def gen_sample(self,sample_length,initial_values=None):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_length : int
            A positive integer
        initial_valules : numpy ndarray, optional
            1 dimensional float array whose size coincide with ``self.c_degree``, 
            by default None.

        Returns
        -------
        x : numpy ndarray
            1 dimensional float array whose size is ``sammple_length``.
        """
        _check.pos_int(sample_length,'sample_length',DataFormatError)
        x = np.zeros(sample_length+self.c_degree)
        if initial_values is not None:
            _check.float_vec(initial_values,'initial_values',DataFormatError)
            if initial_values.shape != (self.c_degree,):
                raise(DataFormatError("initial_values must be a 1 dimensional float array whose size coincide with ``self.c_degree``"))
            x[:self.c_degree] = initial_values
        
        _explanatory_vec = np.ones(self.c_degree+1)
        for n in range(self.c_degree,sample_length+self.c_degree):
            _explanatory_vec[1:] = x[n-self.c_degree:n]
            x[n] = self.rng.normal(loc=self.theta_vec @ _explanatory_vec, scale=1.0/np.sqrt(self.tau))

        return x[self.c_degree:]
        
    def save_sample(self,filename,sample_length,initial_values=None):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_length : int
            A positive integer
        initial_valules : numpy ndarray, optional
            1 dimensional float array whose size coincide with ``self.c_degree``, 
            by default None.
        
        See Also
        --------
        numpy.savez_compressed
        """
        np.savez_compressed(filename,x=self.gen_sample(sample_length,initial_values))

    def visualize_model(self,sample_length=50,sample_num=5,initial_values=None):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_length : int, optional
            A positive integer, by default 50
        sample_num : int, optional
            A positive integer, by default 5
        initial_valules : numpy ndarray, optional
            1 dimensional float array whose size coincide with ``self.c_degree``, 
            by default None.
        
        Examples
        --------
        >>> import numpy as np
        >>> from bayesml import autoregressive
        >>> model = autoregressive.GenModel(c_degree=1,theta_vec=np.array([0,1]))
        >>> model.visualize_model()
        theta_vec:[0,1]
        tau:1.0

        .. image:: ./images/autoregressive_example.png
        """
        _check.pos_int(sample_length,'sample_length',DataFormatError)
        _check.pos_int(sample_num,'sample_num',DataFormatError)
        print(f"theta_vec:{self.theta_vec}")
        print(f"tau:{self.tau}")
        fig, ax = plt.subplots()
        for i in range(sample_num):
            x = self.gen_sample(sample_length,initial_values)
            ax.plot(x)
        ax.set_ylabel("x_n")
        ax.set_xlabel("Time")
        plt.show()

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_degree : int
        a positive integer.
    h0_mu_vec : numpy ndarray, optional
        a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
    h0_lambda_mat : numpy ndarray, optional
        a positive definate matrix, by default the identity matrix
    h0_alpha : float, optional
        a positive real number, by default 1.0
    h0_beta : float, optional
        a positive real number, by default 1.0

    Attributes
    ----------
    hn_mu_vec : numpy ndarray
        a vector of real numbers
    hn_lambda_mat : numpy ndarray
        a positive definate matrix
    hn_alpha : float
        a positive real number
    hn_beta : float
        a positive real number
    p_m : float
        a positive real number
    p_lambda : float
        a positive real number
    p_nu : float
        a positive real number
    """
    def __init__(
            self,
            c_degree,
            h0_mu_vec=None,
            h0_lambda_mat=None,
            h0_alpha=1.0,
            h0_beta=1.0,
            ):
        # constants
        self.c_degree = _check.nonneg_int(c_degree,'c_degree',ParameterFormatError)

        # h0_params
        self.h0_mu_vec = np.zeros(self.c_degree+1)
        self.h0_lambda_mat = np.eye(self.c_degree+1)
        self.h0_alpha = 1.0
        self.h0_beta = 1.0

        # hn_params
        self.hn_mu_vec = np.zeros(self.c_degree+1)
        self.hn_lambda_mat = np.eye(self.c_degree+1)
        self.hn_alpha = 1.0
        self.hn_beta = 1.0

        # p_params
        self.p_m = 0.0
        self.p_lambda = 0.5
        self.p_nu = 2.0

        self.set_h0_params(
            h0_mu_vec,
            h0_lambda_mat,
            h0_alpha,
            h0_beta,
        )

    def get_constants(self):
        """Get constants of LearnModel.

        Returns
        -------
        constants : dict of {str: int}
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {'c_degree':self.c_degree}

    def set_h0_params(
            self,
            h0_mu_vec=None,
            h0_lambda_mat=None,
            h0_alpha=None,
            h0_beta=None,
            ):
        """Set initial values of the hyperparameter of the posterior distribution.

        Note that the parameters of the predictive distribution are also calculated from 
        ``self.h0_mu_vec``, ``slef.h0_lambda_mat``, ``self.h0_alpha`` and ``self.h0_beta``.

        Parameters
        ----------
        h0_mu_vec : numpy ndarray, optional
            a vector of real numbers, by default None.
        h0_lambda_mat : numpy ndarray, optional
            a positive definate matrix, by default None.
        h0_alpha : float, optional
            a positive real number, by default None.
        h0_beta : float, optional
            a positive real number, by default None.
        """
        if h0_mu_vec is not None:
            _check.float_vec(h0_mu_vec,'h0_mu_vec',ParameterFormatError)
            _check.shape_consistency(
                h0_mu_vec.shape[0],'h0_mu_vec.shape[0]',
                self.c_degree+1,'self.c_degree+1',
                ParameterFormatError
            )
            self.h0_mu_vec[:] = h0_mu_vec

        if h0_lambda_mat is not None:
            _check.pos_def_sym_mat(h0_lambda_mat,'h0_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                h0_lambda_mat.shape[0],'h0_lambda_mat.shape[0] and h0_lambda_mat.shape[1]',
                self.c_degree+1,'self.c_degree+1',
                ParameterFormatError
            )
            self.h0_lambda_mat[:] = h0_lambda_mat

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
        h0_params : dict of {str: float or numpy ndarray}
            * ``"h0_mu_vec"`` : The value of ``self.h0_mu_vec``
            * ``"h0_lambda_mat"`` : The value of ``self.h0_lambda_mat``
            * ``"h0_alpha"`` : The value of ``self.h0_alpha``
            * ``"h0_beta"`` : The value of ``self.h0_beta``
        """
        return {"h0_mu_vec":self.h0_mu_vec, "h0_lambda_mat":self.h0_lambda_mat, "h0_alpha":self.h0_alpha, "h0_beta":self.h0_beta}
    
    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float or numpy ndarray}
            * ``"hn_mu_vec"`` : The value of ``self.hn_mu_vec``
            * ``"hn_lambda_mat"`` : The value of ``self.hn_lambda_mat``
            * ``"hn_alpha"`` : The value of ``self.hn_alpha``
            * ``"hn_beta"`` : The value of ``self.hn_beta``
        """
        return {"hn_mu_vec":self.hn_mu_vec, "hn_lambda_mat":self.hn_lambda_mat, "hn_alpha":self.hn_alpha, "hn_beta":self.hn_beta}
    
    def set_hn_params(
            self,
            hn_mu_vec=None,
            hn_lambda_mat=None,
            hn_alpha=None,
            hn_beta=None,
            ):
        """Set updated values of the hyperparameter of the posterior distribution.

        Note that the parameters of the predictive distribution are also calculated from 
        ``self.hn_mu_vec``, ``slef.hn_lambda_mat``, ``self.hn_alpha`` and ``self.hn_beta``.

        Parameters
        ----------
        hn_mu_vec : numpy ndarray, optional
            a vector of real numbers, by default None.
        hn_lambda_mat : numpy ndarray, optional
            a positive definate matrix, by default None.
        hn_alpha : float, optional
            a positive real number, by default None.
        hn_beta : float, optional
            a positive real number, by default None.
        """
        if hn_mu_vec is not None:
            _check.float_vec(hn_mu_vec,'hn_mu_vec',ParameterFormatError)
            _check.shape_consistency(
                hn_mu_vec.shape[0],'hn_mu_vec.shape[0]',
                self.c_degree+1,'self.c_degree+1',
                ParameterFormatError
            )
            self.hn_mu_vec[:] = hn_mu_vec

        if hn_lambda_mat is not None:
            _check.pos_def_sym_mat(hn_lambda_mat,'hn_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                hn_lambda_mat.shape[0],'hn_lambda_mat.shape[0] and hn_lambda_mat.shape[1]',
                self.c_degree+1,'self.c_degree+1',
                ParameterFormatError
            )
            self.hn_lambda_mat[:] = hn_lambda_mat

        if hn_alpha is not None:
            self.hn_alpha = _check.pos_float(hn_alpha,'hn_alpha',ParameterFormatError)
        if hn_beta is not None:
            self.hn_beta = _check.pos_float(hn_beta,'hn_beta',ParameterFormatError)

        self.calc_pred_dist(np.zeros(self.c_degree))
        return self

    def update_posterior(self,x,padding=None):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy ndarray
            1 dimensional float array
        padding : str, optional
            Padding option for data values at negative time points.
            Default is ``None``, in which case the first ``self.c_degree`` values of ``X`` are used as initial values.
            If \"zeros\" is given, the zero vector is used as a initial value.
        """
        _check.float_vec(x,'x',DataFormatError)
        n = x.shape[0]
        if n <= self.c_degree:
            raise(DataFormatError("The length of x must greater than self.c_degree"))
        x_mat = np.zeros((n,self.c_degree+1))
        x_mat[:,0] = 1.0
        for i in range(1,self.c_degree+1):
            x_mat[i,-i:] = x[:i]
        for i in range(self.c_degree+1,n):
            x_mat[i,1:] = x[i-self.c_degree:i]
        
        mu_tmp = np.array(self.hn_mu_vec)
        lambda_tmp = np.array(self.hn_lambda_mat)
        if padding == "zeros":
            self.hn_lambda_mat += x_mat.T @ x_mat
            self.hn_mu_vec[:] = np.linalg.solve(self.hn_lambda_mat,
                                                lambda_tmp @ mu_tmp + x_mat.T @ x)
            self.hn_alpha += n / 2.0
            self.hn_beta += (-self.hn_mu_vec @ self.hn_lambda_mat @ self.hn_mu_vec
                            + x @ x + mu_tmp @ lambda_tmp @ mu_tmp) / 2.0
        else:
            self.hn_lambda_mat += x_mat[self.c_degree:].T @ x_mat[self.c_degree:]
            self.hn_mu_vec[:] = np.linalg.solve(self.hn_lambda_mat,
                                                lambda_tmp @ mu_tmp + x_mat[self.c_degree:].T @ x[self.c_degree:])
            self.hn_alpha += (n-self.c_degree) / 2.0
            self.hn_beta += (-self.hn_mu_vec @ self.hn_lambda_mat @ self.hn_mu_vec 
                            + x[self.c_degree:] @ x[self.c_degree:] + mu_tmp @ lambda_tmp @ mu_tmp) / 2.0
        return self

    def estimate_params(self,loss="squared"):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Note that the criterion is applied to estimating ``theta_vec`` and ``tau`` independently.
        Therefore, a tuple of the student's t-distribution and the gamma distribution will be returned when loss=\"KL\"

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".

        Returns
        -------
        Estimates : tuple of {numpy ndarray, float, None, or rv_frozen}
            * ``theta_vec_hat`` : the estimate for theta_vec
            * ``tau_hat`` : the estimate for tau
            The estimated values under the given loss function. If it is not exist, `None` will be returned.
            If the loss function is \"KL\", the posterior distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        if loss == "squared":
            return self.hn_mu_vec, self.hn_alpha / self.hn_beta
        elif loss == "0-1":
            if self.hn_alpha >= 1.0:
                return self.hn_mu_vec, (self.hn_alpha - 1.0) / self.hn_beta
            else:
                return self.hn_mu_vec, 0
        elif loss == "abs":
            return self.hn_mu_vec, ss_gamma.median(a=self.hn_alpha,scale=1.0/self.hn_beta)
        elif loss == "KL":
            return (ss_multivariate_t(loc=self.hn_mu_vec,
                                        shape=np.linalg.inv(self.hn_alpha / self.hn_beta * self.hn_lambda_mat),
                                        df=2.0*self.hn_alpha),
                    ss_gamma(a=self.hn_alpha,scale=1.0/self.hn_beta))
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))
    
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import autoregressive
        >>> gen_model = autoregressive.GenModel(c_degree=1,theta_vec=np.array([0,1]),tau=1.0)
        >>> x = gen_model.gen_sample(50)
        >>> learn_model = autoregressive.LearnModel()
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()
        
        .. image:: ./images/autoregressive_posterior.png
        """
        if self.c_degree != 1:
            raise(ParameterFormatError("if self.c_degree != 1, it is impossible to visualize posterior by this function."))
        theta_vec_pdf, tau_pdf = self.estimate_params(loss="KL")
        hn_lambda_mat_inv = np.linalg.inv(self.hn_lambda_mat)

        fig, axes = plt.subplots(1,2)
        # for theta_vec
        x = np.linspace(self.hn_mu_vec[0]-3.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_mat_inv[0,0]),
                        self.hn_mu_vec[0]+3.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_mat_inv[0,0]),
                        100)
        y = np.linspace(self.hn_mu_vec[1]-3.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_mat_inv[1,1]),
                        self.hn_mu_vec[1]+3.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_mat_inv[1,1]),
                        100)
        xx, yy = np.meshgrid(x,y)
        grid = np.empty((100,100,2))
        grid[:,:,0] = xx
        grid[:,:,1] = yy
        axes[0].contourf(xx,yy,theta_vec_pdf.pdf(grid))
        axes[0].plot(self.hn_mu_vec[0],self.hn_mu_vec[1],marker="x",color="red")
        axes[0].set_xlabel("theta_vec[0]")
        axes[0].set_ylabel("theta_vec[1]")

        #for tau
        x = np.linspace(max(1.0e-8,self.hn_alpha/self.hn_beta-4.0*np.sqrt(self.hn_alpha)/self.hn_beta),
                        self.hn_alpha/self.hn_beta+4.0*np.sqrt(self.hn_alpha)/self.hn_beta,
                        100)
        axes[1].plot(x,tau_pdf.pdf(x))
        axes[1].set_xlabel("tau")
        axes[1].set_ylabel("posterior probability")

        fig.tight_layout()
        plt.show()
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: float}
            * ``"p_m"`` : The value of ``self.p_m``
            * ``"p_lambda"`` : The value of ``self.p_lambda``
            * ``"p_nu"`` : The value of ``self.p_nu``
        """
        return {"p_m":self.p_m, "p_lambda":self.p_lambda, "p_nu":self.p_nu}

    def calc_pred_dist(self,x):
        """Calculate the parameters of the predictive distribution.
        
        Parameters
        ----------
        x : numpy ndarray
            1 dimensional float array whose size is ``self.c_degree``
        """
        _check.float_vec(x,'x',DataFormatError)
        if x.shape != (self.c_degree,):
            raise(DataFormatError("x must be a 1 dimensional float array whose size coincide with ``self.c_degree``"))
        _explanatory_vec = np.ones(self.c_degree+1)
        _explanatory_vec[1:] = x
        self.p_m = self.hn_mu_vec @ _explanatory_vec
        self.p_lambda = self.hn_alpha / self.hn_beta / (1.0 + _explanatory_vec @ np.linalg.solve(self.hn_lambda_mat,_explanatory_vec))
        self.p_nu = 2.0 * self.hn_alpha
        return self

    def make_prediction(self,loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".

        Returns
        -------
        Predicted_value : {float, rv_frozen}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        if loss == "squared" or loss == "0-1" or loss == "abs":
            return self.p_m
        elif loss == "KL":
            return ss_t(loc=self.p_m, scale=1.0/np.sqrt(self.p_lambda), df=self.p_nu)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def predict_interval(self,credibility=0.95):
        """Credible interval of the prediction.

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
        return ss_t.interval(alpha=credibility,loc=self.p_m, scale=1.0/np.sqrt(self.p_lambda), df=self.p_nu)

    def pred_and_update(self,x,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : numpy ndarray
            1 dimensional float array whose size is ``self.c_degree + 1``,
            which consists of the ``self.c_degree`` number of past values
            and the current value.
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".

        Returns
        -------
        Predicted_value : {float, rv_frozen}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as rv_frozen object of scipy.stats.
        """
        _check.float_vec(x,'x',DataFormatError)
        if x.shape != (self.c_degree+1,):
            raise(DataFormatError("x must be a 1 dimensional float array whose size coincide with ``self.c_degree + 1``"))
        self.calc_pred_dist(x[:self.c_degree])
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x)
        return prediction
