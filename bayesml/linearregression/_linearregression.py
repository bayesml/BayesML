# Code Author
# Taisuke Ishiwatari <taisuke.ishiwatari@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Taisuke Ishiwatari <taisuke.ishiwatari@fuji.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
import warnings
import numpy as np
from scipy.stats import gamma as ss_gamma
from scipy.stats import multivariate_t as ss_multivariate_t
from scipy.stats import t as ss_t
from scipy.special import gammaln
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
        a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
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
        theta_vec = None, 
        tau = 1.0,
        h_mu_vec = None, 
        h_lambda_mat = None, 
        h_alpha = 1.0, 
        h_beta = 1.0, 
        seed = None):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.theta_vec = np.zeros(self.c_degree)
        self.tau = 1.0

        # h_params
        self.h_mu_vec = np.zeros(self.c_degree)
        self.h_lambda_mat = np.eye(self.c_degree)
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
                self.c_degree,'self.c_degree',
                ParameterFormatError
            )
            self.h_mu_vec[:] = h_mu_vec

        if h_lambda_mat is not None:
            _check.pos_def_sym_mat(h_lambda_mat,'h_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                h_lambda_mat.shape[0],'h_lambda_mat.shape[0] and h_lambda_mat.shape[1]',
                self.c_degree,'self.c_degree',
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
        self.tau =  self.rng.gamma(shape=self.h_alpha, scale=1.0/self.h_beta)
        self.theta_vec = self.rng.multivariate_normal(mean=self.h_mu_vec,cov=np.linalg.inv(self.tau*self.h_lambda_mat))
        return self

    def set_params(self,theta_vec=None,tau=None):
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
                self.c_degree,'self.c_degree',
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

    def gen_sample(self,sample_size=None,x=None,constant=True):
        """Generate a sample from the stochastic data generative model.

        If x is given, it will be used for explanatory variables as it is 
        (independent of the other options: sample_size and constant).

        If x is not given, it will be generated from i.i.d. standard normal distribution.
        The size of the generated sample is defined by sample_size.
        If constant is True, the last element of the generated explanatory variables will be overwritten by 1.0.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default ``None``.
        x : numpy ndarray, optional
            float array whose shape is ``(sammple_length,c_degree)``, by default ``None``.
        constant : bool, optional
            A boolean value, by default ``True``.

        Returns
        -------
        x : numpy ndarray
            float array whose shape is ``(sammple_length,c_degree)``.
        y : numpy ndarray
            1 dimensional float array whose size is ``sammple_length``.
        """
        if x is not None:
            _check.float_vecs(x,'x',DataFormatError)
            x = x.reshape([-1,self.c_degree])
            sample_size = x.shape[0]
        elif sample_size is not None:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            x = self.rng.multivariate_normal(np.zeros(self.c_degree),np.eye(self.c_degree), size=sample_size)
            if constant:
                x[:,-1] = 1.0
        else:
            raise(DataFormatError("Either of the sample_size and the x must be given as an input."))
        
        y = np.empty(sample_size)
        for i in range(sample_size):
            y[i] = self.rng.normal(loc = x[i] @ self.theta_vec, scale = 1.0 / np.sqrt(self.tau))

        return x, y

    def save_sample(self,filename,sample_size=None,x=None,constant=True):
        """Save the generated sample as NumPy ``.npz`` format.

        If x is given, it will be used for explanatory variables as it is 
        (independent of the other options: sample_size and constant).

        If x is not given, it will be generated from i.i.d. standard normal distribution.
        The size of the generated sample is defined by sample_size.
        If constant is True, the last element of the generated explanatory variables will be overwritten by 1.0.

        The generated sample is saved as a NpzFile with keyword: \"x\", \"y\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        x : numpy ndarray, optional
            float array whose shape is ``(sammple_length,c_degree)``, by default ``None``.
        sample_size : int, optional
            A positive integer, by default ``None``.
        constant : bool, optional
            A boolean value, by default ``True``.
        
        See Also
        --------
        numpy.savez_compressed
        """
        x, y = self.gen_sample(sample_size,x,constant)
        np.savez_compressed(filename, x=x, y=y)

    def visualize_model(self,sample_size=100,constant=True):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 50
        constant : bool, optional

        Examples
        --------
        >>> import numpy as np
        >>> from bayesml import linearregression
        >>> model = linearregression.GenModel(c_degree=2,theta_vec=np.array([2,1]))
        >>> model.visualize_model()

        .. image:: ./images/linearregression_example.png
        """
        if self.c_degree == 2 and constant==True:
            print(f"theta_vec:\n{self.theta_vec}")
            print(f"tau:\n{self.tau}")
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_y = self.gen_sample(sample_size=sample_size,constant=True)
            fig, ax = plt.subplots()
            ax.scatter(sample_x[:,0],sample_y)

            x = np.linspace(sample_x[:,0].min()-(sample_x[:,0].max()-sample_x[:,0].min())*0.25,
                            sample_x[:,0].max()+(sample_x[:,0].max()-sample_x[:,0].min())*0.25,
                            100)
            ax.plot(x, x*self.theta_vec[0] + self.theta_vec[1],label=f'y={self.theta_vec[0]:.2f}*x + {self.theta_vec[1]:.2f}',c='red')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        elif self.c_degree == 1 and constant==False:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_y = self.gen_sample(sample_size=sample_size,constant=False)
            fig, ax = plt.subplots()
            ax.scatter(sample_x,sample_y)

            x = np.linspace(sample_x.min()-(sample_x.max()-sample_x.min())*0.25,
                            sample_x.max()+(sample_x.max()-sample_x.min())*0.25,
                            100)
            ax.plot(x, x*self.theta_vec,label=f'y={self.theta_vec[0]:.2f}*x',c='red')
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        else:
            raise(ParameterFormatError("This function supports only the following cases: c_degree = 2 and constant = True; c_degree = 1 and constant = False."))



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
            h0_mu_vec = None, 
            h0_lambda_mat = None, 
            h0_alpha = 1.0, 
            h0_beta = 1.0, 
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)

        # h0_params
        self.h0_mu_vec = np.zeros(self.c_degree)
        self.h0_lambda_mat = np.eye(self.c_degree)
        self.h0_alpha = 1.0
        self.h0_beta = 1.0

        # hn_params
        self.hn_mu_vec = np.zeros(self.c_degree)
        self.hn_lambda_mat = np.eye(self.c_degree)
        self.hn_alpha = 1.0
        self.hn_beta = 1.0

        # p_params
        self.p_m = 0.0
        self.p_lambda =  1.0
        self.p_nu = 2.0

        # sample size
        self._n = 0

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
            h0_beta=None
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
                self.c_degree,'self.c_degree',
                ParameterFormatError
            )
            self.h0_mu_vec[:] = h0_mu_vec

        if h0_lambda_mat is not None:
            _check.pos_def_sym_mat(h0_lambda_mat,'h0_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                h0_lambda_mat.shape[0],'h0_lambda_mat.shape[0] and h0_lambda_mat.shape[1]',
                self.c_degree,'self.c_degree',
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
    
    def set_hn_params(
            self,
            hn_mu_vec=None,
            hn_lambda_mat=None,
            hn_alpha=None,
            hn_beta=None
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
        self._n = 0
        if hn_mu_vec is not None:
            _check.float_vec(hn_mu_vec,'hn_mu_vec',ParameterFormatError)
            _check.shape_consistency(
                hn_mu_vec.shape[0],'hn_mu_vec.shape[0]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
            )
            self.hn_mu_vec[:] = hn_mu_vec

        if hn_lambda_mat is not None:
            _check.pos_def_sym_mat(hn_lambda_mat,'hn_lambda_mat',ParameterFormatError)
            _check.shape_consistency(
                hn_lambda_mat.shape[0],'hn_lambda_mat.shape[0] and hn_lambda_mat.shape[1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
            )
            self.hn_lambda_mat[:] = hn_lambda_mat

        if hn_alpha is not None:
            self.hn_alpha = _check.pos_float(hn_alpha,'hn_alpha',ParameterFormatError)
        if hn_beta is not None:
            self.hn_beta = _check.pos_float(hn_beta,'hn_beta',ParameterFormatError)

        self.calc_pred_dist(np.zeros(self.c_degree))
        return self

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
    
    def _check_sample_x(self,x):
        _check.float_vecs(x,'x',DataFormatError)
        if x.shape[-1] != self.c_degree:
            raise(DataFormatError(f"x.shape[-1] must be c_degree:{self.c_degree}"))

    def _check_sample_y(self,y):
        _check.floats(y,'y',DataFormatError)

    def _check_sample(self,x,y):
        self._check_sample_x(x)
        self._check_sample_y(y)
        if type(y) is np.ndarray:
            if x.shape[:-1] != y.shape: 
                raise(DataFormatError(f"x.shape[:-1] and y.shape must be same."))
        elif x.shape[:-1] != ():
            raise(DataFormatError(f"If y is a scaler, x.shape[:-1] must be the empty tuple ()."))
        return x.reshape(-1,self.c_degree), np.ravel(y)

    def update_posterior(self, x, y):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        y : numpy ndarray
            float array.
        """
        x,y = self._check_sample(x,y)            
        n = x.shape[0]
        
        hn1_Lambda = np.array(self.hn_lambda_mat)
        hn1_mu = np.array(self.hn_mu_vec)
        self.hn_lambda_mat +=  x.T @ x
        self.hn_mu_vec[:] = np.linalg.solve(self.hn_lambda_mat, x.T @ y + hn1_Lambda @ hn1_mu)
        self.hn_alpha += n/2.0
        self.hn_beta += (-self.hn_mu_vec @ self.hn_lambda_mat @ self.hn_mu_vec
                         + y @ y + hn1_mu @ hn1_Lambda @ hn1_mu) /2.0
        self._n += n
        return self

    def _update_posterior(self, x, y):
        """Update opsterior without input check."""
        n = x.shape[0]
        hn1_Lambda = np.array(self.hn_lambda_mat)
        hn1_mu = np.array(self.hn_mu_vec)
        self.hn_lambda_mat +=  x.T @ x
        self.hn_mu_vec[:] = np.linalg.solve(self.hn_lambda_mat, x.T @ y + hn1_Lambda @ hn1_mu)
        self.hn_alpha += n/2.0
        self.hn_beta += (-self.hn_mu_vec @ self.hn_lambda_mat @ self.hn_mu_vec
                         + y @ y + hn1_mu @ hn1_Lambda @ hn1_mu) /2.0
        self._n += n
        return self

    def estimate_params(self,loss="squared",dict_out=False):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Note that the criterion is applied to estimating ``theta_vec`` and ``tau`` independently.
        Therefore, a tuple of the student's t-distribution and the gamma distribution will be returned when loss=\"KL\"

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        dict_out : bool, optional
            If ``True``, output will be a dict, by default ``False``.

        Returns
        -------
        estimates : tuple of {numpy ndarray, float, None, or rv_frozen}
            * ``theta_vec`` : the estimate for w
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
            if dict_out:
                return {'theta_vec':self.hn_mu_vec,'tau':self.hn_alpha/self.hn_beta}
            else:
                return  self.hn_mu_vec, self.hn_alpha/self.hn_beta
        elif loss == "0-1":
            if self.hn_alpha >= 1.0:
                if dict_out:
                    return {'theta_vec':self.hn_mu_vec,'tau':(self.hn_alpha - 1.0) / self.hn_beta}
                else:
                    return self.hn_mu_vec, (self.hn_alpha - 1.0) / self.hn_beta
            else:
                if dict_out:
                    return {'theta_vec':self.hn_mu_vec,'tau':0.0}
                else:
                    return self.hn_mu_vec, 0.0
        elif loss == "abs":
            if dict_out:
                return {'theta_vec':self.hn_mu_vec,'tau':ss_gamma.median(a=self.hn_alpha,scale=1.0/self.hn_beta)}
            else:
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
        >>> from bayesml import linearregression
        >>> gen_model = linearregression.GenModel(c_degree=2,theta_vec=np.array([1,1]),tau=1.0)
        >>> x,y = gen_model.gen_sample(sample_size=50)
        >>> learn_model = linearregression.LearnModel()
        >>> learn_model.update_posterior(x,y)
        >>> learn_model.visualize_posterior()
        
        .. image:: ./images/linearregression_posterior.png
        """
        theta_vec_pdf, tau_pdf = self.estimate_params(loss="KL")
        if self.c_degree == 1:
            hn_lambda_inv = np.linalg.inv(self.hn_lambda_mat)
            fig, axes = plt.subplots(1,2)

            # for mu_vec
            x = np.linspace(self.hn_mu_vec[0]-4.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_inv[0,0]),
                            self.hn_mu_vec[0]+4.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_inv[0,0]),
                            100)
            axes[0].plot(x,theta_vec_pdf.pdf(x))
            axes[0].set_xlabel("theta_vec")
            axes[0].set_ylabel("Density")

            #for tau
            x = np.linspace(max(1.0e-8,self.hn_alpha/self.hn_beta-4.0*np.sqrt(self.hn_alpha)/self.hn_beta),
                            self.hn_alpha/self.hn_beta+4.0*np.sqrt(self.hn_alpha)/self.hn_beta,
                            100)
            axes[1].plot(x,tau_pdf.pdf(x))
            axes[1].set_xlabel("tau")
            axes[1].set_ylabel("posterior probability")

            fig.tight_layout()
            plt.show()
        elif self.c_degree == 2:
            hn_lambda_inv = np.linalg.inv(self.hn_lambda_mat)
            fig, axes = plt.subplots(1,2)
            
            #for theta
            x = np.linspace(self.hn_mu_vec[0]-3.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_inv[0,0]),
                            self.hn_mu_vec[0]+3.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_inv[0,0]),
                            100)
            y = np.linspace(self.hn_mu_vec[1]-3.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_inv[1,1]),
                            self.hn_mu_vec[1]+3.0*np.sqrt(self.hn_beta / self.hn_alpha * hn_lambda_inv[1,1]),
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
        
        else:
            raise(ParameterFormatError("if self.c_degree > 2, it is impossible to visualize posterior by this function."))

    
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
    
    def calc_pred_dist(self, x):
        """Calculate the parameters of the predictive distribution.

        Parameters
        ----------
        x : numpy ndarray
            1 dimensional float array whose size is ``self.c_degree``
        """
        _check.float_vec(x,'x',DataFormatError)
        if x.shape != (self.c_degree,):
            raise(DataFormatError("x must be a 1 dimensional float array whose size coincide with ``self.c_degree``"))
        self.p_m = x @ self.hn_mu_vec
        self.p_lambda = self.hn_alpha / self.hn_beta / (1.0 + x @ np.linalg.solve(self.hn_lambda_mat,x))
        self.p_nu = 2.0 * self.hn_alpha
        return self

    def _calc_pred_dist(self, x):
        """Calculate predictive distribution without check."""
        self.p_m = x @ self.hn_mu_vec
        self.p_lambda = self.hn_alpha / self.hn_beta / (1.0 + x @ np.linalg.solve(self.hn_lambda_mat,x))
        self.p_nu = 2.0 * self.hn_alpha
        return self

    def _calc_pred_density(self,y):
        return ss_t.pdf(y,loc=self.p_m, scale=1.0/np.sqrt(self.p_lambda), df=self.p_nu)
    
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

    def pred_and_update(self,x,y,loss="squared"):
        """Predict a new data and update the posterior sequentially.

        Parameters
        ----------
        x : numpy ndarray
            1 dimensional float array whose size is ``self.c_degree``.
        y : float

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
        self.calc_pred_dist(x)
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x,y)
        return prediction

    def calc_log_marginal_likelihood(self):
        """Calculate log marginal likelihood

        Returns
        -------
        log_marginal_likelihood : float
            The log marginal likelihood.
        """
        return (
            self.h0_alpha * np.log(self.h0_beta)
            - self.hn_alpha * np.log(self.hn_beta)
            + gammaln(self.hn_alpha)
            - gammaln(self.h0_alpha)
            + 0.5 * (
                np.linalg.slogdet(self.h0_lambda_mat)[1]
                - np.linalg.slogdet(self.hn_lambda_mat)[1]
                - self._n * np.log(2*np.pi)
            )
        )
    
    def calc_pred_var(self):
        """Calculate the variance of the predictive distribution.
        
        Returns
        -------
        var : float
            The variance of the predictive distribution.
        """
        if self.p_nu > 2:
            return self.p_nu / self.p_lambda / (self.p_nu-2)
        else:
            warnings.warn("Variance of the predictive distribution cannot defined for the current p_nu.",ResultWarning)
            return None