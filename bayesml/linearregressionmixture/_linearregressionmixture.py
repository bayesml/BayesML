# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
import warnings
import numpy as np
from scipy.stats import t as ss_t
from scipy.stats import multivariate_t as ss_multivariate_t
from scipy.stats import dirichlet as ss_dirichlet
from scipy.stats import gamma as ss_gamma
from scipy.special import gammaln, digamma, xlogy, logsumexp
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from matplotlib.cm import tab20

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

_TAB_COLOR_LIST = [
    'tab:blue',
    'tab:orange',
    'tab:green',
    'tab:red',
    'tab:purple',
    'tab:brown',
    'tab:pink',
    'tab:gray',
    'tab:olive',
    'tab:cyan',
]

_LOG_2PI = np.log(2.0*np.pi)

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    c_num_classes : int
        A positive integer
    c_degree : int
        A positive integer
    pi_vec : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]`, 
        by default [1/c_num_classes, 1/c_num_classes, ... , 1/c_num_classes]
        Sum of its elements must be 1.0.
    theta_vecs : numpy.ndarray, optional
        Vectors of real numbers, by default zero vectors.
    taus : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    h_gamma_vec : float or numpy.ndarray, optional
        A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
        If a single real number is input, it will be broadcasted.
    h_mu_vecs : numpy.ndarray, optional
        Vectors of real numbers, by default zero vectors
    h_lambda_mats : numpy.ndarray, optional
        Positive definite symetric matrices, 
        by default the identity matrices.
        If a single matrix is input, it will be broadcasted.
    h_alphas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    h_betas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(
            self,
            c_num_classes,
            c_degree,
            *,
            pi_vec=None,
            theta_vecs=None,
            taus=None,
            h_gamma_vec=None,
            h_mu_vecs=None,
            h_lambda_mats=None,
            h_alphas=None,
            h_betas=None,
            seed=None
            ):
        # constants
        self.c_num_classes = _check.pos_int(c_num_classes,'c_num_classes',ParameterFormatError)
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.pi_vec = np.ones(self.c_num_classes) / self.c_num_classes
        self.theta_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.taus = np.ones(self.c_num_classes)

        # h_params
        self.h_gamma_vec = np.ones(self.c_num_classes) / 2.0
        self.h_mu_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.h_lambda_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])
        self.h_alphas = np.ones(self.c_num_classes)
        self.h_betas = np.ones(self.c_num_classes)

        self.set_params(pi_vec,theta_vecs,taus)
        self.set_h_params(h_gamma_vec,h_mu_vecs,h_lambda_mats,h_alphas,h_betas)

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_num_classes"`` : the value of ``self.c_num_classes``
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {"c_num_classes":self.c_num_classes, "c_degree":self.c_degree}

    def set_params(
            self,
            pi_vec=None,
            theta_vecs=None,
            taus=None
            ):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        pi_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]`, 
            by default [1/c_num_classes, 1/c_num_classes, ... , 1/c_num_classes]
            Sum of its elements must be 1.0.
        theta_vecs : numpy.ndarray, optional
            Vectors of real numbers, by default zero vectors.
        taus : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        """
        if pi_vec is not None:
            _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError)
            _check.shape_consistency(
                pi_vec.shape[-1],'pi_vec.shape[-1]',
                self.c_num_classes,'self.c_num_classes',
                ParameterFormatError
                )
            self.pi_vec[:] = pi_vec

        if theta_vecs is not None:
            _check.float_vecs(theta_vecs,'theta_vecs',ParameterFormatError)
            _check.shape_consistency(
                theta_vecs.shape[-1],'theta_vecs.shape[-1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.theta_vecs[:] = theta_vecs

        if taus is not None:
            _check.pos_floats(taus,'taus',ParameterFormatError)
            self.taus[:] = taus

    def set_h_params(
            self,
            h_gamma_vec=None,
            h_mu_vecs=None,
            h_lambda_mats=None,
            h_alphas=None,
            h_betas=None,
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_gamma_vec : float or numpy.ndarray, optional
            A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
            If a single real number is input, it will be broadcasted.
        h_mu_vecs : numpy.ndarray, optional
            Vectors of real numbers, by default zero vectors
        h_lambda_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices.
            If a single matrix is input, it will be broadcasted.
        h_alphas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        h_betas : float or numpy.ndarray, optional
            Positive real numbers, 
            by default [1.0, 1.0, ... , 1.0]
            If a single real number is input, it will be broadcasted.
        """
        if h_gamma_vec is not None:
            _check.pos_floats(h_gamma_vec,'h_gamma_vec',ParameterFormatError)
            self.h_gamma_vec[:] = h_gamma_vec

        if h_mu_vecs is not None:
            _check.float_vecs(h_mu_vecs,'h_mu_vecs',ParameterFormatError)
            _check.shape_consistency(
                h_mu_vecs.shape[-1],'h_mu_vecs.shape[-1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h_mu_vecs[:] = h_mu_vecs

        if h_lambda_mats is not None:
            _check.pos_def_sym_mats(h_lambda_mats,'h_lambda_mats',ParameterFormatError)
            _check.shape_consistency(
                h_lambda_mats.shape[-1],'h_lambda_mats.shape[-1] and h_lambda_mats.shape[-2]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h_lambda_mats[:] = h_lambda_mats

        if h_alphas is not None:
            _check.pos_floats(h_alphas,'h_alphas',ParameterFormatError)
            self.h_alphas[:] = h_alphas

        if h_betas is not None:
            _check.pos_floats(h_betas,'h_betas',ParameterFormatError)
            self.h_betas[:] = h_betas
        
    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : {str: numpy.ndarray}
            * ``"pi_vec"`` : The value of ``self.pi_vec``
            * ``"theta_vecs"`` : The value of ``self.theta_vecs``
            * ``"taus"`` : The value of ``self.taus``
        """
        return {'pi_vec':self.pi_vec,
                'theta_vecs':self.theta_vecs,
                'taus':self.taus}
        
    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.
        
        Returns
        -------
        h_params : {str:float, np.ndarray}
            * ``"h_gamma_vec"`` : The value of ``self.h_gamma_vec``
            * ``"h_mu_vecs"`` : The value of ``self.h_mu_vecs``
            * ``"h_lambda_mats"`` : The value of ``self.h_lambda_mats``
            * ``"h_alphas"`` : The value of ``self.h_alphas``
            * ``"h_betas"`` : The value of ``self.h_betas``
        """
        return {'h_gamma_vec':self.h_gamma_vec,
                'h_mu_vecs':self.h_mu_vecs,
                'h_lambda_mats':self.h_lambda_mats,
                'h_alphas':self.h_alphas,
                'h_betas':self.h_betas}
    
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.pi_vec``, ``self.theta_vecs`` and ``self.lambda_mats``.
        """
        self.pi_vec[:] = self.rng.dirichlet(self.h_gamma_vec)
        for k in range(self.c_num_classes):
            self.taus[k] =  self.rng.gamma(
                shape=self.h_alphas[k],
                scale=1.0/self.h_betas[k]
            )
            self.theta_vecs[k] = self.rng.multivariate_normal(
                mean=self.h_mu_vecs[k],
                cov=np.linalg.inv(self.taus[k]*self.h_lambda_mats[k])
            )
        return self

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
            float array whose shape is ``(sample_size,c_degree)``, by default ``None``.
        constant : bool, optional
            A boolean value, by default ``True``.

        Returns
        -------
        x : numpy ndarray
            2-dimensional array whose shape is ``(sample_size,c_degree)`` and its elements are real numbers.
        z : numpy ndarray
            2-dimensional array whose shape is ``(sample_size,c_num_classes)`` whose rows are one-hot vectors.
        y : numpy ndarray
            1 dimensional float array whose size is ``sample_size``.
        """
        if x is not None:
            _check.float_vecs(x,'x',DataFormatError)
            _check.shape_consistency(
                x.shape[-1],"x.shape[-1]", 
                self.c_degree,"self.c_degree", 
                ParameterFormatError
                )
            x = x.reshape([-1,self.c_degree])
            sample_size = x.shape[0]
        elif sample_size is not None:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            x = self.rng.multivariate_normal(np.zeros(self.c_degree),np.eye(self.c_degree), size=sample_size)
            if constant:
                x[:,-1] = 1.0
        else:
            raise(DataFormatError("Either of the sample_size and the x must be given as an input."))

        z = np.zeros([sample_size,self.c_num_classes],dtype=int)
        y = np.empty(sample_size)

        for i in range(sample_size):
            k = self.rng.choice(self.c_num_classes,p=self.pi_vec)
            z[i,k] = 1
            y[i] = self.rng.normal(loc = x[i] @ self.theta_vecs[k], scale = 1.0 / np.sqrt(self.taus[k]))
        return x,z,y
    
    def save_sample(self,filename,sample_size=None,x=None,constant=True):
        """Save the generated sample as NumPy ``.npz`` format.

        If x is given, it will be used for explanatory variables as it is 
        (independent of the other options: sample_size and constant).

        If x is not given, it will be generated from i.i.d. standard normal distribution.
        The size of the generated sample is defined by sample_size.
        If constant is True, the last element of the generated explanatory variables will be overwritten by 1.0.
        
        The generated sample is saved as a NpzFile with keyword: \"x\", \"z\", \"y\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int, optional
            A positive integer, by default ``None``.
        x : numpy ndarray, optional
            float array whose shape is ``(sample_size,c_degree)``, by default ``None``.
        constant : bool, optional
            A boolean value, by default ``True``.
        
        See Also
        --------
        numpy.savez_compressed
        """
        x,z,y=self.gen_sample(sample_size,x,constant)
        np.savez_compressed(filename,x=x,z=z,y=y)
    
    def visualize_model(self,sample_size=100,constant=True):
        """Visualize the stochastic data generative model and generated samples.
        
        If x is given, it will be used for explanatory variables as it is 
        (independent of the other options: sample_size and constant).

        If x is not given, it will be generated from i.i.d. standard normal distribution.
        The size of the generated sample is defined by sample_size.
        If constant is True, the last element of the generated explanatory variables will be overwritten by 1.0.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 100
        constant : bool, optional
        
        Examples
        --------
        >>> from bayesml import linearregressionmixture
        >>> import numpy as np
        >>> model = linearregressionmixture.GenModel(
        >>>     c_num_classes=2,
        >>>     c_degree=2,
        >>>     theta_vecs=np.array([[1,3],
        >>>                          [-1,-3]]),
        >>> )
        >>> model.visualize_model()

        pi_vec:
        [0.5 0.5]
        theta_vecs:
        [[ 1.  3.]
         [-1. -3.]]
        taus:
        [1. 1.]
        
        .. image:: ./images/linearregressionmixture_example.png
        """
        print(f"pi_vec:\n{self.pi_vec}")
        print(f"theta_vecs:\n{self.theta_vecs}")
        print(f"taus:\n{self.taus}")
        if self.c_degree == 2 and constant==True:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_z, sample_y = self.gen_sample(sample_size=sample_size,constant=True)
            fig, ax = plt.subplots()

            x = np.linspace(sample_x[:,0].min()-(sample_x[:,0].max()-sample_x[:,0].min())*0.25,
                            sample_x[:,0].max()+(sample_x[:,0].max()-sample_x[:,0].min())*0.25,
                            100)
            for k in range(self.c_num_classes):
                ax.scatter(
                    sample_x[:,0][sample_z[:,k]==1],
                    sample_y[sample_z[:,k]==1],
                    color=_TAB_COLOR_LIST[k%10],
                )
                ax.plot(
                    x,
                    x*self.theta_vecs[k][0] + self.theta_vecs[k][1],
                    label=f'y={self.theta_vecs[k][0]:.2f}*x + {self.theta_vecs[k][1]:.2f}',
                    color=_TAB_COLOR_LIST[k%10],
                )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        elif self.c_degree == 1 and constant==False:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_z, sample_y = self.gen_sample(sample_size=sample_size,constant=False)
            fig, ax = plt.subplots()
            ax.scatter(sample_x,sample_y)

            x = np.linspace(sample_x.min()-(sample_x.max()-sample_x.min())*0.25,
                            sample_x.max()+(sample_x.max()-sample_x.min())*0.25,
                            100)
            for k in range(self.c_num_classes):
                ax.scatter(
                    sample_x[sample_z[:,k]==1],
                    sample_y[sample_z[:,k]==1],
                    color=_TAB_COLOR_LIST[k%10],
                )
                ax.plot(
                    x,
                    x*self.theta_vecs[k],
                    label=f'y={self.theta_vecs[k][0]:.2f}*x',
                    color=_TAB_COLOR_LIST[k%10],
                )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        else:
            raise(ParameterFormatError(
                "This function supports only the following cases: "
                +"c_degree = 2 and constant = True; c_degree = 1 "
                +"and constant = False."
                )
            )

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_num_classes : int
        a positive integer
    c_degree : int
        a positive integer
    h0_gamma_vec : float or numpy.ndarray, optional
        A vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
        If a single real number is input, it will be broadcasted.
    h0_mu_vecs : numpy.ndarray, optional
        Vectors of real numbers, by default zero vectors
    h0_lambda_mats : numpy.ndarray, optional
        Positive definite symetric matrices, 
        by default the identity matrices.
        If a single matrix is input, it will be broadcasted.
    h0_alphas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    h0_betas : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None

    Attributes
    ----------
    hn_gamma_vec : float or numpy.ndarray
        A vector of positive real numbers.
        If a single real number is input, it will be broadcasted.
    hn_mu_vecs : numpy.ndarray
        Vectors of real numbers.
    hn_lambda_mats : numpy.ndarray
        Positive definite symetric matrices. 
    hn_lambda_mats_inv : numpy.ndarray
        Positive definite symetric matrices. 
    hn_alphas : float or numpy.ndarray
        Positive real numbers. 
    hn_betas : float or numpy.ndarray
        Positive real numbers. 
    r_vecs : numpy.ndarray
        vectors of real numbers. The sum of its elenemts is 1.
    ns : numpy.ndarray
        positive real numbers
    vl : float
        real number
    p_pi_vecs : numpy.ndarray
        A vector of real numbers in :math:`[0, 1]`. 
        Sum of its elements must be 1.0.
    p_ms : numpy.ndarray
        Real numbers
    p_lambdas : numpy.ndarray
        Positive real numbers
    p_nus : numpy.ndarray
        Positive real numbers
    """
    def __init__(
            self,
            c_num_classes,
            c_degree,
            *,
            h0_gamma_vec=None,
            h0_mu_vecs=None,
            h0_lambda_mats=None,
            h0_alphas=None,
            h0_betas=None,
            seed = None
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.c_num_classes = _check.pos_int(c_num_classes,'c_num_classes',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h0_params
        self.h0_gamma_vec = np.ones(self.c_num_classes) / 2.0
        self.h0_mu_vecs = np.zeros([self.c_num_classes,self.c_degree])
        self.h0_lambda_mats = np.tile(np.identity(self.c_degree),[self.c_num_classes,1,1])
        self.h0_alphas = np.ones(self.c_num_classes)
        self.h0_betas = np.ones(self.c_num_classes)

        self._ln_c_h0_gamma = 0.0
        self._ln_det_h0_lambda_mats = np.empty((self.c_num_classes))
        self._h0_lambda_mu_vecs = np.empty((self.c_num_classes,self.c_degree))
        self._gammaln_h0_alphas = np.empty(self.c_num_classes)
        self._ln_h0_betas = np.empty(self.c_num_classes)

        # hn_params
        self.hn_gamma_vec = np.empty(self.c_num_classes)
        self.hn_mu_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.hn_lambda_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.hn_lambda_mats_inv = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.hn_alphas = np.empty(self.c_num_classes)
        self.hn_betas = np.empty(self.c_num_classes)

        self._ln_rho = None
        self.r_vecs = None
        self._e_ln_pi_vec = np.empty(self.c_num_classes)
        self._e_taus = np.empty(self.c_num_classes)
        self._e_ln_taus = np.empty(self.c_num_classes)

        # statistics
        self.x_r_x_mats = np.empty([self.c_num_classes,self.c_degree,self.c_degree])
        self.x_r_y_vecs = np.empty([self.c_num_classes,self.c_degree])
        self.y_r_ys = np.empty([self.c_num_classes])
        self.ns = np.empty(self.c_num_classes)

        # variational lower bound
        self.vl = 0.0
        self._vl_p_y = 0.0
        self._vl_p_z = 0.0
        self._vl_p_pi = 0.0
        self._vl_p_theta_tau = 0.0
        self._vl_q_z = 0.0
        self._vl_q_pi = 0.0
        self._vl_q_theta_tau = 0.0

        # p_params
        self.p_pi_vecs = np.empty(self.c_num_classes)
        self.p_ms = np.empty(self.c_num_classes)        
        self.p_lambdas = np.empty(self.c_num_classes)
        self.p_nus = np.empty(self.c_num_classes)
        
        self.set_h0_params(
            h0_gamma_vec,
            h0_mu_vecs,
            h0_lambda_mats,
            h0_alphas,
            h0_betas,
        )

    def get_constants(self):
        """Get constants of LearnModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_num_classes"`` : the value of ``self.c_num_classes``
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {"c_num_classes":self.c_num_classes, "c_degree":self.c_degree}

    def set_h0_params(
            self,
            h0_gamma_vec=None,
            h0_mu_vecs=None,
            h0_lambda_mats=None,
            h0_alphas=None,
            h0_betas=None,
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_gamma_vec : float or numpy.ndarray, optional
            A vector of positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        h0_mu_vecs : numpy.ndarray, optional
            Vectors of real numbers, by default None.
        h0_lambda_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices.
            If a single matrix is input, it will be broadcasted.
        h0_alphas : float or numpy.ndarray, optional
            Positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        h0_betas : float or numpy.ndarray, optional
            Positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        """
        if h0_gamma_vec is not None:
            _check.pos_floats(h0_gamma_vec,'h0_gamma_vec',ParameterFormatError)
            self.h0_gamma_vec[:] = h0_gamma_vec

        if h0_mu_vecs is not None:
            _check.float_vecs(h0_mu_vecs,'h0_mu_vecs',ParameterFormatError)
            _check.shape_consistency(
                h0_mu_vecs.shape[-1],'h0_mu_vecs.shape[-1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h0_mu_vecs[:] = h0_mu_vecs

        if h0_lambda_mats is not None:
            _check.pos_def_sym_mats(h0_lambda_mats,'h0_lambda_mats',ParameterFormatError)
            _check.shape_consistency(
                h0_lambda_mats.shape[-1],'h0_lambda_mats.shape[-1] and h0_lambda_mats.shape[-2]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.h0_lambda_mats[:] = h0_lambda_mats

        if h0_alphas is not None:
            _check.pos_floats(h0_alphas,'h0_alphas',ParameterFormatError)
            self.h0_alphas[:] = h0_alphas

        if h0_betas is not None:
            _check.pos_floats(h0_betas,'h0_betas',ParameterFormatError)
            self.h0_betas[:] = h0_betas

        self._calc_prior_features()
        self.reset_hn_params()

    def get_h0_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h0_params : dict of {str: numpy.ndarray}
            * ``"h0_gamma_vec"`` : the value of ``self.h0_gamma_vec``
            * ``"h0_mu_vecs"`` : the value of ``self.h0_mu_vecs``
            * ``"h0_lambda_mats"`` : the value of ``self.h0_lambda_mats``
            * ``"h0_alphas"`` : the value of ``self.h0_alphas``
            * ``"h0_betas"`` : the value of ``self.h0_betas``
        """
        return {'h0_gamma_vec':self.h0_gamma_vec,
                'h0_mu_vecs':self.h0_mu_vecs,
                'h0_lambda_mats':self.h0_lambda_mats,
                'h0_alphas':self.h0_alphas,
                'h0_betas':self.h0_betas}
    
    def set_hn_params(
            self,
            hn_gamma_vec=None,
            hn_mu_vecs=None,
            hn_lambda_mats=None,
            hn_alphas=None,
            hn_betas=None,
            ):
        """Set the hyperparameter of the posterior distribution.

        Parameters
        ----------
        hn_gamma_vec : float or numpy.ndarray, optional
            A vector of positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        hn_mu_vecs : numpy.ndarray, optional
            Vectors of real numbers, by default None.
        hn_lambda_mats : numpy.ndarray, optional
            Positive definite symetric matrices, 
            by default the identity matrices.
            If a single matrix is input, it will be broadcasted.
        hn_alphas : float or numpy.ndarray, optional
            Positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        hn_betas : float or numpy.ndarray, optional
            Positive real numbers, by default None.
            If a single real number is input, it will be broadcasted.
        """
        if hn_gamma_vec is not None:
            _check.pos_floats(hn_gamma_vec,'hn_gamma_vec',ParameterFormatError)
            self.hn_gamma_vec[:] = hn_gamma_vec

        if hn_mu_vecs is not None:
            _check.float_vecs(hn_mu_vecs,'hn_mu_vecs',ParameterFormatError)
            _check.shape_consistency(
                hn_mu_vecs.shape[-1],'hn_mu_vecs.shape[-1]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.hn_mu_vecs[:] = hn_mu_vecs

        if hn_lambda_mats is not None:
            _check.pos_def_sym_mats(hn_lambda_mats,'hn_lambda_mats',ParameterFormatError)
            _check.shape_consistency(
                hn_lambda_mats.shape[-1],'hn_lambda_mats.shape[-1] and hn_lambda_mats.shape[-2]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.hn_lambda_mats[:] = hn_lambda_mats
            self.hn_lambda_mats_inv[:] = np.linalg.inv(self.hn_lambda_mats)

        if hn_alphas is not None:
            _check.pos_floats(hn_alphas,'hn_alphas',ParameterFormatError)
            self.hn_alphas[:] = hn_alphas

        if hn_betas is not None:
            _check.pos_floats(hn_betas,'hn_betas',ParameterFormatError)
            self.hn_betas[:] = hn_betas

        self._calc_q_theta_tau_features()
        self._calc_q_pi_features()
        self.calc_pred_dist(np.zeros(self.c_degree))

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: numpy.ndarray}
            * ``"hn_gamma_vec"`` : the value of ``self.hn_gamma_vec``
            * ``"hn_mu_vecs"`` : the value of ``self.hn_mu_vecs``
            * ``"hn_lambda_mats"`` : the value of ``self.hn_lambda_mats``
            * ``"hn_alphas"`` : the value of ``self.hn_alphas``
            * ``"hn_betas"`` : the value of ``self.hn_betas``
        """
        return {'hn_gamma_vec':self.hn_gamma_vec,
                'hn_mu_vecs':self.hn_mu_vecs,
                'hn_lambda_mats':self.hn_lambda_mats,
                'hn_alphas':self.hn_alphas,
                'hn_betas':self.hn_betas}

    def _check_sample_x(self,x):
        _check.float_vecs(x,'x',DataFormatError)
        if x.shape[-1] != self.c_degree:
            raise(DataFormatError(f"x.shape[-1] must be c_degree:{self.c_degree}"))
        return x.reshape(-1,self.c_degree)
    
    def _check_sample_y(self,y):
        return _check.floats(y,'y',DataFormatError)

    def _check_sample(self,x,y):
        self._check_sample_x(x)
        self._check_sample_y(y)
        if type(y) is np.ndarray:
            if x.shape[:-1] != y.shape: 
                raise(DataFormatError(f"x.shape[:-1] and y.shape must be same."))
        elif x.shape[:-1] != ():
            raise(DataFormatError(f"If y is a scaler, x.shape[:-1] must be the empty tuple ()."))
        return x.reshape(-1,self.c_degree), np.ravel(y)

    def update_posterior(
            self,
            x,
            y,
            max_itr=100,
            num_init=10,
            tolerance=1.0E-8,
            init_type='random_responsibility',
            ):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        y : numpy ndarray
            float array.
        max_itr : int, optional
            maximum number of iterations, by default 100
        num_init : int, optional
            number of initializations, by default 10
        tolerance : float, optional
            convergence criterion of variational lower bound, by default 1.0E-8
        init_type : str, optional
            * ``'random_responsibility'``: randomly assign responsibility to ``r_vecs``
            * ``'subsampling'``: for each latent class, extract a subsample whose size is ``int(np.sqrt(x.shape[0]))``.
              and use it to update q(theta_k,tau_k).
            Type of initialization, by default ``'random_responsibility'``
        """
        x,y = self._check_sample(x,y)
        self._ln_rho = np.empty([x.shape[0],self.c_num_classes])
        self.r_vecs = np.empty([x.shape[0],self.c_num_classes])

        tmp_vl = 0.0
        tmp_gamma_vec = np.array(self.hn_gamma_vec)
        tmp_mu_vecs = np.array(self.hn_mu_vecs)
        tmp_lambda_mats = np.array(self.hn_lambda_mats)
        tmp_alphas = np.array(self.hn_alphas)
        tmp_betas = np.array(self.hn_betas)

        convergence_flag = True
        for i in range(num_init):
            self.reset_hn_params()
            self._init_rho_r()
            if init_type == 'subsampling':
                self._init_subsampling(x,y)
                self._update_q_z(x,y)
            elif init_type == 'random_responsibility':
                self._init_random_responsibility(x,y)
            else:
                raise(ValueError(
                    f'init_type={init_type} is unsupported. '
                    + 'This function supports only '
                    + '"subsampling" and "random_responsibility"'))
            self._calc_vl(x,y)
            print(f'\r{i}. VL: {self.vl}',end='')
            for t in range(max_itr):
                vl_before = self.vl
                self._update_q_theta_tau()
                self._update_q_pi()
                self._update_q_z(x,y)
                self._calc_vl(x,y)
                print(f'\r{i}. VL: {self.vl} t={t} ',end='')
                if np.abs((self.vl-vl_before)/vl_before) < tolerance:
                    convergence_flag = False
                    print(f'(converged)',end='')
                    break
            if i==0 or self.vl > tmp_vl:
                print('*')
                tmp_vl = self.vl
                tmp_gamma_vec[:] = self.hn_gamma_vec
                tmp_mu_vecs[:] = self.hn_mu_vecs
                tmp_lambda_mats[:] = self.hn_lambda_mats
                tmp_alphas[:] = self.hn_alphas
                tmp_betas[:] = self.hn_betas
            else:
                print('')
        if convergence_flag:
            warnings.warn("Algorithm has not converged even once.",ResultWarning)
        
        self.hn_gamma_vec[:] = tmp_gamma_vec
        self.hn_mu_vecs[:] = tmp_mu_vecs
        self.hn_lambda_mats[:] = tmp_lambda_mats
        self.hn_lambda_mats_inv[:] = np.linalg.inv(self.hn_lambda_mats)
        self.hn_alphas[:] = tmp_alphas
        self.hn_betas[:] = tmp_betas
        self._calc_q_pi_features()
        self._calc_q_theta_tau_features()
        self._update_q_z(x,y)
        return self

    def estimate_params(self,loss="squared"):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Note that the criterion is applied to estimating 
        ``pi_vec``, ``theta_vecs`` and ``taus`` independently.
        Therefore, a tuple of the dirichlet distribution, 
        the student's t-distributions and 
        the wishart distributions will be returned when loss=\"KL\"

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        Estimates : a tuple of {numpy ndarray, float, None, or rv_frozen}
            * ``pi_vec_hat`` : the estimate for pi_vec
            * ``theta_vecs_hat`` : the estimate for theta_vecs
            * ``taus_hat`` : the estimate for taus
            The estimated values under the given loss function. 
            If it is not exist, `np.nan` will be returned.
            If the loss function is \"KL\", the posterior distribution itself 
            will be returned as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """

        if loss == "squared":
            return self.hn_gamma_vec/self.hn_gamma_vec.sum(), self.hn_mu_vecs, self._e_taus
        elif loss == "0-1":
            pi_vec_hat = np.empty(self.c_num_classes)
            if np.all(self.hn_gamma_vec > 1):
                pi_vec_hat[:] = (self.hn_gamma_vec - 1) / (np.sum(self.hn_gamma_vec) - self.c_degree)
            else:
                warnings.warn("MAP estimate of pi_vec doesn't exist for the current hn_gamma_vec.",ResultWarning)
                pi_vec_hat[:] = np.nan

            taus_hat = np.zeros(self.c_num_classes)
            indices = self.hn_alphas >= 1.0
            taus_hat[indices] = (self.hn_alphas[indices] - 1) / self.hn_betas[indices]
            return pi_vec_hat, self.hn_mu_vecs, taus_hat
        elif loss == "KL":
            theta_vec_pdfs = []
            tau_pdfs = []
            for k in range(self.c_num_classes):
                theta_vec_pdfs.append(ss_multivariate_t(loc=self.hn_mu_vecs[k],
                                        shape=np.linalg.inv(self.hn_alphas[k] / self.hn_betas[k] * self.hn_lambda_mats[k]),
                                        df=2.0*self.hn_alphas[k]))
                tau_pdfs.append(ss_gamma(a=self.hn_alphas[k],scale=1.0/self.hn_betas[k]))
            return (ss_dirichlet(self.hn_gamma_vec),
                    theta_vec_pdfs,
                    tau_pdfs)
        else:
            raise(CriteriaError(f"loss={loss} is unsupported. "
                                +"This function supports \"squared\", \"0-1\", and \"KL\"."))

    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> import numpy as np
        >>> from bayesml import linearregressionmixture
        >>> gen_model = linearregressionmixture.GenModel(
        >>>     c_num_classes=2,
        >>>     c_degree=2,
        >>>     theta_vecs=np.array([[1,3],[-1,-3]]),
        >>>     taus=np.array([0.5,1.0]),
        >>>     )
        >>> x,z,y = gen_model.gen_sample(100)
        >>> learn_model = linearregressionmixture.LearnModel(
        >>>     c_num_classes=2,
        >>>     c_degree=2,
        >>>     )
        >>> learn_model.update_posterior(x,y)
        >>> learn_model.visualize_posterior()
        hn_gamma_vec:
        [53.46589867 47.53410133]
        E[pi_vec]:
        [0.52936533 0.47063467]
        hn_mu_vecs:
        [[-1.12057057 -3.14175971]
        [ 1.15046197  2.72935847]]
        hn_lambda_mats:
        [[[ 73.28683786  -1.18874056]
        [ -1.18874056  53.96589867]]

        [[ 39.13313893 -10.37075427]
        [-10.37075427  48.03410133]]]
        hn_alphas:
        [27.48294934 24.51705066]
        hn_betas:
        [27.13542998 43.09024752]
        E[taus]:
        [1.01280685 0.56896983]

        .. image:: ./images/linearregressionmixture_posterior.png
        """
        print("hn_gamma_vec:")
        print(f"{self.hn_gamma_vec}")
        print("E[pi_vec]:")
        print(f"{self.hn_gamma_vec / self.hn_gamma_vec.sum()}")
        print("hn_mu_vecs:")
        print(f"{self.hn_mu_vecs}")
        print("hn_lambda_mats:")
        print(f"{self.hn_lambda_mats}")
        print("hn_alphas:")
        print(f"{self.hn_alphas}")
        print("hn_betas:")
        print(f"{self.hn_betas}")
        print("E[taus]:")
        print(f"{self.hn_alphas / self.hn_betas}")
        _, theta_vec_pdfs, tau_pdfs = self.estimate_params(loss="KL")
        if self.c_degree == 1:
            fig, axes = plt.subplots(1,2)
            axes[0].set_xlabel("theta_vecs")
            axes[0].set_ylabel("Density")
            axes[1].set_xlabel("taus")
            axes[1].set_ylabel("Density")
            for k in range(self.c_num_classes):
                # for theta_vecs
                x = np.linspace(self.hn_mu_vecs[k,0]-4.0*np.sqrt((self.hn_lambda_mats_inv[k] / self.hn_alphas[k] * self.hn_betas[k])[0,0]),
                                self.hn_mu_vecs[k,0]+4.0*np.sqrt((self.hn_lambda_mats_inv[k] / self.hn_alphas[k] * self.hn_betas[k])[0,0]),
                                100)
                axes[0].plot(x,theta_vec_pdfs[k].pdf(x))
                # for taus
                x = np.linspace(max(1.0e-8,self.hn_alphas[k]/self.hn_betas[k]-4.0*np.sqrt(self.hn_alphas[k]/self.hn_betas[k]**2)),
                                self.hn_alphas[k]/self.hn_betas[k]+4.0*np.sqrt(self.hn_alphas[k]/self.hn_betas[k]**2),
                                500)
                axes[1].plot(x,tau_pdfs[k].pdf(x))

            fig.tight_layout()
            plt.show()

        elif self.c_degree == 2:
            fig, axes = plt.subplots(1,2)
            axes[0].set_xlabel("theta_vec[0]")
            axes[0].set_ylabel("theta_vec[1]")
            axes[1].set_xlabel("taus")
            axes[1].set_ylabel("Density")
            for k in range(self.c_num_classes):
                # for theta_vecs
                x = np.linspace(self.hn_mu_vecs[k,0]-3.0*np.sqrt((self.hn_lambda_mats_inv[k] / self.hn_alphas[k] * self.hn_betas[k])[0,0]),
                                self.hn_mu_vecs[k,0]+3.0*np.sqrt((self.hn_lambda_mats_inv[k] / self.hn_alphas[k] * self.hn_betas[k])[0,0]),
                                100)
                y = np.linspace(self.hn_mu_vecs[k,1]-3.0*np.sqrt((self.hn_lambda_mats_inv[k] / self.hn_alphas[k] * self.hn_betas[k])[1,1]),
                                self.hn_mu_vecs[k,1]+3.0*np.sqrt((self.hn_lambda_mats_inv[k] / self.hn_alphas[k] * self.hn_betas[k])[1,1]),
                                100)
                xx, yy = np.meshgrid(x,y)
                grid = np.empty((100,100,2))
                grid[:,:,0] = xx
                grid[:,:,1] = yy
                axes[0].contour(xx,yy,theta_vec_pdfs[k].pdf(grid),cmap='Blues',alpha=self.hn_gamma_vec[k]/self.hn_gamma_vec.sum())
                axes[0].plot(self.hn_mu_vecs[k,0],self.hn_mu_vecs[k,1],marker="x",color='red')
                # for taus
                x = np.linspace(max(1.0e-8,self.hn_alphas[k]/self.hn_betas[k]-4.0*np.sqrt(self.hn_alphas[k]/self.hn_betas[k]**2)),
                                self.hn_alphas[k]/self.hn_betas[k]+4.0*np.sqrt(self.hn_alphas[k]/self.hn_betas[k]**2),
                                500)
                axes[1].plot(x,tau_pdfs[k].pdf(x))
            
            fig.tight_layout()
            plt.show()

        else:
            raise(ParameterFormatError("if c_degree > 2, it is impossible to visualize the model by this function."))
        
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy.ndarray}
            * ``"p_pi_vecs"`` : the value of ``self.p_pi_vecs``
            * ``"p_ms"`` : the value of ``self.p_ms``
            * ``"p_lambdas"`` : the value of ``self.p_lambdas``
            * ``"p_nus"`` : the value of ``self.p_nus``
        """
        return {'p_pi_vecs':self.p_pi_vecs,
                'p_ms':self.p_ms,
                'p_lambdas':self.p_lambdas,
                'p_nus':self.p_nus}

    def calc_pred_dist(self,x):
        """Calculate the parameters of the predictive distribution.

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        """
        x = self._check_sample_x(x)
        self.p_pi_vecs = np.ones((x.shape[0],self.c_num_classes)) * self.hn_gamma_vec / self.hn_gamma_vec.sum()
        self.p_ms = x @ self.hn_mu_vecs.T
        self.p_lambdas = self.hn_alphas / self.hn_betas / (1.0 + np.einsum('ij,kjl,il->ik', x, self.hn_lambda_mats_inv, x))
        self.p_nus = np.ones((x.shape[0],self.c_num_classes)) * 2.0 * self.hn_alphas
        return self

    def make_prediction(self,loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\" and \"0-1\".

        Returns
        -------
        predicted_value : numpy.ndarray
            The predicted value under the given loss function. 
            The size of the predicted values is the same as the sample size of x when you called calc_pred_dist(x).
        """
        if loss == "squared":
            return (self.p_pi_vecs * self.p_ms).sum(axis=1)
        elif loss == "0-1":
            val = self.p_pi_vecs * ss_t.pdf(
                    x=self.p_ms,
                    loc=self.p_ms,
                    scale=1.0/np.sqrt(self.p_lambdas),
                    df=self.p_nus)
            indices = np.argmax(val,axis=1)
            return self.p_ms[np.arange(self.p_ms.shape[0]),indices]
        else:
            raise(CriteriaError(f"loss={loss} is unsupported. "
                                +"This function supports \"squared\" and \"0-1\"."))

    def pred_and_update(
            self,
            x,
            y,
            loss="squared",
            max_itr=100,
            num_init=10,
            tolerance=1.0E-8,
            init_type='random_responsibility',
            ):
        """Update the hyperparameters of the posterior distribution using traning data.

        h0_params will be overwritten by current hn_params 
        before updating hn_params by x

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        y : numpy ndarray
            float array.
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\" and \"0-1\".
        max_itr : int, optional
            maximum number of iterations, by default 100
        num_init : int, optional
            number of initializations, by default 10
        tolerance : float, optional
            convergence criterion of variational lower bound, by default 1.0E-8
        init_type : str, optional
            * ``'random_responsibility'``: randomly assign responsibility to ``r_vecs``
            * ``'subsampling'``: for each latent class, extract a subsample whose size is ``int(np.sqrt(x.shape[0]))``.
              and use it to update q(theta_k,tau_k).
            Type of initialization, by default ``'random_responsibility'``
        
        Returns
        -------
        predicted_value : numpy.ndarray
            The predicted value under the given loss function. 
            The size of the predicted values is the same as the sample size of x when you called calc_pred_dist(x).
        """
        self.calc_pred_dist(x)
        prediction = self.make_prediction(loss=loss)
        self.overwrite_h0_params()
        self.update_posterior(
            x,
            y,
            max_itr=max_itr,
            num_init=num_init,
            tolerance=tolerance,
            init_type=init_type
            )
        return prediction

    def _calc_prior_features(self):
        self._ln_c_h0_gamma = gammaln(self.h0_gamma_vec.sum()) - gammaln(self.h0_gamma_vec).sum()
        self._ln_det_h0_lambda_mats[:] = np.linalg.slogdet(self.h0_lambda_mats)[1]
        self._h0_lambda_mu_vecs[:] = np.einsum('kij,kj->ki',self.h0_lambda_mats,self.h0_mu_vecs)
        self._gammaln_h0_alphas[:] = gammaln(self.h0_alphas)
        self._ln_h0_betas[:] = np.log(self.h0_betas)

    def _init_rho_r(self):
        self._ln_rho[:] = 0.0
        self.r_vecs[:] = 1/self.c_num_classes

    def _calc_q_theta_tau_features(self):
        self._e_taus[:] = self.hn_alphas / self.hn_betas
        self._e_ln_taus[:] = digamma(self.hn_alphas) - np.log(self.hn_betas)

    def _calc_q_pi_features(self):
        self._e_ln_pi_vec[:] = digamma(self.hn_gamma_vec) - digamma(self.hn_gamma_vec.sum())
    
    def _calc_statistics_with_r(self,x,y):
        self.ns[:] = self.r_vecs.sum(axis=0)
        for k in range(self.c_num_classes):
            self.x_r_x_mats[k] =  (self.r_vecs[:,k] * x.T) @ x
            self.x_r_y_vecs[k] = x.T @ (self.r_vecs[:,k] * y)
            self.y_r_ys[k] = y @ (self.r_vecs[:,k] * y)
    
    def _calc_vl(self,x,y):
        # E[ln p(y|x,z,theta,tau)]
        self._vl_p_y = (
            - x.shape[0] * _LOG_2PI
            + self.ns @ self._e_ln_taus
            - self._e_taus @ (
                self.y_r_ys
                - 2.0 * (self.hn_mu_vecs * self.x_r_y_vecs).sum(axis=1)
                + np.einsum('ki,kij,kj->k', self.hn_mu_vecs, self.x_r_x_mats, self.hn_mu_vecs)
            )
            - (self.hn_lambda_mats_inv * self.x_r_x_mats).sum()
        ) / 2.0

        # E[ln p(z|pi)]
        self._vl_p_z = self.ns @ self._e_ln_pi_vec

        # E[ln p(pi)]
        self._vl_p_pi = self._ln_c_h0_gamma + ((self.h0_gamma_vec - 1.0) * self._e_ln_pi_vec).sum()

        # E[ln p(theta,tau)]
        diffs = self.hn_mu_vecs - self.h0_mu_vecs
        self._vl_p_theta_tau = (
            (self.c_degree * (-_LOG_2PI + self._e_ln_taus)
                + self._ln_det_h0_lambda_mats
                - self._e_taus * np.einsum('ki,kij,kj->k',diffs,self.h0_lambda_mats,diffs)
                - np.einsum('kij,kij->k',self.h0_lambda_mats,self.hn_lambda_mats_inv)
                ) / 2
            + self.h0_alphas * self._ln_h0_betas - self._gammaln_h0_alphas
            + (self.h0_alphas - 1.0) * self._e_ln_taus - self.h0_betas * self._e_taus
        ).sum()

        # -E[ln q(z)]
        self._vl_q_z = -xlogy(self.r_vecs,self.r_vecs).sum()

        # -E[ln q(pi)]
        self._vl_q_pi = ss_dirichlet.entropy(self.hn_gamma_vec)

        # -E[ln q(theta,tau)]
        self._vl_q_theta_tau =  (
            (self.c_degree * (_LOG_2PI - self._e_ln_taus + 1) / 2
             - np.linalg.slogdet(self.hn_lambda_mats)[1] / 2
             + ss_gamma.entropy(self.hn_alphas,scale=1.0/self.hn_betas)).sum()
        )

        self.vl = (
            self._vl_p_y
            + self._vl_p_z
            + self._vl_p_pi
            + self._vl_p_theta_tau
            + self._vl_q_z
            + self._vl_q_pi
            + self._vl_q_theta_tau
        )

    def _init_random_responsibility(self,x,y):
        self.r_vecs[:] = self.rng.dirichlet(np.ones(self.c_num_classes),self.r_vecs.shape[0])
        self._calc_statistics_with_r(x,y)

    def _update_q_theta_tau(self):
        # q(theta,tau)
        self.hn_lambda_mats[:] = (
            self.h0_lambda_mats + self.x_r_x_mats
        )
        self.hn_lambda_mats_inv[:] = np.linalg.inv(self.hn_lambda_mats)
        self.hn_mu_vecs[:] = np.linalg.solve(
            self.hn_lambda_mats,
            (self._h0_lambda_mu_vecs + self.x_r_y_vecs)[:,:,np.newaxis]
        )[:,:,0]
        self.hn_alphas[:] = (
            self.h0_alphas + self.ns/2.0
        )
        self.hn_betas[:] = self.h0_betas + (
            np.einsum('ki,kij,kj->k',self.h0_mu_vecs,self.h0_lambda_mats,self.h0_mu_vecs)
            + self.y_r_ys
            - np.einsum('ki,kij,kj->k',self.hn_mu_vecs,self.hn_lambda_mats,self.hn_mu_vecs)
        ) / 2.0
        self._calc_q_theta_tau_features()
    
    def _update_q_pi(self):
        self.hn_gamma_vec[:] = self.h0_gamma_vec + self.ns
        self._calc_q_pi_features()
    
    def _update_q_z(self,x,y):
        for k in range(self.c_num_classes):
            diff = y - x @ self.hn_mu_vecs[k]
            self._ln_rho[:,k] = self._e_ln_pi_vec[k] + (
                - _LOG_2PI
                + self._e_ln_taus[k]
                - self._e_taus[k] * diff * diff
                + ((x @ self.hn_lambda_mats_inv[k]) * x).sum(axis=1)
            ) / 2.0
        self.r_vecs[:] = np.exp(self._ln_rho - self._ln_rho.max(axis=1,keepdims=True))
        self.r_vecs[:] /= self.r_vecs.sum(axis=1,keepdims=True)
        self._calc_statistics_with_r(x,y)

    def _init_subsampling(self,x,y):
        n = x.shape[0]
        size = int(np.sqrt(n))
        for k in range(self.c_num_classes):
            indices = self.rng.choice(n,size=size,replace=False,axis=0,shuffle=False)
            x_subsample = x[indices]
            y_subsample = y[indices]
            self.hn_lambda_mats[k] = (
                self.h0_lambda_mats[k] + x_subsample.T @ x_subsample
            )
            self.hn_lambda_mats_inv[k] = np.linalg.inv(self.hn_lambda_mats[k])
            self.hn_mu_vecs[k] = np.linalg.solve(
                self.hn_lambda_mats[k],
                self._h0_lambda_mu_vecs[k] + x_subsample.T @ y_subsample
            )
            self.hn_alphas[k] = (
                self.h0_alphas[k] + size/2.0
            )
            self.hn_betas[k] = self.h0_betas[k] + (
                self.h0_mu_vecs[k] @ self.h0_lambda_mats[k] @ self.h0_mu_vecs[k]
                + y_subsample @ y_subsample
                - self.hn_mu_vecs[k] @ self.hn_lambda_mats[k] @ self.hn_mu_vecs[k]
            ) / 2.0
        self._calc_q_theta_tau_features()
    
    def fit(
            self,
            x,
            y,
            max_itr=1000,
            num_init=10,
            tolerance=1.0E-8,
            init_type='random_responsibility',
            ):            
        """Fit the model to the data.

        This function is a wrapper of the following functions:

        >>> self.reset_hn_params()
        >>> self.update_posterior(x,y,max_itr,tolerance,init_type)
        >>> return self

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        y : numpy ndarray
            float array.
        max_itr : int, optional
            maximum number of iterations, by default 1000
        num_init : int, optional
            number of initializations, by default 10
        tolerance : float, optional
            convergence criterion of variational lower bound, by default 1.0E-8
        init_type : str, optional
            * ``'random_responsibility'``: randomly assign responsibility to ``r_vecs``
            * ``'subsampling'``: for each latent class, extract a subsample whose size is ``int(np.sqrt(x.shape[0]))``.
              and use it to update q(theta_k,tau_k).
            Type of initialization, by default ``'random_responsibility'``
        
        Returns
        -------
        self : LearnModel
            The fitted model.
        """
        self.reset_hn_params()
        self.update_posterior(x,y,max_itr,num_init,tolerance,init_type)
        return self

    def predict(self,x):
        """Predict the data.

        This function is a wrapper of the following functions:
        
        >>> self.calc_pred_dist(x)
        >>> return self.make_prediction(loss="squared")

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        
        Returns
        -------
        Predicted_values : numpy ndarray
            The predicted values under the squared loss function. 
            The size of the predicted values is the same as the sample size of x.
        """
        self.calc_pred_dist(x)
        return self.make_prediction(loss="squared")

    def estimate_latent_vars(self,x,y,loss="0-1"):
        """Estimate latent variables corresponding to `x` under the given criterion.

        Note that the criterion is independently applied to each data point.

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        y : numpy ndarray
            float array.
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"0-1\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        estimates : numpy.ndarray
            The estimated values under the given loss function. 
            If the loss function is \"KL\", the posterior distribution will be returned 
            as a numpy.ndarray whose elements consist of occurence probabilities.
        """
        x,y = self._check_sample(x,y)
        self._ln_rho = np.empty([x.shape[0],self.c_num_classes])
        self.r_vecs = np.empty([x.shape[0],self.c_num_classes])
        self._update_q_z(x,y)

        if loss == "squared":
            return self.r_vecs
        elif loss == "0-1":
            return np.eye(self.c_num_classes,dtype=int)[np.argmax(self.r_vecs,axis=1)]
        elif loss == "KL":
            return self.r_vecs
        else:
            raise(CriteriaError(f"loss={loss} is unsupported. "
                                +"This function supports \"squared\", \"0-1\", and \"KL\"."))

    def estimate_latent_vars_and_update(
            self,
            x,
            y,
            loss="0-1",
            max_itr=100,
            num_init=10,
            tolerance=1.0E-8,
            init_type='random_responsibility',
            ):
        """Estimate latent variables and update the posterior sequentially.

        h0_params will be overwritten by current hn_params 
        before updating hn_params by x
        
        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        y : numpy ndarray
            float array.
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"0-1\".
            This function supports \"squared\" and \"0-1\".
        max_itr : int, optional
            maximum number of iterations, by default 100
        num_init : int, optional
            number of initializations, by default 10
        tolerance : float, optional
            convergence croterion of variational lower bound, by default 1.0E-8
        init_type : str, optional
            * ``'subsampling'``: for each latent class, extract a subsample whose size is ``int(np.sqrt(x.shape[0]))``.
              and use its mean and covariance matrix as an initial values of ``hn_m_vecs`` and ``hn_lambda_mats``.
            * ``'random_responsibility'``: randomly assign responsibility to ``r_vecs``
            Type of initialization, by default ``'subsampling'``

        Returns
        -------
        estimates : numpy.ndarray
            The estimated values under the given loss function. 
            If the loss function is \"KL\", the posterior distribution will be returned 
            as a numpy.ndarray whose elements consist of occurence probabilities.
        """
        z_hat = self.estimate_latent_vars(x,y,loss=loss)
        self.overwrite_h0_params()
        self.update_posterior(
            x,
            y,
            max_itr=max_itr,
            num_init=num_init,
            tolerance=tolerance,
            init_type=init_type
            )
        return z_hat
