# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
# Document Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
import warnings
import numpy as np
from scipy.stats import multivariate_normal as ss_multivariate_normal
from scipy.stats import wishart as ss_wishart
from scipy.stats import multivariate_t as ss_multivariate_t
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution

    Parameters
    ----------
    num_classes : int, optional
        a positive integer. Default is None, in which case 
        a value consistent with ``pi_vec``, ``mu_vecs``, 
        ``lambda_mats``, and ``h_alpha_vec`` is used.
        If all of them are not given, num_classes is assumed to be 2.
    pi_vec : numpy.ndarray, optional
        a real vector in :math:`[0, 1]^K`, by default [1/K, 1/K, ... , 1/K]
    degree : int, optional
        a positive integer. Default is None, in which case 
        a value consistent with ``mu_vecs``, ``lambda_mats``, 
        ``h_m_vec``, ``h_w_mat``, and ``h_nu` is used. 
        If all of them are not given, degree is assumed to be 1.
    mu_vecs : numpy.ndarray, optional
        vectors of real numbers, by default zero vectors.
    lambda_mats : numpy.ndarray, optional
        positive definite symetric matrices, by default the identity matrices
    h_alpha_vec : numpy.ndarray, optional
        a vector of positive real numbers, by default [1/2, 1/2, ... , 1/2]
    h_m_vec : numpy.ndarray, optional
        a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
    h_kappa : float, optional
        a positive real number, by default 1.0
    h_nu : float, optional
        a real number > degree-1, by default the value of ``degree``
    h_w_mat : numpy.ndarray, optional
        a positive definite symetric matrix, by default the identity matrix
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(
        self,
        *,
        num_classes=None,
        pi_vec=None,
        degree=None,
        mu_vecs=None,
        lambda_mats=None,
        h_alpha_vec=None,
        h_m_vec=None,
        h_kappa=1.0,
        h_nu=None,
        h_w_mat=None,
        seed=None
        ):
        
        _tmp_degree = np.zeros(5,dtype=int)
        if degree is not None:
            _tmp_degree[0] = _check.pos_int(degree,'degree',ParameterFormatError)
        if mu_vecs is not None:
            _tmp_degree[1] = _check.float_vecs(mu_vecs,'mu_vecs',ParameterFormatError).shape[-1]
        if lambda_mats is not None:
            _tmp_degree[2] = _check.pos_def_sym_mats(lambda_mats,'lambda_mats',ParameterFormatError).shape[-1]
        if h_m_vec is not None:
            _tmp_degree[3] = _check.float_vec(h_m_vec,'h_m_vec',ParameterFormatError).shape[0]
        if h_w_mat is not None:
            _tmp_degree[4] = _check.pos_def_sym_mat(h_w_mat,'h_w_mat',ParameterFormatError).shape[-1]

        _index = (_tmp_degree > 0)
        if np.sum(_index) == 0:
            self.degree = 1 # default value for self.degree
        elif np.all(_tmp_degree[_index] == (_tmp_degree[_index])[0]):
            self.degree = (_tmp_degree[_index])[0]
        else:
            raise(ParameterFormatError(
                "degree and dimensions of mu_vecs, lambda_mats,"
                +" h_m_vec, h_w_mat must be the same,"
                +" if two or more of them are specified."))
        
        _tmp_num_classes = np.zeros(5,dtype=int)
        if num_classes is not None:
            _tmp_num_classes[0] = _check.pos_int(num_classes,'num_classes',ParameterFormatError)
        if mu_vecs is not None:
            _tmp_num_classes[1] = np.prod(_check.float_vecs(mu_vecs,'mu_vecs',ParameterFormatError).shape[:-1])
        if lambda_mats is not None:
            _tmp_num_classes[2] = np.prod(_check.pos_def_sym_mats(lambda_mats,'lambda_mats',ParameterFormatError).shape[:-2])
        if pi_vec is not None:
            _tmp_num_classes[3] = _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError).shape[0]
        if h_alpha_vec is not None:
            _tmp_num_classes[4] = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError).shape[0]

        _index = (_tmp_num_classes > 0)
        if np.sum(_index) == 0:
            self.num_classes = 2 # default value for self.degree
        elif np.all(_tmp_num_classes[_index] == (_tmp_num_classes[_index])[0]):
            self.num_classes = (_tmp_num_classes[_index])[0]
        else:
            raise(ParameterFormatError(
                "num_classes, dimensions of pi_vec and h_alpha_vec,"
                +" and the first dimension of mu_vecs and lambda_mats"
                +" must be the same, if two or more of them are specified."))

        if mu_vecs is None:
            self.mu_vecs = np.zeros([self.num_classes,self.degree])
        else:
            self.mu_vecs = mu_vecs.reshape(self.num_classes,self.degree)

        if lambda_mats is None:
            self.lambda_mats = np.tile(np.identity(self.degree),(self.num_classes,1,1))
        else:
            self.lambda_mats = lambda_mats.reshape(self.num_classes,self.degree,self.degree)

        if h_m_vec is None:
            self.h_m_vec = np.zeros(self.degree)
        else:
            self.h_m_vec = h_m_vec

        if h_w_mat is None:
            self.h_w_mat = np.identity(self.degree)
        else:
            self.h_w_mat = h_w_mat

        if h_nu is None:
            self.h_nu = float(self.degree)
        else:
            self.h_nu = _check.pos_float(h_nu,'h_nu',ParameterFormatError)
            if self.h_nu <= self.degree - 1:
                raise(ParameterFormatError(
                    "degree must be smaller than h_nu + 1"))

        if pi_vec is None:
            self.pi_vec = np.ones(self.num_classes) / self.num_classes
        else:
            self.pi_vec = pi_vec

        if h_alpha_vec is None:
            self.h_alpha_vec = np.ones(self.num_classes) / 2
        else:
            self.h_alpha_vec = h_alpha_vec
        
        self.h_kappa = _check.pos_float(h_kappa,'h_kappa',ParameterFormatError)
        self.rng = np.random.default_rng(seed)
        
    def set_h_params(self,h_alpha_vec,h_m_vec,h_kappa,h_nu,h_w_mat):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_alpha_vec : numpy.ndarray
            a vector of positive real numbers
        h_m_vec : numpy.ndarray
            a vector of real numbers
        h_kappa : float
            a positive real number
        h_nu : float
            a real number > degree-1
        h_w_mat : numpy.ndarray
            a positive definite symetric matrix
        """
        self.h_alpha_vec = _check.pos_float_vec(h_alpha_vec,'h_alpha_vec',ParameterFormatError)
        self.h_m_vec = _check.float_vec(h_m_vec,'h_m_vec',ParameterFormatError)
        self.h_kappa = _check.pos_float(h_kappa,'h_kappa',ParameterFormatError)
        self.h_nu = _check.pos_float(h_nu,'h_nu',ParameterFormatError)
        self.h_w_mat = _check.pos_def_sym_mat(h_w_mat,'h_w_mat',ParameterFormatError)

        if (self.h_m_vec.shape[0] != self.h_w_mat.shape[0]
            or self.h_nu <= self.h_m_vec.shape[0] - 1):
                raise(ParameterFormatError(
                    "They must be h_m_vec.shape[0] == h_w_mat.shape[0]"
                    +" and h_nu > h_m_vec.shape[0] - 1."))

        self.degree = self.h_m_vec.shape[0]
        self.num_classes = self.h_alpha_vec.shape[0]

        if self.degree != self.mu_vecs.shape[-1]:
            self.mu_vecs = np.zeros([self.num_classes,self.degree])
            warnings.warn("mu_vecs is reinitialized to zero vectors because dimensions of mu_vecs and h_params are mismatched.", ParameterFormatWarning)

        if self.degree != self.lambda_mats.shape[-1]:
            self.lambda_mats = np.tile(np.identity(self.degree),[self.num_classes,1,1])
            warnings.warn("lambda_mats is reinitialized to the identity matrices because dimensions of lambda_mats and h_params are mismatched.", ParameterFormatWarning)
        
        if self.num_classes != self.pi_vec.shape[0]:
            self.pi_vec = np.ones(self.num_classes) / self.num_classes
            warnings.warn("pi_vec is reinitialized to [1/num_classes, 1/num_classes, ..., 1/num_classes] because dimensions of mu_vec and h_params are mismatched.", ParameterFormatWarning)

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.
        
        Returns
        -------
        h_params : {str:float, np.ndarray}
            * ``"h_alpha_vec"`` : The value of ``self.h_alpha_vec``
            * ``"h_m_vec"`` : The value of ``self.h_mu_vec``
            * ``"h_kappa"`` : The value of ``self.h_kappa``
            * ``"h_nu"`` : The value of ``self.h_nu``
            * ``"h_w_mat"`` : The value of ``self.h_lambda_mat``
        """
        return {"h_alpha_vec":self.h_alpha_vec,
                "h_m_vec":self.h_m_vec, 
                "h_kappa":self.h_kappa, 
                "h_nu":self.h_nu, 
                "h_w_mat":self.h_w_mat}
    
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.mu_vec`` and ``self.lambda_mat``.
        """
        pass
        # self.lambda_mat[:] = ss_wishart.rvs(df=self.h_nu,scale=self.h_w_mat,random_state=self.rng)
        # self.mu_vec[:] = self.rng.multivariate_normal(mean=self.h_m_vec,cov=np.linalg.inv(self.h_kappa*self.lambda_mat))
    
    def set_params(self,pi_vec,mu_vecs,lambda_mats):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        pi_vec : numpy.ndarray
            a real vector in :math:`[0, 1]^K`. The sum of its elements must be 1.
        mu_vecs : numpy.ndarray
            vectors of real numbers
        lambda_mats : numpy.ndarray
            positive definite symetric matrices
        """
        self.pi_vec = _check.float_vec_sum_1(pi_vec,'pi_vec',ParameterFormatError)
        _tmp_mu_vecs = _check.float_vecs(mu_vecs,'mu_vecs',ParameterFormatError)
        _tmp_shape = _tmp_mu_vecs.shape
        self.mu_vecs = _tmp_mu_vecs.reshape(-1,_tmp_shape[-1])
        _tmp_lambda_mats = _check.pos_def_sym_mats(lambda_mats,'lambda_mats',ParameterFormatError)
        _tmp_shape = _tmp_lambda_mats.shape
        self.lambda_mats = _tmp_lambda_mats.reshape(-1,_tmp_shape[-2],_tmp_shape[-1])
        if (self.pi_vec.shape[0] != self.mu_vecs.shape[0]
            or self.pi_vec.shape[0] != self.lambda_mats.shape[0]):
            raise(ParameterFormatError("The dimension of pi_vec"
                +" and the first dimension of mu_vecs and lambda_mats"
                +" must be the same"))

        if (self.mu_vecs.shape[-1] != self.lambda_mats.shape[-1]):
            raise(ParameterFormatError("The dimensions of mu_vecs and lambda_mats must be the same"))

        self.degree = self.mu_vecs.shape[-1]
        if self.degree != self.h_m_vec.shape[0]:
            self.h_m_vec = np.zeros(self.degree)
            warnings.warn("h_m_vec is reinitialized to [0.0, 0.0, ..., 0.0] because dimension of h_m_vec and mu_vec are mismatched.", ParameterFormatWarning)
        if self.degree != self.h_w_mat.shape[0]:
            self.h_w_mat = np.identity(self.degree)
            warnings.warn("h_w_mat is reinitialized to the identity matrix because dimension of h_w_mat and lambda_mat are mismatched.", ParameterFormatWarning)
        
        self.num_classes = self.pi_vec.shape[0]
        if self.num_classes != self.h_alpha_vec.shape[0]:
            self.h_alpha_vec = np.ones(self.num_classes) / 2
            warnings.warn("h_alpha_vec is reinitialized to [1/2, 1/2, ... , 1/2]] because dimension of h_alpha_vec and pi_vec are mismatched.", ParameterFormatWarning)

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : {str:float, numpy.ndarray}
            * ``"pi_vec"`` : The value of ``self.pi_vec``
            * ``"mu_vecs"`` : The value of ``self.mu_vecs``
            * ``"lambda_mats"`` : The value of ``self.lambda_mats``
        """
        return {"pi_vec":self.pi_vec, "mu_vecs":self.mu_vecs, "lambda_mats":self.lambda_mats}

    def gen_sample(self,sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            2-dimensional array whose shape is ``(sammple_size,degree)`` and its elements are real number.
        """
        pass
        # _check.pos_int(sample_size,'sample_size',DataFormatError)
        # return self.rng.multivariate_normal(mean=self.mu_vec,cov=np.linalg.inv(self.lambda_mat),size=sample_size)
        
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
        pass
        # np.savez_compressed(filename,x=self.gen_sample(sample_size))

    def visualize_model(self,sample_size=100):
        """Visualize the stochastic data generative model and generated samples.
        
        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 1
        
        Examples
        --------
        >>> from bayesml import multivariate_normal
        >>> model = multivariate_normal.GenModel()
        >>> model.visualize_model()
        mu:
        [0. 0.]
        lambda_mat:
        [[1. 0.]
         [0. 1.]]
        
        .. image:: ./images/multivariate_normal_example.png
        """
        pass
        # if self.degree == 1:
        #     print(f"mu: {self.mu_vec}")
        #     print(f"lambda_mat: {self.lambda_mat}")
        #     lambda_mat_inv = np.linalg.inv(self.lambda_mat)
        #     fig, axes = plt.subplots()
        #     sample = self.gen_sample(sample_size)
        #     x = np.linspace(sample.min()-(sample.max()-sample.min())*0.25,
        #                     sample.max()+(sample.max()-sample.min())*0.25,
        #                     100)
        #     axes.plot(x,ss_multivariate_normal.pdf(x,self.mu_vec,lambda_mat_inv))
        #     axes.hist(sample,density=True)
        #     axes.set_xlabel("x")
        #     axes.set_ylabel("Density or frequency")
        #     plt.show()

        # elif self.degree == 2:
        #     print(f"mu:\n{self.mu_vec}")
        #     print(f"lambda_mat:\n{self.lambda_mat}")
        #     lambda_mat_inv = np.linalg.inv(self.lambda_mat)
        #     fig, axes = plt.subplots()
        #     sample = self.gen_sample(sample_size)
        #     x = np.linspace(sample[:,0].min()-(sample[:,0].max()-sample[:,0].min())*0.25,
        #                     sample[:,0].max()+(sample[:,0].max()-sample[:,0].min())*0.25,
        #                     100)
        #     y = np.linspace(sample[:,1].min()-(sample[:,1].max()-sample[:,1].min())*0.25,
        #                     sample[:,1].max()+(sample[:,1].max()-sample[:,1].min())*0.25,
        #                     100)
        #     xx, yy = np.meshgrid(x,y)
        #     grid = np.empty((100,100,2))
        #     grid[:,:,0] = xx
        #     grid[:,:,1] = yy
        #     axes.contourf(xx,yy,ss_multivariate_normal.pdf(grid,self.mu_vec,lambda_mat_inv),cmap='Blues')
        #     axes.plot(self.mu_vec[0],self.mu_vec[1],marker="x",color='red')
        #     axes.set_xlabel("x[0]")
        #     axes.set_ylabel("x[1]")
        #     axes.scatter(sample[:,0],sample[:,1],color='tab:orange')
        #     plt.show()

        # else:
        #     raise(ParameterFormatError("if degree > 2, it is impossible to visualize the model by this function."))

# class LearnModel(base.Posterior,base.PredictiveMixin):
#     """The posterior distribution and the predictive distribution.

#     Parameters
#     ----------
#     degree : int, optional
#         a positive integer. Default is None, in which case 
#         a value consistent with ``h_m_vec``, ``h_w_mat``, 
#         and ``h_nu` is used. If all of them are not given, 
#         degree is assumed to be 2.
#     h0_m_vec : numpy.ndarray, optional
#         a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
#     h0_kappa : float, optional
#         a positive real number, by default 1.0
#     h0_nu : float, optional
#         a real number > degree-1, by default the value of ``degree``
#     h0_w_mat : numpy.ndarray, optional
#         a positive definite symetric matrix, by default the identity matrix

#     Attributes
#     ----------
#     h0_w_mat_inv : numpy.ndarray
#         the inverse matrix of h0_w_mat
#     hn_m_vec : numpy.ndarray
#         a vector of real numbers
#     hn_kappa : float
#         a positive real number
#     hn_nu : float
#         a real number
#     hn_w_mat : numpy.ndarray
#         a positive definite symetric matrix
#     hn_w_mat_inv : numpy.ndarray
#         the inverse matrix of hn_w_mat
#     p_m_vec : numpy.ndarray
#         a vector of real numbers
#     p_nu : float, optional
#         a positive real number
#     p_v_mat : numpy.ndarray
#         a positive definite symetric matrix
#     p_v_mat_inv : numpy.ndarray
#         the inverse matrix of p_v_mat
#     """
#     def __init__(self,degree=None,h0_m_vec=None,h0_kappa=1.0,h0_nu=None,h0_w_mat=None):
#         if degree is not None:
#             self.degree = _check.pos_int(degree,'degree',ParameterFormatError)
#             if h0_m_vec is None:
#                 self.h0_m_vec = np.zeros(self.degree)
#             else:
#                 self.h0_m_vec = _check.float_vec(h0_m_vec,'h0_m_vec',ParameterFormatError)

#             if h0_w_mat is None:
#                 self.h0_w_mat = np.identity(self.degree)
#             else:
#                 self.h0_w_mat = _check.pos_def_sym_mat(h0_w_mat,'h0_w_mat',ParameterFormatError)

#             if h0_nu is None:
#                 self.h0_nu = float(self.degree)
#             else:
#                 self.h0_nu = _check.pos_float(h0_nu,'h0_nu',ParameterFormatError)
        
#         elif h0_m_vec is not None:
#             self.h0_m_vec = _check.float_vec(h0_m_vec,'h0_m_vec',ParameterFormatError)
#             self.degree = self.h0_m_vec.shape[0]
#             if h0_w_mat is None:
#                 self.h0_w_mat = np.identity(self.degree)
#             else:
#                 self.h0_w_mat = _check.pos_def_sym_mat(h0_w_mat,'h0_w_mat',ParameterFormatError)

#             if h0_nu is None:
#                 self.h0_nu = float(self.degree)
#             else:
#                 self.h0_nu = _check.pos_float(h0_nu,'h0_nu',ParameterFormatError)
        
#         elif h0_w_mat is not None:
#             self.h0_w_mat = _check.pos_def_sym_mat(h0_w_mat,'h0_w_mat',ParameterFormatError)
#             self.degree = self.h0_w_mat.shape[0]
#             self.h0_m_vec = np.zeros(self.degree)
#             if h0_nu is None:
#                 self.h0_nu = float(self.degree)
#             else:
#                 self.h0_nu = _check.pos_float(h0_nu,'h0_nu',ParameterFormatError)
        
#         elif h0_nu is not None:
#             self.h0_nu = _check.pos_float(h0_nu,'h0_nu',ParameterFormatError)
#             if self.h0_nu <= 1.0:
#                 self.degree = 1
#             else:
#                 self.degree = 2
#             self.h0_m_vec = np.zeros(self.degree)
#             self.h0_w_mat = np.identity(self.degree)
        
#         else:
#             self.degree = 2
#             self.h0_m_vec = np.zeros(self.degree)
#             self.h0_w_mat = np.identity(self.degree)
#             self.h0_nu = float(self.degree)

#         if (self.degree != self.h0_m_vec.shape[0]
#             or self.degree != self.h0_w_mat.shape[0]
#             or self.h0_nu <= self.degree - 1):
#                 raise(ParameterFormatError(
#                     "degree and dimensions of"
#                     +" h0_m_vec, h0_w_mat must be the same,"
#                     +" and degree must be smaller than h0_nu + 1,"
#                     +" if two or more of them are specified."))

#         self.h0_kappa = _check.pos_float(h0_kappa,'h0_kappa',ParameterFormatError)
#         self.h0_w_mat_inv = np.linalg.inv(self.h0_w_mat)

#         self.hn_m_vec = np.copy(self.h0_m_vec)
#         self.hn_kappa = self.h0_kappa
#         self.hn_nu = self.h0_nu
#         self.hn_w_mat = np.copy(self.h0_w_mat)
#         self.hn_w_mat_inv = np.copy(self.h0_w_mat_inv)

#         self.p_m_vec = np.copy(self.hn_m_vec)
#         self.p_nu = self.hn_nu - self.degree + 1
#         self.p_v_mat = self.hn_kappa*self.p_nu/(self.hn_kappa+1) * self.hn_w_mat
#         self.p_v_mat_inv = (self.hn_kappa+1)/self.hn_kappa/self.p_nu * self.hn_w_mat_inv

#     def set_h0_params(self,h0_m_vec,h0_kappa,h0_nu,h0_w_mat):
#         """Set the hyperparameters of the prior distribution.
        
#         Parameters
#         ----------
#         h0_m_vec : numpy.ndarray
#             a vector of real numbers
#         h0_kappa : float
#             a positive real number
#         h0_nu : float
#             a real number > degree-1
#         h0_w_mat : numpy.ndarray
#             a positive definite symetric matrix
#         """
#         self.h0_m_vec = _check.float_vec(h0_m_vec,'h0_m_vec',ParameterFormatError)
#         self.h0_kappa = _check.pos_float(h0_kappa,'h0_kappa',ParameterFormatError)
#         self.h0_nu = _check.pos_float(h0_nu,'h0_nu',ParameterFormatError)
#         self.h0_w_mat = _check.pos_def_sym_mat(h0_w_mat,'h0_w_mat',ParameterFormatError)

#         if (self.h0_m_vec.shape[0] != self.h0_w_mat.shape[0]
#             or self.h0_nu <= self.h0_m_vec.shape[0] - 1):
#                 raise(ParameterFormatError(
#                     "They must be h0_m_vec.shape[0] == h0_w_mat.shape[0]"
#                     +" and h0_nu > h0_m_vec.shape[0] - 1."))

#         self.h0_w_mat_inv = np.linalg.inv(self.h0_w_mat)
#         self.degree = self.h0_m_vec.shape[0]

#         self.reset_hn_params()

#     def get_h0_params(self):
#         """Get the initial values of the hyperparameters of the posterior distribution.

#         Returns
#         -------
#         h0_params : dict of {str: float, numpy.ndarray}
#             * ``"h0_m_vec"`` : The value of ``self.h0_m_vec``
#             * ``"h0_kappa"`` : The value of ``self.h0_kappa``
#             * ``"h0_nu"`` : The value of ``self.h0_nu``
#             * ``"h0_w_mat"`` : The value of ``self.h0_w_mat``
#         """
#         return {"h0_m_vec":self.h0_m_vec, "h0_kappa":self.h0_kappa, "h0_nu":self.h0_nu, "h0_w_mat":self.h0_w_mat}
    
#     def set_hn_params(self,hn_m_vec,hn_kappa,hn_nu,hn_w_mat):
#         """Set updated values of the hyperparameter of the posterior distribution.

#         Parameters
#         ----------
#         hn_m_vec : numpy.ndarray
#             a vector of real numbers
#         hn_kappa : float
#             a positive real number
#         hn_nu : float
#             a real number > degree-1
#         hn_w_mat : numpy.ndarray
#             a positive definite symetric matrix
#         """
#         self.hn_m_vec = _check.float_vec(hn_m_vec,'hn_m_vec',ParameterFormatError)
#         self.hn_kappa = _check.pos_float(hn_kappa,'hn_kappa',ParameterFormatError)
#         self.hn_nu = _check.pos_float(hn_nu,'hn_nu',ParameterFormatError)
#         self.hn_w_mat = _check.pos_def_sym_mat(hn_w_mat,'hn_w_mat',ParameterFormatError)

#         if (self.hn_m_vec.shape[0] != self.hn_w_mat.shape[0]
#             or self.hn_nu <= self.hn_m_vec.shape[0] - 1):
#                 raise(ParameterFormatError(
#                     "They must be hn_m_vec.shape[0] == hn_w_mat.shape[0]"
#                     +" and hn_nu > hn_m_vec.shape[0] - 1."))

#         self.hn_w_mat_inv = np.linalg.inv(self.hn_w_mat)
#         self.degree = self.hn_m_vec.shape[0]

#         self.calc_pred_dist()

#     def get_hn_params(self):
#         """Get the hyperparameters of the posterior distribution.

#         Returns
#         -------
#         hn_params : dict of {str: numpy.ndarray}
#             * ``"hn_m_vec"`` : The value of ``self.hn_m_vec``
#             * ``"hn_kappa"`` : The value of ``self.hn_kappa``
#             * ``"hn_nu"`` : The value of ``self.hn_nu``
#             * ``"hn_w_mat"`` : The value of ``self.hn_w_mat``
#         """
#         return {"hn_m_vec":self.hn_m_vec, "hn_kappa":self.hn_kappa, "hn_nu":self.hn_nu, "hn_w_mat":self.hn_w_mat}
    
#     def reset_hn_params(self):
#         """Reset the hyperparameters of the posterior distribution to their initial values.
        
#         They are reset to `self.h0_m_vec`, `self.h0_kappa`, `self.h0_nu` and `self.h0_w_hat`.
#         Note that the parameters of the predictive distribution are also calculated from `self.h0_m_vec`, `self.h0_kappa`, `self.h0_nu` and `self.h0_w_hat`.
#         """
#         self.hn_m_vec = np.copy(self.h0_m_vec)
#         self.hn_kappa = self.h0_kappa
#         self.hn_nu = self.h0_nu
#         self.hn_w_mat = np.copy(self.h0_w_mat)
#         self.hn_w_mat_inv = np.copy(self.h0_w_mat_inv)

#         self.calc_pred_dist()
    
#     def overwrite_h0_params(self):
#         """Overwrite the initial values of the hyperparameters of the posterior distribution by the learned values.
        
#         They are overwitten by `self.hn_m_vec`, `self.hn_kappa`, `self.hn_nu` and `self.hn_w_mat`.
#         Note that the parameters of the predictive distribution are also calculated from `self.hn_m_vec`, `self.hn_kappa`, `self.hn_nu` and `self.hn_w_mat`.
#         """
#         self.h0_m_vec = np.copy(self.hn_m_vec)
#         self.h0_kappa = self.hn_kappa
#         self.h0_nu = self.hn_nu
#         self.h0_w_mat = np.copy(self.hn_w_mat)
#         self.h0_w_mat_inv = np.copy(self.hn_w_mat_inv)

#         self.calc_pred_dist()

#     def update_posterior(self,x):
#         """Update the hyperparameters of the posterior distribution using traning data.

#         Parameters
#         ----------
#         x : numpy.ndarray
#             All the elements must be real number.
#         """
#         _check.float_vecs(x,'x',DataFormatError)
#         if self.degree > 1 and x.shape[-1] != self.degree:
#             raise(DataFormatError(f"x.shape[-1] must be degree:{self.degree}"))
#         x = x.reshape(-1,self.degree)

#         n = x.shape[0]
#         x_bar = x.sum(axis=0)/n

#         self.hn_w_mat_inv[:] = (self.hn_w_mat_inv + (x-x_bar).T @ (x-x_bar)
#                                 + (x_bar - self.hn_m_vec)[:,np.newaxis] @ (x_bar - self.hn_m_vec)[np.newaxis,:]
#                                   * self.hn_kappa * n / (self.hn_kappa + n))
#         self.hn_m_vec[:] = (self.hn_kappa*self.hn_m_vec + n*x_bar) / (self.hn_kappa+n)
#         self.hn_kappa += n
#         self.hn_nu += n

#         self.hn_w_mat[:] = np.linalg.inv(self.hn_w_mat_inv) 

#     def estimate_params(self,loss="squared"):
#         """Estimate the parameter of the stochastic data generative model under the given criterion.

#         Note that the criterion is applied to estimating ``mu_vec`` and ``lambda_mat`` independently.
#         Therefore, a tuple of the student's t-distribution and the wishart distribution will be returned when loss=\"KL\"

#         Parameters
#         ----------
#         loss : str, optional
#             Loss function underlying the Bayes risk function, by default \"squared\".
#             This function supports \"squared\", \"0-1\", and \"KL\".

#         Returns
#         -------
#         Estimates : tuple of {numpy ndarray, float, None, or rv_frozen}
#             * ``mu_vec_hat`` : the estimate for mu_vec
#             * ``lambda_mat_hat`` : the estimate for lambda_mat
#             The estimated values under the given loss function. If it is not exist, `None` will be returned.
#             If the loss function is \"KL\", the posterior distribution itself will be returned
#             as rv_frozen object of scipy.stats.

#         See Also
#         --------
#         scipy.stats.rv_continuous
#         scipy.stats.rv_discrete
#         """

#         if loss == "squared":
#             return self.hn_m_vec, self.hn_nu * self.hn_w_mat
#         elif loss == "0-1":
#             if self.hn_nu >= self.degree + 1:
#                 return self.hn_m_vec, (self.hn_nu - self.degree - 1) * self.hn_w_mat
#             else:
#                 warnings.warn("MAP estimate of lambda_mat doesn't exist for the current hn_nu.",ResultWarning)
#                 return self.hn_m_vec, None
#         elif loss == "KL":
#             return (ss_multivariate_t(loc=self.hn_m_vec,
#                                         shape=self.hn_w_mat_inv / self.hn_kappa / (self.hn_nu - self.degree + 1),
#                                         df=self.hn_nu - self.degree + 1),
#                     ss_wishart(df=self.hn_nu,scale=self.hn_w_mat))
#         else:
#             raise(CriteriaError("Unsupported loss function! "
#                                 +"This function supports \"squared\", \"0-1\", and \"KL\"."))
    
#     def visualize_posterior(self):
#         """Visualize the posterior distribution for the parameter.
        
#         Examples
#         --------
#         >>> from bayesml import multivariate_normal
#         >>> gen_model = multivariate_normal.GenModel()
#         >>> x = gen_model.gen_sample(100)
#         >>> learn_model = multivariate_normal.LearnModel()
#         >>> learn_model.update_posterior(x)
#         >>> learn_model.visualize_posterior()
#         hn_m_vec:
#         [-0.06924909  0.08126454]
#         hn_kappa:
#         101.0
#         hn_nu:
#         102.0
#         hn_w_mat:
#         [[ 0.00983415 -0.00059828]
#         [-0.00059828  0.00741698]]
#         E[lambda_mat]=
#         [[ 1.0030838  -0.06102455]
#         [-0.06102455  0.7565315 ]]

#         .. image:: ./images/multivariate_normal_posterior.png
#         """
#         print("hn_m_vec:")
#         print(f"{self.hn_m_vec}")
#         print("hn_kappa:")
#         print(f"{self.hn_kappa}")
#         print("hn_nu:")
#         print(f"{self.hn_nu}")
#         print("hn_w_mat:")
#         print(f"{self.hn_w_mat}")
#         print("E[lambda_mat]=")
#         print(f"{self.hn_nu * self.hn_w_mat}")
#         mu_vec_pdf, w_mat_pdf = self.estimate_params(loss="KL")
#         if self.degree == 1:
#             fig, axes = plt.subplots(1,2)
#             # for mu_vec
#             x = np.linspace(self.hn_m_vec[0]-4.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[0,0]),
#                             self.hn_m_vec[0]+4.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[0,0]),
#                             100)
#             axes[0].plot(x,mu_vec_pdf.pdf(x))
#             axes[0].set_xlabel("mu_vec")
#             axes[0].set_ylabel("Density")
#             # for lambda_mat
#             x = np.linspace(max(1.0e-8,self.hn_nu*self.hn_w_mat)-4.0*np.sqrt(self.hn_nu/2.0)*(2.0*self.hn_w_mat),
#                             self.hn_nu*self.hn_w_mat+4.0*np.sqrt(self.hn_nu/2.0)*(2.0*self.hn_w_mat),
#                             100)
#             print(self.hn_w_mat)
#             axes[1].plot(x[:,0,0],w_mat_pdf.pdf(x[:,0,0]))
#             axes[1].set_xlabel("w_mat")
#             axes[1].set_ylabel("Density")

#             fig.tight_layout()
#             plt.show()

#         elif self.degree == 2:
#             fig, axes = plt.subplots()
#             x = np.linspace(self.hn_m_vec[0]-3.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[0,0]),
#                             self.hn_m_vec[0]+3.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[0,0]),
#                             100)
#             y = np.linspace(self.hn_m_vec[1]-3.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[1,1]),
#                             self.hn_m_vec[1]+3.0*np.sqrt((self.hn_w_mat_inv / self.hn_kappa / self.hn_nu)[1,1]),
#                             100)
#             xx, yy = np.meshgrid(x,y)
#             grid = np.empty((100,100,2))
#             grid[:,:,0] = xx
#             grid[:,:,1] = yy
#             axes.contourf(xx,yy,mu_vec_pdf.pdf(grid),cmap='Blues')
#             axes.plot(self.hn_m_vec[0],self.hn_m_vec[1],marker="x",color='red')
#             axes.set_xlabel("mu_vec[0]")
#             axes.set_ylabel("mu_vec[1]")
#             plt.show()

#         else:
#             raise(ParameterFormatError("if degree > 2, it is impossible to visualize the model by this function."))
    
#     def get_p_params(self):
#         """Get the parameters of the predictive distribution.

#         Returns
#         -------
#         p_params : dict of {str: numpy.ndarray}
#             * ``"p_m_vec"`` : The value of ``self.p_m_vec``
#             * ``"p_nu"`` : The value of ``self.p_nu``
#             * ``"p_v_mat"`` : The value of ``self.p_v_mat``
#         """
#         return {"p_m_vec":self.p_m_vec, "p_nu":self.p_nu, "p_v_mat":self.p_v_mat}
    
#     def calc_pred_dist(self):
#         """Calculate the parameters of the predictive distribution."""
#         self.p_m_vec = np.copy(self.hn_m_vec)
#         self.p_nu = self.hn_nu - self.degree + 1
#         self.p_v_mat = self.hn_kappa*self.p_nu/(self.hn_kappa+1) * self.hn_w_mat
#         self.p_v_mat_inv = (self.hn_kappa+1)/self.hn_kappa/self.p_nu * self.hn_w_mat_inv

#     def make_prediction(self,loss="squared"):
#         """Predict a new data point under the given criterion.

#         Parameters
#         ----------
#         loss : str, optional
#             Loss function underlying the Bayes risk function, by default \"squared\".
#             This function supports \"squared\", \"0-1\", and \"KL\".

#         Returns
#         -------
#         Predicted_value : {float, numpy.ndarray}
#             The predicted value under the given loss function. 
#             If the loss function is \"KL\", the posterior distribution itself will be returned
#             as rv_frozen object of scipy.stats.

#         See Also
#         --------
#         scipy.stats.rv_continuous
#         scipy.stats.rv_discrete
#         """
#         if loss == "squared" or loss == "0-1":
#             return self.p_m_vec
#         elif loss == "KL":
#             return ss_multivariate_t(loc=self.p_m_vec,
#                                      shape=self.p_v_mat_inv,
#                                      df=self.p_nu)
#         else:
#             raise(CriteriaError("Unsupported loss function! "
#                                 +"This function supports \"squared\", \"0-1\", and \"KL\"."))

#     def pred_and_update(self,x,loss="squared"):
#         """Predict a new data point and update the posterior sequentially.

#         Parameters
#         ----------
#         x : numpy.ndarray
#             It must be a degree-dimensional vector
#         loss : str, optional
#             Loss function underlying the Bayes risk function, by default \"squared\".
#             This function supports \"squared\", \"0-1\", and \"KL\".

#         Returns
#         -------
#         Predicted_value : {float, numpy.ndarray}
#             The predicted value under the given loss function. 
#         """
#         _check.float_vec(x,'x',DataFormatError)
#         if x.shape != (self.degree,):
#             raise(DataFormatError(f"x must be a 1-dimensional float array whose size is degree: {self.degree}."))
#         self.calc_pred_dist()
#         prediction = self.make_prediction(loss=loss)
#         self.update_posterior(x[np.newaxis,:])
#         return prediction
