# Code Author
# Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
import warnings
import numpy as np
from scipy.stats import multivariate_normal as ss_multivariate_normal
from scipy.special import expit
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
    w_vec : numpy ndarray, optional
        a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
    h_mu_vec : numpy ndarray, optional
        a vector of real numbers, by default [0.0, 0.0, ... , 0.0]
    h_lambda_mat : numpy ndarray, optional
        a positive definate matrix, by default the identity matrix
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None
    """
    def __init__(
            self,
            c_degree,
            *,
            w_vec=None,
            h_mu_vec=None,
            h_lambda_mat=None,
            seed=None
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.w_vec = np.zeros([self.c_degree])

        # h_params
        self.h_mu_vec = np.zeros([self.c_degree])
        self.h_lambda_mat = np.identity(self.c_degree)

        self.set_params(w_vec)
        self.set_h_params(h_mu_vec,h_lambda_mat)

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int}
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {'c_degree':self.c_degree}

    def set_params(
            self,
            w_vec=None,
            ):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        w_vec : numpy ndarray, optional
            a vector of real numbers, by default None
        """
        if w_vec is not None:
            _check.float_vec(w_vec,'w_vec',ParameterFormatError)
            _check.shape_consistency(
                w_vec.shape[0],'w_vec.shape[0]',
                self.c_degree,'self.c_degree',
                ParameterFormatError
                )
            self.w_vec[:] = w_vec

    def set_h_params(
            self,
            h_mu_vec=None,
            h_lambda_mat=None,
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_mu_vec : numpy ndarray, optional
            a vector of real numbers, by default None.
        h_lambda_mat : numpy ndarray, optional
            a positive definate matrix, by default None.
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

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str: float or numpy ndarray}
            * ``"w_vec"`` : The value of ``self.w_vec``.
        """
        return {'w_vec':self.w_vec}

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float or numpy ndarray}
            * ``"h_mu_vec"`` : The value of ``self.h_mu_vec``
            * ``"h_lambda_mat"`` : The value of ``self.h_lambda_mat``
        """
        return {'h_mu_vec':self.h_mu_vec,'h_lambda_mat':self.h_lambda_mat}

    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        The generated vaule is set at ``self.w_vec``.
        """
        self.w_vec = self.rng.multivariate_normal(mean=self.h_mu_vec,cov=np.linalg.inv(self.h_lambda_mat))
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
            float array whose shape is ``(sample_size,c_degree)``.
        y : numpy ndarray
            1 dimensional int array whose size is ``sample_size``.
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
        
        y = np.empty(sample_size,dtype=int)
        for i in range(sample_size):
            prob = expit(x[i] @ self.w_vec)
            y[i] = self.rng.choice([0, 1], p=[1 - prob, prob])

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
        x, y = self.gen_sample(sample_size, x, constant)
        np.savez_compressed(filename, x=x, y=y)

    def visualize_model(self,sample_size=100,constant=True):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 100
        constant : bool, optional
            A boolean value, by default ``True``.

        Examples
        --------
        >>> import numpy as np
        >>> from bayesml import logisticregression
        >>> model = logisticregression.GenModel(c_degree=2,w_vec=np.array([7,3]))
        >>> model.visualize_model()
        w_vec:
        [7. 3.]

        .. image:: ./images/logisticregression_example.png
        """
        print(f"w_vec:\n{self.w_vec}")
        if self.c_degree == 2 and constant==True:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_y = self.gen_sample(sample_size=sample_size,constant=True)
            fig, ax = plt.subplots()

            x = np.linspace(sample_x[:,0].min(),sample_x[:,0].max(),1000)
            ax.plot(x, expit(x*self.w_vec[0] + self.w_vec[1]),label=rf'$\sigma$({self.w_vec[0]:.2f}*x + {self.w_vec[1]:.2f})',c='black')

            negative = (sample_y==0)
            positive = (sample_y==1)
            ax.scatter(sample_x[negative,0],sample_y[negative],c='blue')
            ax.scatter(sample_x[positive,0],sample_y[positive],c='red')

            ax.set_ylim([-0.1,1.1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        elif self.c_degree == 1 and constant==False:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_y = self.gen_sample(sample_size=sample_size,constant=False)
            fig, ax = plt.subplots()

            x = np.linspace(sample_x.min(),sample_x.max(),1000)
            ax.plot(x, expit(x*self.w_vec),label=rf'$\sigma$({self.w_vec[0]:.2f}*x)',c='black')

            negative = (sample_y==0)
            positive = (sample_y==1)
            ax.scatter(sample_x[negative,0],sample_y[negative],c='blue')
            ax.scatter(sample_x[positive,0],sample_y[positive],c='red')

            ax.set_ylim([-0.1,1.1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        elif self.c_degree == 3 and constant==True:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_y = self.gen_sample(sample_size=sample_size,constant=True)
            fig, ax = plt.subplots()

            x = np.linspace(sample_x[:,0].min()-(sample_x[:,0].max()-sample_x[:,0].min())*0.1,
                            sample_x[:,0].max()+(sample_x[:,0].max()-sample_x[:,0].min())*0.1,
                            1000)
            y = np.linspace(sample_x[:,1].min()-(sample_x[:,1].max()-sample_x[:,1].min())*0.1,
                            sample_x[:,1].max()+(sample_x[:,1].max()-sample_x[:,1].min())*0.1,
                            1000)
            xx, yy = np.meshgrid(x,y)
            grid = np.empty((1000,1000,3))
            grid[:,:,0] = xx
            grid[:,:,1] = yy
            grid[:,:,2] = 1.0
            ax.imshow(
                expit(grid @ self.w_vec),
                extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                origin='lower',
                cmap='bwr',
                alpha=0.5,
                vmin=0,
                vmax=1,
                aspect='auto',
            )

            negative = (sample_y==0)
            positive = (sample_y==1)
            ax.scatter(sample_x[negative,0],sample_x[negative,1],color='blue')
            ax.scatter(sample_x[positive,0],sample_x[positive,1],color='red')

            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            plt.show()
        elif self.c_degree == 2 and constant==False:
            _check.pos_int(sample_size,'sample_size',DataFormatError)
            sample_x, sample_y = self.gen_sample(sample_size=sample_size,constant=False)
            fig, ax = plt.subplots()

            x = np.linspace(sample_x[:,0].min()-(sample_x[:,0].max()-sample_x[:,0].min())*0.1,
                            sample_x[:,0].max()+(sample_x[:,0].max()-sample_x[:,0].min())*0.1,
                            1000)
            y = np.linspace(sample_x[:,1].min()-(sample_x[:,1].max()-sample_x[:,1].min())*0.1,
                            sample_x[:,1].max()+(sample_x[:,1].max()-sample_x[:,1].min())*0.1,
                            1000)
            xx, yy = np.meshgrid(x,y)
            grid = np.empty((1000,1000,2))
            grid[:,:,0] = xx
            grid[:,:,1] = yy
            ax.imshow(
                expit(grid @ self.w_vec),
                extent=[xx.min(), xx.max(), yy.min(), yy.max()],
                origin='lower',
                cmap='bwr',
                alpha=0.5,
                vmin=0,
                vmax=1,
                aspect='auto',
            )

            negative = (sample_y==0)
            positive = (sample_y==1)
            ax.scatter(sample_x[negative,0],sample_x[negative,1],color='blue')
            ax.scatter(sample_x[positive,0],sample_x[positive,1],color='red')

            ax.set_xlabel("x[0]")
            ax.set_ylabel("x[1]")
            plt.show()
        else:
            raise(ParameterFormatError(
                "This function supports only the following cases: "
                + "c_degree = 2 and constant = True; "
                + "c_degree = 1 and constant = False; "
                + "c_degree = 3 and constant = True; "
                + "c_degree = 2 and constant = False."))

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

    Attributes
    ----------
    hn_mu_vec : numpy ndarray
        a vector of real numbers
    hn_lambda_mat : numpy ndarray
        a positive definate matrix
    hn_lambda_mat_inv : numpy ndarray
        a positive definate matrix
    xis : numpy ndarray
        real numbers
    vl : float
        real number
    p_sigmas_sq : numpy ndarray
        positive real numbers
    p_mus : numpy ndarray
        real numbers
    """
    def __init__(
            self,
            c_degree,
            *,
            h0_mu_vec=None,
            h0_lambda_mat=None,
            seed = None
            ):
        # constants
        self.c_degree = _check.pos_int(c_degree,'c_degree',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h0_params
        self.h0_mu_vec = np.zeros([self.c_degree])
        self.h0_lambda_mat = np.identity(self.c_degree)

        self._ln_det_h0_lambda_mat = 0.0
        self._h0_lambda_mu_vec = np.empty(self.c_degree)
        self._h0_mu_lambda_mu = 0.0

        # hn_params
        self.hn_mu_vec = np.empty(self.c_degree)
        self.hn_lambda_mat = np.empty([self.c_degree,self.c_degree])
        self.hn_lambda_mat_inv = np.empty([self.c_degree,self.c_degree])

        self.xis = None
        self._xis_sq = None
        self._sigma_xis = None
        self._lambda_xis = None

        # statistics
        self._y_x_vec = np.empty(self.c_degree)

        # p_params
        self.p_sigmas_sq = 0.0
        self.p_mus = 0.0
        
        # variational lower bound
        self.vl = 0.0

        self.set_h0_params(
            h0_mu_vec,
            h0_lambda_mat,
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
            ):
        """Set initial values of the hyperparameter of the posterior distribution.

        Note that the parameters of the predictive distribution are also calculated from 
        ``self.h0_mu_vec`` and ``self.h0_lambda_mat``.

        Parameters
        ----------
        h0_mu_vec : numpy ndarray, optional
            a vector of real numbers, by default None.
        h0_lambda_mat : numpy ndarray, optional
            a positive definate matrix, by default None.
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
        
        self._calc_prior_features()
        self.reset_hn_params()

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the posterior distribution.

        Returns
        -------
        h0_params : dict of {str: float or numpy ndarray}
            * ``"h0_mu_vec"`` : The value of ``self.h0_mu_vec``
            * ``"h0_lambda_mat"`` : The value of ``self.h0_lambda_mat``
        """
        return {'h0_mu_vec':self.h0_mu_vec,'h0_lambda_mat':self.h0_lambda_mat}
    
    def set_hn_params(
            self,
            hn_mu_vec=None,
            hn_lambda_mat=None,
            ):
        """Set updated values of the hyperparameter of the posterior distribution.

        Note that the parameters of the predictive distribution are also calculated from 
        ``self.hn_mu_vec`` and ``self.hn_lambda_mat``.

        Parameters
        ----------
        hn_mu_vec : numpy ndarray, optional
            a vector of real numbers, by default None.
        hn_lambda_mat : numpy ndarray, optional
            a positive definate matrix, by default None.
        """
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
            self.hn_lambda_mat_inv[:] = np.linalg.inv(self.hn_lambda_mat)

        self.calc_pred_dist(np.zeros(self.c_degree))

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float or numpy ndarray}
            * ``"hn_mu_vec"`` : The value of ``self.hn_mu_vec``
            * ``"hn_lambda_mat"`` : The value of ``self.hn_lambda_mat``
        """
        return {'hn_mu_vec':self.hn_mu_vec,'hn_lambda_mat':self.hn_lambda_mat}
    
    def _check_sample_x(self,x):
        _check.float_vecs(x,'x',DataFormatError)
        if x.shape[-1] != self.c_degree:
            raise(DataFormatError(f"x.shape[-1] must be c_degree:{self.c_degree}"))
        return x.reshape(-1,self.c_degree)
    
    def _check_sample_y(self,y):
        return _check.ints_of_01(y,'y',DataFormatError)

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
            max_itr=1000,
            tolerance=1.0E-8,
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
            maximum number of iterations, by default 1000
        tolerance : float, optional
            convergence criterion of variational lower bound, by default 1.0E-8
        """
        x,y = self._check_sample(x,y)
        self._y_x_vec[:] = (y-0.5) @ x

        # initialization
        self.reset_hn_params()
        self._init_xi(x)
        self._update_q_w(x)
        self._calc_vl()
        print(f'VL: {self.vl}',end='')

        convergence_flag = True
        for t in range(max_itr):
            vl_before = self.vl
            self._update_xi(x)
            self._update_q_w(x)
            self._calc_vl()
            print(f'\rVL: {self.vl} t={t} ',end='')
            if np.abs((self.vl-vl_before)/vl_before) < tolerance:
                convergence_flag = False
                print(f'(converged)')
                break
        
        if convergence_flag:
            print('(not converged)')
            warnings.warn("Algorithm has not converged.",ResultWarning)
        
        return self

    def estimate_params(self,loss="squared"):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".

        Returns
        -------
        w_vec_hat : numpy ndarray or rv_frozen
            The estimated value under the given loss function. If it is not exist, `None` will be returned.
            If the loss function is \"KL\", the posterior distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
        if loss == "squared" or loss == "0-1" or loss == "abs":
            return  self.hn_mu_vec
        elif loss == "KL":
            return ss_multivariate_normal(mean=self.hn_mu_vec,
                                          cov=np.linalg.inv(self.hn_lambda_mat))
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import logisticregression
        >>> gen_model = logisticregression.GenModel(c_degree=2,w_vec=np.array([7,3]))
        >>> x,y = gen_model.gen_sample(sample_size=200)
        >>> learn_model = logisticregression.LearnModel(c_degree=2)
        >>> learn_model.update_posterior(x,y)
        >>> learn_model.visualize_posterior()
        hn_mu_vec:
        [4.0979368  1.64361963]
        hn_lambda_mat:
        [[18.12098962 -4.70969658]
        [-4.70969658 32.42845112]]

        .. image:: ./images/logisticregression_posterior.png
        """
        print("hn_mu_vec:")
        print(f"{self.hn_mu_vec}")
        print("hn_lambda_mat:")
        print(f"{self.hn_lambda_mat}")
        w_vec_pdf = self.estimate_params(loss="KL")
        if self.c_degree == 1:
            hn_lambda_inv = np.linalg.inv(self.hn_lambda_mat)
            fig, ax = plt.subplots()

            x = np.linspace(self.hn_mu_vec[0]-4.0*np.sqrt(hn_lambda_inv[0,0]),
                            self.hn_mu_vec[0]+4.0*np.sqrt(hn_lambda_inv[0,0]),
                            100)
            ax.plot(x,w_vec_pdf.pdf(x))
            ax.set_xlabel("w_vec")
            ax.set_ylabel("Density")
            plt.show()
        elif self.c_degree == 2:
            hn_lambda_inv = np.linalg.inv(self.hn_lambda_mat)
            fig, ax = plt.subplots()
            
            #for theta
            x = np.linspace(self.hn_mu_vec[0]-3.0*np.sqrt(hn_lambda_inv[0,0]),
                            self.hn_mu_vec[0]+3.0*np.sqrt(hn_lambda_inv[0,0]),
                            100)
            y = np.linspace(self.hn_mu_vec[1]-3.0*np.sqrt(hn_lambda_inv[1,1]),
                            self.hn_mu_vec[1]+3.0*np.sqrt(hn_lambda_inv[1,1]),
                            100)
            xx, yy = np.meshgrid(x,y)
            grid = np.empty((100,100,2))
            grid[:,:,0] = xx
            grid[:,:,1] = yy
            ax.contourf(xx,yy,w_vec_pdf.pdf(grid))
            ax.plot(self.hn_mu_vec[0],self.hn_mu_vec[1],marker="x",color="red")
            ax.set_xlabel("w_vec[0]")
            ax.set_ylabel("w_vec[1]")
            plt.show()
        else:
            raise(ParameterFormatError("if self.c_degree > 2, it is impossible to visualize posterior by this function."))
        
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy ndarray}
            * ``"p_sigmas_sq"`` : The value of ``self.p_sigmas_sq``
            * ``"p_mus"`` : The value of ``self.p_mus``
        """
        return {'p_sigmas_sq':self.p_sigmas_sq,'p_mus':self.p_mus}

    def calc_pred_dist(self,x):
        """Calculate the parameters of the predictive distribution.

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        """
        x = self._check_sample_x(x)
        self.p_sigmas_sq = np.einsum('ij,jk,ik->i',x,self.hn_lambda_mat_inv,x)
        self.p_mus = x @ self.hn_mu_vec

    def make_prediction(self,loss="0-1"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"0-1\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".

        Returns
        -------
        Predicted_values : numpy ndarray
            The predicted values under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        probs = expit(
            self.p_mus / np.sqrt(1 + (np.pi * self.p_sigmas_sq) / 8)
        )
        if loss == "squared":
            return probs
        elif loss == "0-1" or loss == "abs":
            pred_values = np.ones(probs.shape[0],dtype=int)
            pred_values[probs <= 0.5] = 0
            return pred_values
        elif loss == "KL":
            pred_dists = np.empty((probs.shape[0],2))
            pred_dists[:,0] = 1.0 - probs
            pred_dists[:,1] = probs
            return pred_dists
        else:
            raise(CriteriaError("Unsupported loss function! "
                                "This function supports \"squared\", \"0-1\", \"abs\", and \"KL\"."))

    def pred_and_update(
            self,
            x,
            y,
            loss="0-1",
            max_itr=1000,
            tolerance=1.0E-8,
            ):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        y : numpy ndarray
            float array.
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"0-1\".
            This function supports \"squared\", \"0-1\", \"abs\", and \"KL\".
        max_itr : int, optional
            maximum number of iterations, by default 1000
        tolerance : float, optional
            convergence criterion of variational lower bound, by default 1.0E-8
        
        Returns
        -------
        Predicted_values : numpy ndarray
            The predicted values under the given loss function. 
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as numpy.ndarray.
        """
        self.calc_pred_dist(x)
        prediction = self.make_prediction(loss=loss)
        self.overwrite_h0_params()
        self.update_posterior(
            x,
            y,
            max_itr=max_itr,
            tolerance=tolerance,
            )
        return prediction

    def _calc_prior_features(self):
        self._ln_det_h0_lambda_mat = np.linalg.slogdet(self.h0_lambda_mat)[1]
        self._h0_lambda_mu_vec[:] = self.h0_lambda_mat @ self.h0_mu_vec
        self._h0_mu_lambda_mu = self.h0_mu_vec @ self._h0_lambda_mu_vec

    def _init_xi(self,x):
        n = x.shape[0]
        self.xis = np.ones(n)
        self._xis_sq = np.ones(n)
        self._sigma_xis = expit(self.xis)
        self._lambda_xis = (self._sigma_xis - 0.5) / (2.0 * self.xis)

    def _calc_xi_features(self):
        self._sigma_xis[:] = expit(self.xis)
        self._lambda_xis[:] = (self._sigma_xis - 0.5) / (2.0 * self.xis)

    def _calc_vl(self):
        # This bound is tight but valid only right after updating q(w).
        self.vl = (
            0.5 * (
                -np.linalg.slogdet(self.hn_lambda_mat)[1]
                + self._ln_det_h0_lambda_mat
                + self.hn_mu_vec @ self.hn_lambda_mat @ self.hn_mu_vec
                - self._h0_mu_lambda_mu
            ) + (
                np.log(self._sigma_xis)
                - 0.5 * self.xis
                + self._lambda_xis * self._xis_sq
            ).sum()
        )

    def _calc_vl_for_test(self,x):
        # This bound monotonically increases for both updates of xi and q(w), but it is not tight.
        _vl_h_w_xi = (
            np.log(self._sigma_xis)
            - 0.5 * self.xis
            - self._lambda_xis * (
                np.einsum(
                    'ij,jk,ik->i',
                    x,
                    self.hn_lambda_mat_inv + np.outer(self.hn_mu_vec,self.hn_mu_vec),
                    x,
                )
                - self._xis_sq
            )
        ).sum() + self.hn_mu_vec @ self._y_x_vec

        diff = self.hn_mu_vec - self.h0_mu_vec
        _vl_p_w = (
            -self.c_degree*np.log(2*np.pi)
            + self._ln_det_h0_lambda_mat
            - diff @ self.h0_lambda_mat @ diff
            - (self.h0_lambda_mat * self.hn_lambda_mat_inv).sum()
        ) / 2.0

        _vl_q_w = (
            self.c_degree * (np.log(2*np.pi) + 1)
            - np.linalg.slogdet(self.hn_lambda_mat)[1]
        ) / 2.0

        self.vl = _vl_h_w_xi + _vl_p_w + _vl_q_w

    def _update_q_w(self,x):
        self.hn_lambda_mat[:] = self.h0_lambda_mat + 2.0 * (self._lambda_xis * x.T) @ x
        self.hn_lambda_mat_inv[:] = np.linalg.inv(self.hn_lambda_mat)
        self.hn_mu_vec[:] = np.linalg.solve(
            self.hn_lambda_mat,
            self._h0_lambda_mu_vec + self._y_x_vec
        )

    def _update_xi(self,x):
        self._xis_sq[:] = np.einsum(
            'ij,jk,ik->i',
            x,
            self.hn_lambda_mat_inv + np.outer(self.hn_mu_vec,self.hn_mu_vec),
            x,
        )
        self.xis[:] = np.sqrt(self._xis_sq)
        self._calc_xi_features()

    def fit(
            self,
            x,
            y,
            max_itr=1000,
            tolerance=1.0E-8,
            ):            
        """Fit the model to the data.

        This function is a wrapper of the following functions:

        >>> self.reset_hn_params()
        >>> self.update_posterior(x,y,max_itr,tolerance)
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
        tolerance : float, optional
            convergence criterion of variational lower bound, by default 1.0E-8
        
        Returns
        -------
        self : LearnModel
            The fitted model.
        """
        self.reset_hn_params()
        self.update_posterior(x,y,max_itr,tolerance)
        return self
    
    def predict(self,x):
        """Predict the data.

        This function is a wrapper of the following functions:

        >>> self.calc_pred_dist(x)
        >>> return self.make_prediction(loss="0-1")

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        
        Returns
        -------
        predicted_values : numpy.ndarray
            The predicted values under the 0-1 loss function. 
            The size of the predicted values is the same as the sample size of x.
        """
        self.calc_pred_dist(x)
        return self.make_prediction(loss="0-1")

    def predict_proba(self,x):
        """Predict the data.

        This function is a wrapper of the following functions:

        >>> self.calc_pred_dist(x)
        >>> return self.make_prediction(loss="KL")

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        
        Returns
        -------
        predicted_values : numpy.ndarray
            The predicted values under the 0-1 loss function. 
            The size of the predicted values is the same as the sample size of x.
        """
        self.calc_pred_dist(x)
        return self.make_prediction(loss="KL")
