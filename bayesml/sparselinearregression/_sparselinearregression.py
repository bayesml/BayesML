# Code Author
# Kohei Horinouchi <horinouchi@aoni.waseda.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Kohei Horinouchi <horinouchi@aoni.waseda.jp>
import warnings
import numpy as np
from scipy.stats import gamma as ss_gamma
from scipy.stats import multivariate_t as ss_multivariate_t
from scipy.stats import t as ss_t
from scipy.special import gammaln, digamma
import matplotlib.pyplot as plt

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning
from .. import _check

_LOG_2PI = np.log(2.0*np.pi)

class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution.

    Parameters
    ----------
    c_degree : int
        A positive integer. The dimension of the explanatory variable.
        If you consider an intercept term, it should be included as one of the
        elements of the explanatory variable.
    theta_vec : numpy ndarray, optional
        A vector of real numbers, by default ``[0.0, ..., 0.0]``.
    tau : float, optional
        A positive real number, by default ``1.0``.
    h_lambdas : float or numpy.ndarray, optional
        Positive real numbers, (Laplace regularization strength),
        by default ``[1.0, 1.0, ... , 1.0]``.
        If a single real number is input, it will be broadcasted.
    h_alpha : float, optional
        A positive real number, by default ``1.0``.
    h_beta : float, optional
        A positive real number, by default ``1.0``.
    seed : {None, int}, optional
        A seed to initialize ``numpy.random.default_rng()``,
        by default ``None``.
    """

    def __init__(
            self,
            c_degree,
            theta_vec=None,
            tau=1.0,
            h_lambdas=None,
            h_alpha=1.0,
            h_beta=1.0,
            seed=None):
        # constants
        self.c_degree = _check.pos_int(c_degree, 'c_degree', ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # params
        self.theta_vec = np.zeros(self.c_degree)
        self.tau = 1.0
        self.v_vec = np.ones(self.c_degree)

        # h_params
        self.h_lambdas = np.ones(self.c_degree)
        self.h_alpha = 1.0
        self.h_beta = 1.0

        self.set_params(theta_vec, tau)
        self.set_h_params(h_lambdas, h_alpha, h_beta)

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int}
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {'c_degree': self.c_degree}

    def set_h_params(self, h_lambdas=None, h_alpha=None, h_beta=None):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_lambdas : float or numpy.ndarray, optional
            Positive real numbers, (Laplace regularization strength),
            by default ``None``.
            If a single real number is input, it will be broadcasted.
        h_alpha : float, optional
            A positive real number, by default ``None``.
        h_beta : float, optional
            A positive real number, by default ``None``.
        """
        if h_lambdas is not None:
            self.h_lambdas[:] = _check.pos_floats(h_lambdas, 'h_lambdas', ParameterFormatError)
        if h_alpha is not None:
            self.h_alpha = _check.pos_float(h_alpha, 'h_alpha', ParameterFormatError)
        if h_beta is not None:
            self.h_beta = _check.pos_float(h_beta, 'h_beta', ParameterFormatError)
        return self

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float}
            * ``"h_lambdas"`` : The value of ``self.h_lambdas``
            * ``"h_alpha"`` : The value of ``self.h_alpha``
            * ``"h_beta"`` : The value of ``self.h_beta``
        """
        return {'h_lambdas': self.h_lambdas, 'h_alpha': self.h_alpha, 'h_beta': self.h_beta}

    def gen_params(self):
        """Generate parameters from the prior distribution.

        The generated values are set at ``self.theta_vec``, ``self.tau``,
        and ``self.v_vec``.
        """
        self.tau = self.rng.gamma(shape=self.h_alpha, scale=1.0 / self.h_beta)
        self.v_vec[:] = self.rng.exponential(
            scale=2.0 / self.h_lambdas ** 2,
        )
        self.theta_vec[:] = self.rng.normal(
            loc=0.0,
            scale=np.sqrt(self.v_vec / self.tau)
        )
        return self

    def set_params(self, theta_vec=None, tau=None):
        """Set the parameters of the stochastic data generative model.

        Parameters
        ----------
        theta_vec : numpy ndarray, optional
            A vector of real numbers, by default ``None``.
        tau : float, optional
            A positive real number, by default ``None``.
        """
        if theta_vec is not None:
            _check.float_vec(theta_vec, 'theta_vec', ParameterFormatError)
            _check.shape_consistency(
                theta_vec.shape[0], 'theta_vec.shape[0]',
                self.c_degree, 'self.c_degree',
                ParameterFormatError
            )
            self.theta_vec[:] = theta_vec

        if tau is not None:
            self.tau = _check.pos_float(tau, 'tau', ParameterFormatError)

        return self

    def get_params(self):
        """Get the parameters of the stochastic data generative model.

        Returns
        -------
        params : dict of {str: float or numpy ndarray}
            * ``"theta_vec"`` : The value of ``self.theta_vec``
            * ``"tau"`` : The value of ``self.tau``
        """
        return {'theta_vec': self.theta_vec, 'tau': self.tau}

    def gen_sample(self, sample_size=None, x=None, constant=True):
        """Generate a sample from the stochastic data generative model.

        If ``x`` is given, it will be used for explanatory variables as it is
        (independent of the other options: ``sample_size`` and ``constant``).

        If ``x`` is not given, it will be generated from i.i.d. standard
        normal distributions. The size of the generated sample is defined by
        ``sample_size``. If ``constant`` is ``True``, the last element of the
        generated explanatory variables will be overwritten by ``1.0``.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default ``None``.
        x : numpy ndarray, optional
            A float array whose shape is ``(sample_size, c_degree)``,
            by default ``None``.
        constant : bool, optional
            A boolean value, by default ``True``.

        Returns
        -------
        x : numpy ndarray
            A float array whose shape is ``(sample_size, c_degree)``.
        y : numpy ndarray
            A 1-dimensional float array whose size is ``sample_size``.
        """
        if x is not None:
            _check.float_vecs(x, 'x', DataFormatError)
            x = x.reshape(-1, self.c_degree)
            sample_size = x.shape[0]
        elif sample_size is not None:
            _check.pos_int(sample_size, 'sample_size', DataFormatError)
            x = self.rng.multivariate_normal(
                np.zeros(self.c_degree),
                np.eye(self.c_degree),
                size=sample_size
            )
            if constant:
                x[:, -1] = 1.0
        else:
            raise DataFormatError(
                "Either of sample_size or x must be given as an input."
            )

        y = self.rng.normal(
            loc=x @ self.theta_vec,
            scale=1.0 / np.sqrt(self.tau)
        )
        return x, y

    def save_sample(self, filename, sample_size=None, x=None, constant=True):
        """Save the generated sample as NumPy ``.npz`` format.

        The generated sample is saved as a NpzFile with keywords ``"x"``
        and ``"y"``.

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it is not there.
        sample_size : int, optional
            A positive integer, by default ``None``.
        x : numpy ndarray, optional
            A float array whose shape is ``(sample_size, c_degree)``,
            by default ``None``.
        constant : bool, optional
            A boolean value, by default ``True``.

        See Also
        --------
        numpy.savez_compressed
        """
        x, y = self.gen_sample(sample_size, x, constant)
        np.savez_compressed(filename, x=x, y=y)

    def visualize_model(self, sample_size=100, constant=True):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default ``100``.
        constant : bool, optional
            A boolean value, by default ``True``.

        Examples
        --------
        >>> import numpy as np
        >>> from bayesml import sparselinearregression
        >>> model = sparselinearregression.GenModel(
        ...     c_degree=2,
        ...     theta_vec=np.array([2.0, 1.0])
        ... )
        >>> model.visualize_model()

        .. image:: ./images/sparselinearregression_example.png
        """
        if self.c_degree == 2 and constant:
            print(f"theta_vec:\n{self.theta_vec}")
            print(f"tau:\n{self.tau}")
            _check.pos_int(sample_size, 'sample_size', DataFormatError)
            sample_x, sample_y = self.gen_sample(
                sample_size=sample_size, constant=True
            )
            fig, ax = plt.subplots()
            ax.scatter(sample_x[:, 0], sample_y)
            x = np.linspace(
                sample_x[:, 0].min() - (sample_x[:, 0].max() - sample_x[:, 0].min()) * 0.25,
                sample_x[:, 0].max() + (sample_x[:, 0].max() - sample_x[:, 0].min()) * 0.25,
                100
            )
            ax.plot(
                x,
                x * self.theta_vec[0] + self.theta_vec[1],
                label=f'y={self.theta_vec[0]:.2f}*x + {self.theta_vec[1]:.2f}',
                c='red'
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        elif self.c_degree == 1 and not constant:
            print(f"theta_vec:\n{self.theta_vec}")
            print(f"tau:\n{self.tau}")
            _check.pos_int(sample_size, 'sample_size', DataFormatError)
            sample_x, sample_y = self.gen_sample(
                sample_size=sample_size, constant=False
            )
            fig, ax = plt.subplots()
            ax.scatter(sample_x[:, 0], sample_y)
            x = np.linspace(
                sample_x[:, 0].min() - (sample_x[:, 0].max() - sample_x[:, 0].min()) * 0.25,
                sample_x[:, 0].max() + (sample_x[:, 0].max() - sample_x[:, 0].min()) * 0.25,
                100
            )
            ax.plot(
                x,
                x * self.theta_vec[0],
                label=f'y={self.theta_vec[0]:.2f}*x',
                c='red'
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.legend()
            plt.show()
        else:
            raise ParameterFormatError(
                "This function supports only the following cases: "
                "c_degree = 2 and constant = True; "
                "c_degree = 1 and constant = False."
            )


class LearnModel(base.Posterior, base.PredictiveMixin):
    r"""The posterior distribution and the predictive distribution.

    Inference is performed by variational Bayes. The variational posterior is

    Parameters
    ----------
    c_degree : int
        A positive integer. The dimension of the explanatory variable.
    h0_lambdas : float, optional
        Positive real numbers, (Laplace regularization strength),
        by default ``[1.0, 1.0, ... , 1.0]``.
        If a single real number is input, it will be broadcasted.
    h0_alpha : float, optional
        A positive real number, by default ``1.0``.
    h0_beta : float, optional
        A positive real number, by default ``1.0``.
    seed : {None, int}, optional
        A seed to initialize ``numpy.random.default_rng()`` used for VB
        random restarts, by default ``None``.

    Attributes
    ----------
    hn_mu_vec : numpy ndarray
        A vector of real numbers.
    hn_lambda_mat : numpy ndarray
        A positive definite matrix.
    hn_alpha : float
        A positive real number.
    hn_beta : float
        A positive real number.
    hn_as : numpy ndarray
        A vector of positive real numbers (:math:`= \lambda_0^2`).
    hn_bs : numpy ndarray
        A vector of positive real numbers.
    p_ms : float
        A real number.
    p_lambdas : float
        A positive real number.
    p_nus : float
        A positive real number.
    vl : float
        The variational lower bound.
    """

    def __init__(
            self,
            c_degree,
            h0_lambdas=None,
            h0_alpha=1.0,
            h0_beta=1.0,
            seed=None):
        # constants
        self.c_degree = _check.pos_int(c_degree, 'c_degree', ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h0_params
        self.h0_lambdas = np.ones(self.c_degree)
        self.h0_alpha = 1.0
        self.h0_beta = 1.0

        # hn_params
        self.hn_mu_vec = np.zeros(c_degree)
        self.hn_lambda_mat = np.eye(c_degree)
        self.hn_alpha = 1.0
        self.hn_beta = 1.0
        self.hn_as = np.ones(c_degree)
        self.hn_bs = np.ones(c_degree)

        # p_params
        self.p_ms = 0.0
        self.p_lambdas = 1.0
        self.p_nus = 2.0

        # VB features (private)
        self._e_tau = 1.0
        self._e_ln_tau = 0.0
        self._hn_lambda_mat_inv = np.eye(c_degree)
        self._e_v_invs = np.ones(c_degree)

        # ELBO and its components
        self.vl = 0.0
        self._vl_p_y = 0.0
        # combined: E[ln p(θ|τ,v)] + E[ln p(v|λ0)] + H[q(v)] after cancellations
        self._vl_p_theta_v_q_v = 0.0
        self._vl_p_tau = 0.0
        self._vl_q_theta_tau = 0.0

        # cached sufficient statistics (set in update_posterior)
        self._n = 0
        self._XX_mat = np.zeros((c_degree, c_degree))
        self._Xy_vec = np.zeros(c_degree)
        self._yy = 0.0

        self.set_h0_params(h0_lambdas, h0_alpha, h0_beta)

    def get_constants(self):
        """Get constants of LearnModel.

        Returns
        -------
        constants : dict of {str: int}
            * ``"c_degree"`` : the value of ``self.c_degree``
        """
        return {'c_degree': self.c_degree}

    def set_h0_params(self, h0_lambdas=None, h0_alpha=None, h0_beta=None):
        """Set initial values of the hyperparameters of the prior distribution.

        Note that ``reset_hn_params()`` is called inside this method.

        Parameters
        ----------
        h0_lambdas : float, optional
            Positive real numbers, (Laplace regularization strength),
            by default ``None``.
            If a single real number is input, it will be broadcasted.
        h0_alpha : float, optional
            A positive real number, by default ``None``.
        h0_beta : float, optional
            A positive real number, by default ``None``.
        """
        if h0_lambdas is not None:
            self.h0_lambdas[:] = _check.pos_floats(h0_lambdas, 'h0_lambdas', ParameterFormatError)
        if h0_alpha is not None:
            self.h0_alpha = _check.pos_float(h0_alpha, 'h0_alpha', ParameterFormatError)
        if h0_beta is not None:
            self.h0_beta = _check.pos_float(h0_beta, 'h0_beta', ParameterFormatError)
        self.reset_hn_params()
        return self

    def get_h0_params(self):
        """Get the initial values of the hyperparameters of the prior distribution.

        Returns
        -------
        h0_params : dict of {str: float}
            * ``"h0_lambdas"`` : The value of ``self.h0_lambdas``
            * ``"h0_alpha"`` : The value of ``self.h0_alpha``
            * ``"h0_beta"`` : The value of ``self.h0_beta``
        """
        return {'h0_lambdas': self.h0_lambdas, 'h0_alpha': self.h0_alpha, 'h0_beta': self.h0_beta}

    def reset_hn_params(self):
        """Reset the hyperparameters of the posterior distribution to initial values.

        Usualy, `hn_params` are reset to the output of `self.get_h0_params()`,
        but the prior distribution and the posterior distribution have different form in this model,
        therefore, they are set to the solution of variational Bayesian updating formula with no data.

        Note that ``calc_pred_dist`` is called with a zero vector inside this method.
        """
        # Solution of updating formula with no data
        self.set_hn_params(
            hn_mu_vec=np.zeros(self.c_degree),
            hn_lambda_mat=np.eye(self.c_degree),
            hn_alpha=self.h0_alpha,
            hn_beta=self.h0_beta,
            hn_as=self.h0_lambdas ** 2,
            hn_bs=1.0 / self.h0_lambdas ** 2,
        )
        return self

    def overwrite_h0_params(self):
        """Overwrite the initial values of the hyperparameters of the prior distribution by the learned values.

        Usualy, `h0_params` are overwritten to the output of `self.get_hn_params()`,
        but the prior distribution and the posterior distribution have different form in this model,
        therefore, `h0_lambdas` are set to minimize KL(GIG(1/2,`hn_as`,`hn_bs`)||Exp(0.5*`h0_lambdas`**2)).

        Note that ``reset_hn_params()`` is called inside this method.
        """
        self.set_h0_params(
            h0_lambdas=np.sqrt((2 * self.hn_as) / (1 + np.sqrt(self.hn_as * self.hn_bs))),
            h0_alpha=self.hn_alpha,
            h0_beta=self.hn_beta
        )
        return self

    def set_hn_params(
            self,
            hn_mu_vec=None,
            hn_lambda_mat=None,
            hn_alpha=None,
            hn_beta=None,
            hn_as=None,
            hn_bs=None):
        """Set updated values of the hyperparameters of the posterior distribution.

        Note that ``calc_pred_dist`` is called with a zero vector inside this method.

        Parameters
        ----------
        hn_mu_vec : numpy ndarray, optional
            A vector of real numbers, by default ``None``.
        hn_lambda_mat : numpy ndarray, optional
            A positive definite matrix, by default ``None``.
        hn_alpha : float, optional
            A positive real number, by default ``None``.
        hn_beta : float, optional
            A positive real number, by default ``None``.
        hn_as : numpy ndarray, optional
            A vector of positive real numbers, by default ``None``.
        hn_bs : numpy ndarray, optional
            A vector of positive real numbers, by default ``None``.
        """
        if hn_mu_vec is not None:
            _check.float_vec(hn_mu_vec, 'hn_mu_vec', ParameterFormatError)
            _check.shape_consistency(
                hn_mu_vec.shape[0], 'hn_mu_vec.shape[0]',
                self.c_degree, 'self.c_degree',
                ParameterFormatError
            )
            self.hn_mu_vec[:] = hn_mu_vec
        if hn_lambda_mat is not None:
            _check.pos_def_sym_mat(hn_lambda_mat, 'hn_lambda_mat', ParameterFormatError)
            _check.shape_consistency(
                hn_lambda_mat.shape[0], 'hn_lambda_mat.shape[0]',
                self.c_degree, 'self.c_degree',
                ParameterFormatError
            )
            self.hn_lambda_mat[:] = hn_lambda_mat
        if hn_alpha is not None:
            self.hn_alpha = _check.pos_float(hn_alpha, 'hn_alpha', ParameterFormatError)
        if hn_beta is not None:
            self.hn_beta = _check.pos_float(hn_beta, 'hn_beta', ParameterFormatError)
        if hn_as is not None:
            _check.pos_floats(hn_as, 'hn_as', ParameterFormatError)
            self.hn_as[:] = hn_as
        if hn_bs is not None:
            _check.pos_floats(hn_bs, 'hn_bs', ParameterFormatError)
            self.hn_bs[:] = hn_bs
        self._calc_q_theta_tau_features()
        self._calc_q_v_features()
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
            * ``"hn_as"`` : The value of ``self.hn_as``
            * ``"hn_bs"`` : The value of ``self.hn_bs``
        """
        return {
            'hn_mu_vec': self.hn_mu_vec,
            'hn_lambda_mat': self.hn_lambda_mat,
            'hn_alpha': self.hn_alpha,
            'hn_beta': self.hn_beta,
            'hn_as': self.hn_as,
            'hn_bs': self.hn_bs,
        }

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

    # ------------------------------------------------------------------ #
    # VB private methods
    # ------------------------------------------------------------------ #

    def _calc_q_theta_tau_features(self):
        self._e_tau = self.hn_alpha / self.hn_beta
        self._e_ln_tau = digamma(self.hn_alpha) - np.log(self.hn_beta)
        self._hn_lambda_mat_inv[:] = np.linalg.inv(self.hn_lambda_mat)

    def _calc_q_v_features(self):
        # E[v_j^{-1}] = sqrt(a_{n,j} / b_{n,j})
        self._e_v_invs[:] = np.sqrt(self.hn_as / self.hn_bs)

    def _update_q_theta_tau(self):
        self.hn_lambda_mat[:] = self._XX_mat + np.diag(self._e_v_invs)
        self.hn_mu_vec[:] = np.linalg.solve(self.hn_lambda_mat, self._Xy_vec)
        self._hn_lambda_mat_inv[:] = np.linalg.inv(self.hn_lambda_mat)
        self.hn_alpha = self.h0_alpha + 0.5 * self._n
        self.hn_beta = self.h0_beta + 0.5 * (
            self._yy - self.hn_mu_vec @ (self.hn_lambda_mat @ self.hn_mu_vec)
        )
        self._calc_q_theta_tau_features()

    def _update_q_v(self):
        # a_{n,j} = lambda_0^2 (constant)
        # b_{n,j} = E[tau] * mu_{n,j}^2 + (Lambda_n^{-1})_{jj}
        self.hn_as[:] = self.h0_lambdas ** 2
        self.hn_bs[:] = (
            self._e_tau * self.hn_mu_vec ** 2
            + np.diag(self._hn_lambda_mat_inv)
        )
        self._calc_q_v_features()

    def _init_q_v_exponential_rv(self):
        # Randomize b_{n,j} to explore different local optima
        self.hn_bs[:] = self.rng.exponential(scale=1.0, size=self.c_degree)
        self._calc_q_v_features()

    def _init_q_v_OLS(self):
        # Using this method, initial mu and lambda are set to the OLS solution
        self.hn_bs[:] = np.inf
        self._calc_q_v_features()

    def _init_q_v_Ridge(self):
        # Using this method, initial mu and lambda are set to the Ridge solution
        # with h0_lambdas as the regularization strength
        self.hn_bs[:] = 1.0
        self._calc_q_v_features()

    def _init_q_v_manual(self, initial_bs):
        self.hn_bs[:] = initial_bs
        self._calc_q_v_features()

    def _calc_vl(self):
        # E[ln p(y | X, theta, tau)]
        self._vl_p_y = (
            0.5 * self._n * self._e_ln_tau
            - 0.5 * self._n * _LOG_2PI
            - 0.5 * (self._e_tau * (
                        self._yy - 2.0 * self.hn_mu_vec @ self._Xy_vec
                        + self.hn_mu_vec @ (self._XX_mat @ self.hn_mu_vec)
                    ) + np.sum(self._XX_mat * self._hn_lambda_mat_inv))
        )

        # Combined: E[ln p(theta|tau,v)] + E[ln p(v|lam0)] + H[q(v)]
        # E[ln v_j] and E[v_j] terms cancel exactly between these three parts.
        # Remaining terms (derived in variational_lower_bound.md):
        #   (d/2) E[ln tau]
        #   - (1/2) sum_j E[v_j^{-1}] (E[tau] mu_j^2 + S_{jj})
        #   + sum_j [ ln(lam0) - ln(2) - (1/2) b_{n,j} E[v_j^{-1}] ]
        self._vl_p_theta_v_q_v = (
            0.5 * self.c_degree * self._e_ln_tau
            - 0.5 * np.sum(
                self._e_v_invs * (
                    self._e_tau * self.hn_mu_vec ** 2
                    + np.diag(self._hn_lambda_mat_inv)
                    + self.hn_bs
                )
            )
            + np.log(self.h0_lambdas).sum() - self.c_degree * np.log(2.0)
        )

        # E[ln p(tau | alpha0, beta0)]
        self._vl_p_tau = (
            self.h0_alpha * np.log(self.h0_beta)
            - gammaln(self.h0_alpha)
            + (self.h0_alpha - 1.0) * self._e_ln_tau
            - self.h0_beta * self._e_tau
        )

        # H[q(theta, tau)] = -E[ln q(theta, tau)]
        self._vl_q_theta_tau = (
            0.5 * self.c_degree * (1.0 + _LOG_2PI - self._e_ln_tau)
            - 0.5 * np.linalg.slogdet(self.hn_lambda_mat)[1]
            + ss_gamma.entropy(a=self.hn_alpha, scale=1.0 / self.hn_beta)
        )

        self.vl = (self._vl_p_y
                   + self._vl_p_theta_v_q_v
                   + self._vl_p_tau
                   + self._vl_q_theta_tau)

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def update_posterior(
            self,
            x,
            y,
            max_itr=100,
            num_init=10,
            tolerance=1.0e-8,
            init_type='Ridge',
            initial_bs=None,
            warm_start=False):
        """Update the hyperparameters of the posterior distribution using training data.

        Parameters
        ----------
        x : numpy ndarray
            A float array whose shape is ``(sample_size, c_degree)``.
            If you want to use a constant term, it should be included in ``x``.
        y : numpy ndarray
            A 1-dimensional float array whose size is ``sample_size``.
        max_itr : int, optional
            Maximum number of VB iterations per initialization, by default ``100``.
        num_init : int, optional
            Number of random restarts, by default ``10``.
            If ``init_type`` is ``'Ridge'``, ``'OLS'``, or ``'manual'`` or ``warm_start`` is ``True`` this argument is ignored.
        tolerance : float, optional
            Convergence threshold on relative change of VL, by default ``1.0e-8``.
        init_type : str, optional
            Initialization method for the variational Bayesian method, by default ``'Ridge'``.
            This function supports the following values:
            * ``'Ridge'`` : Initialize hn_mu_vec and hn_lambda_mat to the Ridge solution with h0_lambdas as the regularization strength.
            * ``'OLS'`` : Initialize hn_mu_vec and hn_lambda_mat to the OLS solution.
            * ``'exponential_rv'`` : Randomly initialize hn_bs from an exponential distribution with scale 1.0.
            * ``'manual'`` : Manually set initial values of hn_bs. In this case, the argument ``initial_bs`` must be given.
        initial_bs : numpy ndarray, optional
            A vector of positive real numbers, by default ``None``.
            If ``init_type`` is ``'manual'``, this argument must be given.
        warm_start : bool, optional
            If True, use the sufficient statistics of past samples and 
            the current posterior as the starting point for the next update.
        """
        x, y = self._check_sample(x, y)
        if warm_start:
            self._n += x.shape[0]
            self._XX_mat[:] += x.T @ x
            self._Xy_vec[:] += x.T @ y
            self._yy += y @ y
        else:
            self._n = x.shape[0]
            self._XX_mat[:] = x.T @ x
            self._Xy_vec[:] = x.T @ y
            self._yy = y @ y

        tmp_vl = 0.0
        tmp_mu_vec = np.zeros(self.c_degree)
        tmp_lambda_mat = np.eye(self.c_degree)
        tmp_alpha = self.h0_alpha + self._n / 2.0
        tmp_beta = self.h0_beta
        tmp_as = self.h0_lambdas ** 2
        tmp_bs = np.ones(self.c_degree)

        convergence_flag = True
        for i in range(num_init):
            if not warm_start:
                self.reset_hn_params()
                if init_type == 'Ridge':
                    self._init_q_v_Ridge()
                elif init_type == 'OLS':
                    self._init_q_v_OLS()
                elif init_type == 'exponential_rv':
                    self._init_q_v_exponential_rv()
                elif init_type == 'manual':
                    if initial_bs is None:
                        raise ParameterFormatError(
                            "initial_bs must be given when init_type is 'manual'."
                        )
                    self._init_q_v_manual(initial_bs)
                else:
                    raise ParameterFormatError(
                        "Unsupported init_type! "
                        "This function supports 'Ridge', 'OLS', 'exponential_rv', and 'manual'."
                    )
            self._calc_vl()
            print(f'\r{i}. VL: {self.vl}', end='')
            for t in range(max_itr):
                vl_before = self.vl
                self._update_q_theta_tau()
                self._update_q_v()
                self._calc_vl()
                print(f'\r{i}. VL: {self.vl} t={t} ', end='')
                if np.abs((self.vl - vl_before) / (np.abs(vl_before) + 1.0e-300)) < tolerance:
                    convergence_flag = False
                    print('(converged)', end='')
                    break
            if i == 0 or self.vl > tmp_vl:
                print('*')
                tmp_vl = self.vl
                tmp_mu_vec[:] = self.hn_mu_vec
                tmp_lambda_mat[:] = self.hn_lambda_mat
                tmp_alpha = self.hn_alpha
                tmp_beta = self.hn_beta
                tmp_as[:] = self.hn_as
                tmp_bs[:] = self.hn_bs
            else:
                print('')

            if warm_start or init_type in ('Ridge', 'OLS', 'manual'):
                break

        if convergence_flag:
            warnings.warn("Algorithm has not converged even once.", ResultWarning)

        self.hn_mu_vec[:] = tmp_mu_vec
        self.hn_lambda_mat[:] = tmp_lambda_mat
        self.hn_alpha = tmp_alpha
        self.hn_beta = tmp_beta
        self.hn_as[:] = tmp_as
        self.hn_bs[:] = tmp_bs
        self._calc_q_theta_tau_features()
        self._calc_q_v_features()
        return self

    def estimate_params(self, loss='squared', dict_out=False):
        """Estimate the parameter of the stochastic data generative model under the given criterion.

        Note that the criterion is applied to estimating ``theta_vec`` and ``tau`` independently.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default ``"squared"``.
            This function supports ``"squared"``, ``"0-1"``, ``"abs"``, and ``"KL"``.
        dict_out : bool, optional
            If ``True``, output will be a dict, by default ``False``.

        Returns
        -------
        estimates : tuple of {numpy ndarray, float, None, or rv_frozen}
            * ``theta_vec`` : the estimate for theta
            * ``tau_hat`` : the estimate for tau
            The estimated values under the given loss function. If it does not exist,
            ``None`` will be returned. If the loss is ``"KL"``, the approximate posterior
            distribution itself will be returned as ``rv_frozen`` objects of ``scipy.stats``.

        See Also
        --------
        scipy.stats.rv_continuous
        """
        if loss == 'squared':
            if dict_out:
                return {'theta_vec': self.hn_mu_vec, 'tau': self.hn_alpha / self.hn_beta}
            return self.hn_mu_vec, self.hn_alpha / self.hn_beta
        elif loss == '0-1':
            tau_mode = max((self.hn_alpha - 1.0) / self.hn_beta, 0.0)
            if dict_out:
                return {'theta_vec': self.hn_mu_vec, 'tau': tau_mode}
            return self.hn_mu_vec, tau_mode
        elif loss == 'abs':
            tau_med = ss_gamma.median(a=self.hn_alpha, scale=1.0 / self.hn_beta)
            if dict_out:
                return {'theta_vec': self.hn_mu_vec, 'tau': tau_med}
            return self.hn_mu_vec, tau_med
        elif loss == 'KL':
            return (
                ss_multivariate_t(
                    loc=self.hn_mu_vec,
                    shape=np.linalg.inv(self.hn_alpha / self.hn_beta * self.hn_lambda_mat),
                    df=2.0 * self.hn_alpha
                ),
                ss_gamma(a=self.hn_alpha, scale=1.0 / self.hn_beta)
            )
        else:
            raise CriteriaError(
                'Unsupported loss function! '
                'This function supports "squared", "0-1", "abs", and "KL".'
            )

    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.

        Examples
        --------
        >>> import numpy as np
        >>> from bayesml import sparselinearregression
        >>> gen_model = sparselinearregression.GenModel(
        ...     c_degree=2, theta_vec=np.array([1.0, 0.5]), tau=2.0, seed=0
        ... )
        >>> x, y = gen_model.gen_sample(sample_size=50)
        >>> learn_model = sparselinearregression.LearnModel(c_degree=2)
        >>> learn_model.update_posterior(x, y)
        >>> learn_model.visualize_posterior()

        .. image:: ./images/sparselinearregression_posterior.png
        """
        theta_dist, tau_dist = self.estimate_params(loss='KL')
        hn_lambda_inv = self._hn_lambda_mat_inv

        if self.c_degree == 1:
            fig, axes = plt.subplots(1, 2)
            x = np.linspace(
                self.hn_mu_vec[0] - 4.0 * np.sqrt(self.hn_beta / self.hn_alpha * self._hn_lambda_mat_inv[0, 0]),
                self.hn_mu_vec[0] + 4.0 * np.sqrt(self.hn_beta / self.hn_alpha * self._hn_lambda_mat_inv[0, 0]),
                100
            )
            axes[0].plot(x, theta_dist.pdf(x))
            axes[0].set_xlabel("theta_vec[0]")
            axes[0].set_ylabel("Density")
            x = np.linspace(
                max(1.0e-8, self.hn_alpha / self.hn_beta - 4.0 * np.sqrt(self.hn_alpha) / self.hn_beta),
                self.hn_alpha / self.hn_beta + 4.0 * np.sqrt(self.hn_alpha) / self.hn_beta,
                100
            )
            axes[1].plot(x, tau_dist.pdf(x))
            axes[1].set_xlabel("tau")
            axes[1].set_ylabel("Density")
            fig.tight_layout()
            plt.show()
        elif self.c_degree == 2:
            fig, axes = plt.subplots(1, 2)
            x = np.linspace(
                self.hn_mu_vec[0] - 3.0 * np.sqrt(self.hn_beta / self.hn_alpha * self._hn_lambda_mat_inv[0, 0]),
                self.hn_mu_vec[0] + 3.0 * np.sqrt(self.hn_beta / self.hn_alpha * self._hn_lambda_mat_inv[0, 0]),
                100
            )
            y = np.linspace(
                self.hn_mu_vec[1] - 3.0 * np.sqrt(self.hn_beta / self.hn_alpha * self._hn_lambda_mat_inv[1, 1]),
                self.hn_mu_vec[1] + 3.0 * np.sqrt(self.hn_beta / self.hn_alpha * self._hn_lambda_mat_inv[1, 1]),
                100
            )
            xx, yy = np.meshgrid(x, y)
            grid = np.stack([xx, yy], axis=-1)
            axes[0].contourf(xx, yy, theta_dist.pdf(grid))
            axes[0].plot(self.hn_mu_vec[0], self.hn_mu_vec[1], marker='x', color='red')
            axes[0].set_xlabel("theta_vec[0]")
            axes[0].set_ylabel("theta_vec[1]")
            x = np.linspace(
                max(1.0e-8, self.hn_alpha / self.hn_beta - 4.0 * np.sqrt(self.hn_alpha) / self.hn_beta),
                self.hn_alpha / self.hn_beta + 4.0 * np.sqrt(self.hn_alpha) / self.hn_beta,
                100
            )
            axes[1].plot(x, tau_dist.pdf(x))
            axes[1].set_xlabel("tau")
            axes[1].set_ylabel("Density")
            fig.tight_layout()
            plt.show()
        else:
            raise ParameterFormatError(
                "visualize_posterior supports only c_degree = 1 or 2."
            )

    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: float}
            * ``"p_ms"`` : The value of ``self.p_ms``
            * ``"p_lambdas"`` : The value of ``self.p_lambdas``
            * ``"p_nus"`` : The value of ``self.p_nus``
        """
        return {'p_ms': self.p_ms, 'p_lambdas': self.p_lambdas, 'p_nus': self.p_nus}

    def calc_pred_dist(self, x):
        """Calculate the parameters of the predictive distribution.

        Parameters
        ----------
        x : numpy ndarray
            float array. The size along the last dimension must conincides with the c_degree.
            If you want to use a constant term, it should be included in x.
        """
        x = self._check_sample_x(x)
        self.p_ms = x @ self.hn_mu_vec
        self.p_lambdas = self.hn_alpha / self.hn_beta / (1.0 + np.sum(x.T * np.linalg.solve(self.hn_lambda_mat,x.T),axis=0))
        self.p_nus = np.ones(x.shape[0]) * 2.0 * self.hn_alpha
        return self

    def make_prediction(self, loss='squared'):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default ``"squared"``.
            This function supports ``"squared"``, ``"0-1"``, ``"abs"``, and ``"KL"``.

        Returns
        -------
        Predicted_values : {numpy ndarray, rv_frozen}
            The predicted values under the given loss function. 
            The size of the predicted values is the same as the sample size of x when you called calc_pred_dist(x).
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as rv_frozen object of scipy.stats. The rv_frozen object supports broadcasting.

        See Also
        --------
        scipy.stats.rv_continuous
        """
        if loss in ('squared', '0-1', 'abs'):
            return self.p_ms
        elif loss == 'KL':
            return ss_t(loc=self.p_ms, scale=1.0 / np.sqrt(self.p_lambdas), df=self.p_nus)
        else:
            raise CriteriaError(
                'Unsupported loss function! '
                'This function supports "squared", "0-1", "abs", and "KL".'
            )

    def pred_and_update(
            self,
            x,
            y,
            max_itr=100,
            num_init=10,
            tolerance=1.0e-8,
            loss='squared'):
        """Predict a new data point and update the posterior sequentially.

        Note that ``update_posterior`` is called with ``warm_start=True`` inside this method.

        Parameters
        ----------
        x : numpy ndarray
            A float array whose shape is ``(sample_size, c_degree)``.
            If you want to use a constant term, it should be included in ``x``.
        y : numpy ndarray
            A 1-dimensional float array whose size is ``sample_size``.
        max_itr : int, optional
            Maximum number of VB iterations per initialization, by default ``100``.
        num_init : int, optional
            Number of random restarts, by default ``10``.
        tolerance : float, optional
            Convergence threshold on relative change of VL, by default ``1.0e-8``.
        loss : str, optional
            Loss function underlying the Bayes risk function, by default ``"squared"``.

        Returns
        -------
        Predicted_values : {numpy ndarray, rv_frozen}
            The predicted values under the given loss function. 
            The size of the predicted values is the same as the sample size of x when you called calc_pred_dist(x).
            If the loss function is \"KL\", the predictive distribution itself will be returned
            as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        """
        self.calc_pred_dist(x)
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(
            x,
            y,
            max_itr=max_itr,
            num_init=num_init,
            tolerance=tolerance,
            warm_start=True
        )
        return prediction

    def fit(
            self,
            x,
            y,
            max_itr=1000,
            num_init=10,
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
        num_init : int, optional
            number of initializations, by default 10
        tolerance : float, optional
            convergence criterion of variational lower bound, by default 1.0E-8
        
        Returns
        -------
        self : LearnModel
            The fitted model.
        """
        self.reset_hn_params()
        self.update_posterior(x,y,max_itr,num_init,tolerance)
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
