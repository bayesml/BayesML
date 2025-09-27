# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>

from abc import ABCMeta, abstractmethod
import pickle
from ._exceptions import ParameterFormatError

class Generative(metaclass=ABCMeta):
    @abstractmethod
    def set_h_params(self):
        pass

    @abstractmethod
    def get_h_params(self):
        pass
    
    def save_h_params(self,filename):
        """Save the hyperparameters using python ``pickle`` module.

        They are saved as a python dictionary obtained by ``GenModel.get_h_params()``.

        Parameters
        ----------
        filename : str
            The filename (including a extention like .pkl) to which the hyperparameters are saved.
        
        Warnings
        --------
        The ``pickle`` module is not secure. Only unpickle data you trust.

        See Also
        --------
        pickle
        """
        with open(filename,'wb') as f:
            pickle.dump(self.get_h_params(), f)

    def load_h_params(self,filename):
        """Load the hyperparameters to h_params.

        Parameters
        ----------
        filename : str
            The filename (including a extention like .pkl) to be loaded.
            It must be a pickled python dictionary obtained by
            ``GenModel.save_h_params()``, ``LearnModel.save_h0_params()`` 
            or ``LearnModel.save_hn_params()``.

        Warnings
        --------
        The ``pickle`` module is not secure. Only unpickle data you trust.

        See Also
        --------
        pickle
        """
        with open(filename, 'rb') as f:
            tmp_h_params = pickle.load(f)
        if type(tmp_h_params) is dict:
            self.set_h_params(*tmp_h_params.values())
            return self
        
        raise(ParameterFormatError(
            filename+" must be a pickled python dictionary obtained by "
            +"``GenModel.save_h_params()``, ``LearnModel.save_h0_params()`` "
            +'or ``LearnModel.save_hn_params()``.')
            )

    @abstractmethod
    def gen_params(self):
        pass

    @abstractmethod
    def set_params(self):
        pass

    @abstractmethod
    def get_params(self):
        pass

    def save_params(self,filename):
        """Save the parameters using python ``pickle`` module.

        They are saved as a pickled python dictionary obtained by ``GenModel.get_params()``.

        Parameters
        ----------
        filename : str
            The filename (including a extention like .pkl) to which the parameters are saved.
        
        Warnings
        --------
        The ``pickle`` module is not secure. Only unpickle data you trust.

        See Also
        --------
        pickle
        """
        with open(filename,'wb') as f:
            pickle.dump(self.get_params(), f)

    def load_params(self,filename):
        """Load the parameters saved by ``save_params``.

        Parameters
        ----------
        filename : str
            The filename (including a extention like .pkl) to be loaded.
            It must be a pickled python dictionary obtained by ``GenModel.save_params()``.

        Warnings
        --------
        The ``pickle`` module is not secure. Only unpickle data you trust.

        See Also
        --------
        pickle
        """
        with open(filename, 'rb') as f:
            params = pickle.load(f)
        if type(params) is dict:
            self.set_params(*params.values())
            return self
        
        raise(ParameterFormatError(filename+" must be a pickled python dictionary obtained by ``GenModel.save_params()``"))

    @abstractmethod
    def gen_sample(self):
        pass

    @abstractmethod
    def save_sample(self):
        pass

    @abstractmethod
    def visualize_model(self):
        pass

class Posterior(metaclass=ABCMeta):
    @abstractmethod
    def set_h0_params(self):
        pass

    @abstractmethod
    def get_h0_params(self):
        pass
    
    def save_h0_params(self,filename):
        """Save the hyperparameters using python ``pickle`` module.

        They are saved as a pickled python dictionary obtained by ``LearnModel.get_h0_params()``.

        Parameters
        ----------
        filename : str
            The filename (including a extention like .pkl) to which the hyperparameters are saved.
        
        Warnings
        --------
        The ``pickle`` module is not secure. Only unpickle data you trust.

        See Also
        --------
        pickle
        """
        with open(filename,'wb') as f:
            pickle.dump(self.get_h0_params(), f)

    def load_h0_params(self,filename):
        """Load the hyperparameters to h0_params.

        Parameters
        ----------
        filename : str
            The filename (including a extention like .pkl) to be loaded.
            It must be a pickled python dictionary obtained by
            ``GenModel.save_h_params()``, ``LearnModel.save_h0_params()`` 
            or ``LearnModel.save_hn_params()``.

        Warnings
        --------
        The ``pickle`` module is not secure. Only unpickle data you trust.

        See Also
        --------
        pickle
        """
        with open(filename, 'rb') as f:
            tmp_h_params = pickle.load(f)
        if type(tmp_h_params) is dict:
            self.set_h0_params(*tmp_h_params.values())
            return self
        
        raise(ParameterFormatError(
            filename+" must be a pickled python dictionary obtained by "
            +"``GenModel.save_h_params()``, ``LearnModel.save_h0_params()`` "
            +'or ``LearnModel.save_hn_params()``.')
            )

    @abstractmethod
    def set_hn_params(self):
        pass

    @abstractmethod
    def get_hn_params(self):
        pass
    
    def save_hn_params(self,filename):
        """Save the hyperparameters using python ``pickle`` module.

        They are saved as a pickled python dictionary obtained by ``LearnModel.get_hn_params()``.

        Parameters
        ----------
        filename : str
            The filename (including a extention like .pkl) to which the hyperparameters are saved.
        
        Warnings
        --------
        The ``pickle`` module is not secure. Only unpickle data you trust.

        See Also
        --------
        pickle
        """
        with open(filename,'wb') as f:
            pickle.dump(self.get_hn_params(), f)

    def load_hn_params(self,filename):
        """Load the hyperparameters to hn_params.

        Parameters
        ----------
        filename : str
            The filename (including a extention like .pkl) to be loaded.
            It must be a pickled python dictionary obtained by
            ``GenModel.save_h_params()``, ``LearnModel.save_h0_params()`` 
            or ``LearnModel.save_hn_params()``.

        Warnings
        --------
        The ``pickle`` module is not secure. Only unpickle data you trust.

        See Also
        --------
        pickle
        """
        with open(filename, 'rb') as f:
            tmp_h_params = pickle.load(f)
        if type(tmp_h_params) is dict:
            self.set_hn_params(*tmp_h_params.values())
            return self
        
        raise(ParameterFormatError(
            filename+" must be a pickled python dictionary obtained by "
            +"``GenModel.save_h_params()``, ``LearnModel.save_h0_params()`` "
            +'or ``LearnModel.save_hn_params()``.')
            )

    def reset_hn_params(self):
        """Reset the hyperparameters of the posterior distribution to their initial values.
        
        They are reset to the output of `self.get_h0_params()`.
        Note that the parameters of the predictive distribution are also calculated from them.
        """
        self.set_hn_params(*self.get_h0_params().values())
        return self
    
    def overwrite_h0_params(self):
        """Overwrite the initial values of the hyperparameters of the posterior distribution by the learned values.
        
        They are overwitten by the output of `self.get_hn_params()`.
        Note that the parameters of the predictive distribution are also calculated from them.
        """
        self.set_h0_params(*self.get_hn_params().values())
        return self

    @abstractmethod
    def update_posterior(self):
        pass

    @abstractmethod
    def estimate_params(self):
        pass

    @abstractmethod
    def visualize_posterior(self):
        pass

class PredictiveMixin(metaclass=ABCMeta):
    @abstractmethod
    def get_p_params(self):
        pass
    
    # @abstractmethod
    # def set_p_params(self):
    #     pass

    # def save_p_params(self,filename):
    #     """Save the the parameters of the predictive distribution using python ``pickle`` module.

    #     They are saved as a pickled python dictionary obtained by ``get_p_params()``.

    #     Parameters
    #     ----------
    #     filename : str
    #         The filename (including a extention like .pkl) to which the parameters are saved.
        
    #     Warnings
    #     --------
    #     The ``pickle`` module is not secure. Only unpickle data you trust.

    #     See Also
    #     --------
    #     pickle
    #     """
    #     with open(filename,'wb') as f:
    #         pickle.dump(self.get_p_params(), f)

    # def load_p_params(self,filename):
    #     """Load the parameters of the predictive distribution saved by ``save_p_params`` to p_params.

    #     Parameters
    #     ----------
    #     filename : str
    #         The filename (including a extention like .pkl) to be loaded.
    #         It must be a pickled python dictionary with keys obtained by ``get_p_params().keys()``.

    #     Warnings
    #     --------
    #     The ``pickle`` module is not secure. Only unpickle data you trust.

    #     See Also
    #     --------
    #     pickle
    #     """
    #     with open(filename, 'rb') as f:
    #         p_params = pickle.load(f)
    #     if type(p_params) is dict:
    #         if p_params.keys() == self.get_p_params().keys():
    #             self.set_p_params(**p_params)
    #             return
        
    #     raise(ParameterFormatError(filename+" must be a pickled python dictionary with "+str(self.get_p_params().keys())))

    @abstractmethod
    def calc_pred_dist(self):
        pass

    @abstractmethod
    def make_prediction(self):
        pass
    
    @abstractmethod
    def pred_and_update(self):
        pass
    
    