# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check

_CMAP = plt.get_cmap("Blues")

class _Node:
    def __init__(self,
                 depth,
                 c_k,
                 h_g=0.5,
                 ):
        self.depth = depth
        self.children = [None for i in range(c_k)]  # child nodes
        self.h_g = h_g
        self.h_beta_vec = np.ones(c_k) / 2
        self.theta_vec = np.ones(c_k) / c_k
        self.leaf = False
        self.map_leaf = False

class GenModel(base.Generative):
    """ The stochastice data generative model and the prior distribution

    Parameters
    ----------
    c_k : int
        A positive integer
    c_d_max : int, optional
        A positive integer, by default 10
    root : contexttree._Node, optional
        A root node of a context tree, 
        by default a tree consists of only one node.
    h_g : float, optional
        A real number in :math:`[0, 1]`, by default 0.5
    h_beta_vec : numpy.ndarray, optional
        A vector of positive real numbers, 
        by default [1/2, 1/2, ... , 1/2].
        If a single real number is input, it will be broadcasted.
    h_root : contexttree._Node, optional
        A root node of a superposed tree for hyperparameters 
        by default ``None``
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default ``None``
    """
    def __init__(
            self,
            c_k,
            c_d_max=2,
            root=None,
            h_g=0.5,
            h_beta_vec=None,
            h_root=None,
            seed=None,
            ):
        # constants
        self.c_k = _check.pos_int(c_k,'c_k',ParameterFormatError)
        self.c_d_max = _check.pos_int(c_d_max,'c_d_max',ParameterFormatError)
        self.rng = np.random.default_rng(seed)

        # h_params
        self.h_g = 0.5
        self.h_beta_vec = np.ones(self.c_k) / 2
        self.h_root = None

        self.set_h_params(
            h_g,
            h_beta_vec,
            h_root,
        )

        # params
        self.root = _Node(0,self.c_k,self.h_g)
        self.root.h_beta_vec[:] = self.h_beta_vec
        self.root.leaf = True

        self.set_params(
            root,
        )

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_k"`` : the value of ``self.c_k``
            * ``"c_d_max"`` : the value of ``self.c_d_max``
        """
        return {"c_k":self.c_k, "c_d_max":self.c_d_max}

    def _gen_params_recursion(self,node:_Node,h_node:_Node):
        if h_node is None:
            node.h_g = 0.0 if node.depth == self.c_d_max else self.h_g
            node.h_beta_vec[:] = self.h_beta_vec
            if node.depth == self.c_d_max or self.rng.random() > self.h_g:  # leaf node
                node.theta_vec[:] = self.rng.dirichlet(self.h_beta_vec)
                node.leaf = True
            else:  # inner node
                node.leaf = False
                for i in range(self.c_k):
                    if node.children[i] is None:
                        node.children[i] = _Node(node.depth+1,self.c_k)
                    self._gen_params_recursion(node.children[i],None)
        else:
            node.h_g = 0.0 if node.depth == self.c_d_max else h_node.h_g
            node.h_beta_vec[:] = h_node.h_beta_vec
            if node.depth == self.c_d_max or self.rng.random() > h_node.h_g:  # leaf node
                node.theta_vec[:] = self.rng.dirichlet(h_node.h_beta_vec)
                node.leaf = True
            else:  # inner node
                node.leaf = False
                for i in range(self.c_k):
                    if node.children[i] is None:
                        node.children[i] = _Node(node.depth+1,self.c_k)
                    self._gen_params_recursion(node.children[i],h_node.children[i])

    def _gen_params_recursion_tree_fix(self,node:_Node,h_node:_Node):
        if h_node is None:
            node.h_g = 0.0 if node.depth == self.c_d_max else self.h_g
            node.h_beta_vec[:] = self.h_beta_vec
            if node.leaf:  # leaf node
                node.theta_vec[:] = self.rng.dirichlet(self.h_beta_vec)
            else:  # inner node
                for i in range(self.c_k):
                    if node.children[i] is not None:
                        self._gen_params_recursion_tree_fix(node.children[i],None)
        else:
            node.h_g = 0.0 if node.depth == self.c_d_max else h_node.h_g
            node.h_beta_vec[:] = h_node.h_beta_vec
            if node.leaf:  # leaf node
                node.theta_vec[:] = self.rng.dirichlet(h_node.h_beta_vec)
            else:  # inner node
                for i in range(self.c_k):
                    if node.children[i] is not None:
                        self._gen_params_recursion_tree_fix(node.children[i],h_node.children[i])

    def _set_params_recursion(self,node:_Node,original_tree_node:_Node):
        node.h_g = original_tree_node.h_g
        node.h_beta_vec[:] = original_tree_node.h_beta_vec
        node.theta_vec[:] = original_tree_node.theta_vec
        if node.depth == self.c_d_max:  # leaf node
            node.leaf = True
            node.h_g = 0.0
        elif original_tree_node.leaf:  # leaf node
            node.leaf = True
        else:
            node.leaf = False
            for i in range(self.c_k):
                if node.children[i] is None:
                    node.children[i] = _Node(node.depth+1,self.c_k)
                self._set_params_recursion(node.children[i],original_tree_node.children[i])

    def _set_h_g_recursion(self,node:_Node):
        node.h_g = 0.0 if node.depth == self.c_d_max else self.h_g
        for i in range(self.c_k):
            if node.children[i] is not None:
                self._set_h_g_recursion(node.children[i])

    def _set_h_beta_vec_recursion(self,node:_Node):
        node.h_beta_vec[:] = self.h_beta_vec
        for i in range(self.c_k):
            if node.children[i] is not None:
                self._set_h_beta_vec_recursion(node.children[i])

    def _set_h_params_recursion(self,node:_Node,original_tree_node:_Node):
        if original_tree_node is None:
            node.h_g = 0.0 if node.depth == self.c_d_max else self.h_g
            node.h_beta_vec[:] = self.h_beta_vec
            for i in range(self.c_k):
                if node.children[i] is not None:
                    self._set_h_params_recursion(node.children[i],None)
        else:
            node.h_g = original_tree_node.h_g
            node.h_beta_vec[:] = original_tree_node.h_beta_vec
            if node.depth == self.c_d_max:
                node.leaf = True
                node.h_g = 0.0
            elif original_tree_node.leaf:  # leaf node
                node.leaf = True
            else:
                node.leaf = False
                for i in range(self.c_k):
                    if node.children[i] is None:
                        node.children[i] = _Node(node.depth+1,self.c_k)
                    self._set_h_params_recursion(node.children[i],original_tree_node.children[i])

    def _gen_sample_recursion(self,node,x):
        if node.leaf:  # leaf node
            return self.rng.choice(self.c_k,p=node.theta_vec)
        else:
            return self._gen_sample_recursion(node.children[x[-node.depth-1]],x)
    
    def _visualize_model_recursion(self,tree_graph,node:_Node,node_id,parent_id,sibling_num,p_s):
        tmp_id = node_id
        tmp_p_s = p_s
        
        # add node information
        label_string = f'h_g={node.h_g:.2f}\\lp_s={tmp_p_s:.2f}\\ltheta_vec\\l='
        if node.leaf:
            label_string += f'{np.array2string(node.theta_vec,precision=2,max_line_width=11)}\\l'
        else:
            label_string += 'None\\l'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_s))}')
        if tmp_p_s > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if node.leaf != True:
            for i in range(self.c_k):
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,i,tmp_p_s*node.h_g)
        
        return node_id

    def set_h_params(self,
            h_g=None,
            h_beta_vec=None,
            h_root=None,
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_g : float, optional
            A real number in :math:`[0, 1]`, by default ``None``
        h_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default ``None``
        h_root : contexttree._Node, optional
            A root node of a superposed tree for hyperparameters 
            by default ``None``
        """
        if h_g is not None:
            self.h_g = _check.float_in_closed01(h_g,'h_g',ParameterFormatError)
            if self.h_root is not None:
                self._set_h_g_recursion(self.h_root)

        if h_beta_vec is not None:
            _check.pos_floats(h_beta_vec,'h_beta_vec',ParameterFormatError)
            self.h_beta_vec[:] = h_beta_vec
            if self.h_root is not None:
                self._set_h_beta_vec_recursion(self.h_root)
        
        if h_root is not None:
            if type(h_root) is not _Node:
                raise(ParameterFormatError(
                    "h_root must be an instance of contexttree._Node"
                ))
            if self.h_root is None:
                self.h_root = _Node(0,self.c_k)
            self._set_h_params_recursion(self.h_root,h_root)
        return self

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float, numpy.ndarray, contexttree._Node}
            * ``"h_g"`` : the value of ``self.h_g``
            * ``"h_beta_vec"`` : the value of ``self.h_beta_vec``
            * ``"h_root"`` : the value of ``self.h_root``
        """
        return {"h_g":self.h_g, 
                "h_beta_vec":self.h_beta_vec, 
                "h_root":self.h_root}
            
    def gen_params(self,tree_fix=False):
        """Generate the parameter from the prior distribution.

        The generated vaule is set at ``self.root``.

        Parameters
        ----------
        tree_fix : bool
            If ``True``, tree shape will be fixed, by default ``False``.
        """
        if tree_fix:
            self._gen_params_recursion_tree_fix(self.root,self.h_root)
        else:
            self._gen_params_recursion(self.root,self.h_root)
        return self

    def set_params(self,root=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        root : contexttree._Node, optional
            A root node of a contexttree, by default None.
        """
        if root is not None:
            if type(root) is not _Node:
                raise(ParameterFormatError(
                    "root must be an instance of contexttree._Node"
                ))
            self._set_params_recursion(self.root,root)
        return self

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:float}
            * ``"root"`` : The value of ``self.root``.
        """
        return {"root":self.root}

    def gen_sample(self,sample_length,initial_values=None):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_length : int
            A positive integer
        initial_valules : numpy ndarray, optional
            1 dimensional int array whose size coincide with ``self.c_d_max``,
            by default ``None``. Its elements must be in [0,c_k-1]

        Returns
        -------
        x : numpy ndarray
            1 dimensional int array whose size is ``sammple_length``.
        """
        _check.pos_int(sample_length,'sample_length',DataFormatError)
        x = np.zeros(sample_length+self.c_d_max,dtype=int)
        if initial_values is not None:
            _check.nonneg_ints(initial_values,'initial_values',DataFormatError)
            _check.shape_consistency(
                initial_values.shape[0],'initial_values.shape[0]',
                self.c_d_max,'self.c_d_max',
                DataFormatError
            )
            if initial_values.max() >= self.c_k:
                raise(DataFormatError(f"initial_values.max() must smaller than c_k:{self.c_k}"))
            x[:self.c_d_max] = initial_values
        
        for i in range(self.c_d_max,sample_length+self.c_d_max):
            x[i] = self._gen_sample_recursion(self.root,x[:i])

        return x[self.c_d_max:]
        
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
            1 dimensional int array whose size coincide with ``self.c_d_max``,
            by default ``None``. Its elements must be in [0,c_k-1]
        
        See Also
        --------
        numpy.savez_compressed
        """
        x = self.gen_sample(sample_length,initial_values)
        np.savez_compressed(filename,x)

    def visualize_model(self,filename=None,format=None,sample_length=10):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).
        sample_length : int, optional
            A positive integer, by default 10

        Examples
        --------
        >>> from bayesml import contexttree
        >>> gen_model = contexttree.GenModel(c_k=2,c_d_max=3,h_g=0.75)
        >>> gen_model.gen_params()
        >>> gen_model.visualize_model()
        [1 0 1 0 0 0 1 0 0 0]

        .. image:: ./images/contexttree_example.png

        See Also
        --------
        graphviz.Digraph
        """
        _check.pos_int(sample_length,'sample_length',DataFormatError)

        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            self._visualize_model_recursion(tree_graph, self.root, 0, None, None, 1.0)        
            # Can we show the image on the console without saving the file?
            tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
        x = self.gen_sample(sample_length)
        print(x)

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_k : int
        A positive integer
    c_d_max : int, optional
        A positive integer, by default 10
    h0_g : float, optional
        A real number in :math:`[0, 1]`, by default 0.5
    h0_beta_vec : numpy.ndarray, optional
        A vector of positive real numbers, 
        by default [1/2, 1/2, ... , 1/2].
        If a single real number is input, it will be broadcasted.
    h0_root : contexttree._Node, optional
        A root node of a superposed tree for hyperparameters 
        by default ``None``

    Attributes
    ----------
    hn_g : float
        A real number in :math:`[0, 1]`
    hn_beta_vec : numpy.ndarray
        A vector of positive real numbers.
    hn_root : contexttree._Node
        A root node of a superposed tree for hyperparameters.
    """
    def __init__(
            self,
            c_k,
            c_d_max=2,
            h0_g=0.5,
            h0_beta_vec=None,
            h0_root=None,
            ):
        # constants
        self.c_k = _check.pos_int(c_k,'c_k',ParameterFormatError)
        self.c_d_max = _check.pos_int(c_d_max,'c_d_max',ParameterFormatError)

        # h0_params
        self.h0_g = h0_g
        self.h0_beta_vec = np.ones(self.c_k) / 2
        self.h0_root = None

        # hn_params
        self.hn_g = h0_g
        self.hn_beta_vec = np.ones(self.c_k) / 2
        self.hn_root = None

        # p_params
        self.p_theta_vec = np.ones(self.c_k) / self.c_k

        self.set_h0_params(
            h0_g,
            h0_beta_vec,
            h0_root,
        )

    def get_constants(self):
        """Get constants of LearnModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_k"`` : the value of ``self.c_k``
            * ``"c_d_max"`` : the value of ``self.c_d_max``
        """
        return {"c_k":self.c_k, "c_d_max":self.c_d_max}

    def _set_h0_g_recursion(self,node:_Node):
        node.h_g = 0.0 if node.depth == self.c_d_max else self.h0_g
        for i in range(self.c_k):
            if node.children[i] is not None:
                self._set_h0_g_recursion(node.children[i])

    def _set_h0_beta_vec_recursion(self,node:_Node):
        node.h_beta_vec[:] = self.h0_beta_vec
        for i in range(self.c_k):
            if node.children[i] is not None:
                self._set_h0_beta_vec_recursion(node.children[i])

    def _set_h0_params_recursion(self,node:_Node,original_tree_node:_Node):
        """ copy parameters from a fixed tree

        Parameters
        ----------
        node : object
                a object from _Node class
        original_tree_node : object
                a object from _Node class
        """
        if original_tree_node is None:
            node.h_g = 0.0 if node.depth == self.c_d_max else self.h0_g
            node.h_beta_vec[:] = self.h0_beta_vec
            for i in range(self.c_k):
                if node.children[i] is not None:
                    self._set_h0_params_recursion(node.children[i],None)
        else:
            node.h_g = original_tree_node.h_g
            node.h_beta_vec[:] = original_tree_node.h_beta_vec
            if node.depth == self.c_d_max:
                node.leaf = True
                node.h_g = 0.0
            elif original_tree_node.leaf:  # leaf node
                node.leaf = True
            else:
                node.leaf = False
                for i in range(self.c_k):
                    if node.children[i] is None:
                        node.children[i] = _Node(node.depth+1,self.c_k)
                    self._set_h0_params_recursion(node.children[i],original_tree_node.children[i])

    def _set_hn_g_recursion(self,node:_Node):
        node.h_g = 0.0 if node.depth == self.c_d_max else self.hn_g
        for i in range(self.c_k):
            if node.children[i] is not None:
                self._set_hn_g_recursion(node.children[i])

    def _set_hn_beta_vec_recursion(self,node:_Node):
        node.h_beta_vec[:] = self.hn_beta_vec
        for i in range(self.c_k):
            if node.children[i] is not None:
                self._set_hn_beta_vec_recursion(node.children[i])

    def _set_hn_params_recursion(self,node:_Node,original_tree_node:_Node):
        """ copy parameters from a fixed tree

        Parameters
        ----------
        node : object
                a object from _Node class
        original_tree_node : object
                a object from _Node class
        """
        if original_tree_node is None:
            node.h_g = 0.0 if node.depth == self.c_d_max else self.hn_g
            node.h_beta_vec[:] = self.hn_beta_vec
            for i in range(self.c_k):
                if node.children[i] is not None:
                    self._set_hn_params_recursion(node.children[i],None)
        else:
            node.h_g = original_tree_node.h_g
            node.h_beta_vec[:] = original_tree_node.h_beta_vec
            if node.depth == self.c_d_max:
                node.leaf = True
                node.h_g = 0.0
            elif original_tree_node.leaf:  # leaf node
                node.leaf = True
            else:
                node.leaf = False
                for i in range(self.c_k):
                    if node.children[i] is None:
                        node.children[i] = _Node(node.depth+1,self.c_k)
                    self._set_hn_params_recursion(node.children[i],original_tree_node.children[i])

    def set_h0_params(self,
        h0_g=None,
        h0_beta_vec=None,
        h0_root=None,
        ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_g : float, optional
            A real number in :math:`[0, 1]`, by default ``None``
        h0_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default ``None``
        h0_root : contexttree._Node, optional
            A root node of a superposed tree for hyperparameters 
            by default ``None``
        """
        if h0_g is not None:
            self.h0_g = _check.float_in_closed01(h0_g,'h0_g',ParameterFormatError)
            if self.h0_root is not None:
                self._set_h0_g_recursion(self.h0_root)

        if h0_beta_vec is not None:
            _check.pos_floats(h0_beta_vec,'h0_beta_vec',ParameterFormatError)
            self.h0_beta_vec[:] = h0_beta_vec
            if self.h0_root is not None:
                self._set_h0_beta_vec_recursion(self.h0_root)
        
        if h0_root is not None:
            if type(h0_root) is not _Node:
                raise(ParameterFormatError(
                    "h0_root must be an instance of contexttree._Node"
                ))
            if self.h0_root is None:
                self.h0_root = _Node(0,self.c_k)
            self._set_h0_params_recursion(self.h0_root,h0_root)

        self.reset_hn_params()
        return self

    def get_h0_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h0_params : dict of {str: float, numpy.ndarray, contexttre._Node}
            * ``"h0_g"`` : the value of ``self.h0_g``
            * ``"h0_beta_vec"`` : the value of ``self.h0_beta_vec``
            * ``"h0_root"`` : the value of ``self.h0_root``
        """
        return {"h0_g":self.h0_g, 
                "h0_beta_vec":self.h0_beta_vec, 
                "h0_root":self.h0_root}
    
    def set_hn_params(self,
        hn_g=None,
        hn_beta_vec=None,
        hn_root=None,
        ):
        """Set the hyperparameter of the posterior distribution.

        Parameters
        ----------
        hn_g : float, optional
            A real number in :math:`[0, 1]`, by default ``None``
        hn_beta_vec : numpy.ndarray, optional
            A vector of positive real numbers, 
            by default ``None``
        hn_root : contexttree._Node, optional
            A root node of a superposed tree for hyperparameters 
            by default ``None``
        """
        if hn_g is not None:
            self.hn_g = _check.float_in_closed01(hn_g,'hn_g',ParameterFormatError)
            if self.hn_root is not None:
                self._set_hn_g_recursion(self.hn_root)

        if hn_beta_vec is not None:
            _check.pos_floats(hn_beta_vec,'hn_beta_vec',ParameterFormatError)
            self.hn_beta_vec[:] = hn_beta_vec
            if self.hn_root is not None:
                self._set_hn_beta_vec_recursion(self.hn_root)
        
        if hn_root is not None:
            if type(hn_root) is not _Node:
                raise(ParameterFormatError(
                    "hn_root must be an instance of contexttree._Node"
                ))
            if self.hn_root is None:
                self.hn_root = _Node(0,self.c_k)
            self._set_hn_params_recursion(self.hn_root,hn_root)

        self.calc_pred_dist(np.zeros(self.c_d_max,dtype=int))
        return self

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float, numpy.ndarray, contexttre._Node}
            * ``"hn_g"`` : the value of ``self.hn_g``
            * ``"hn_beta_vec"`` : the value of ``self.hn_beta_vec``
            * ``"hn_root"`` : the value of ``self.hn_root``
        """
        return {"hn_g":self.hn_g, 
                "hn_beta_vec":self.hn_beta_vec, 
                "hn_root":self.hn_root}
    
    def _update_posterior_leaf(self,node:_Node,x,i):
            tmp = node.h_beta_vec[x[i]] / node.h_beta_vec.sum()
            node.h_beta_vec[x[i]] += 1
            return tmp

    def _update_posterior_recursion(self,node:_Node,x,i):
        if node.depth < self.c_d_max and i-1-node.depth >= 0:  # inner node
            if node.children[x[i-node.depth-1]] is None:
                node.children[x[i-node.depth-1]] = _Node(
                    node.depth+1,
                    self.c_k,
                    self.hn_g,
                )
                node.children[x[i-node.depth-1]].h_beta_vec[:] = self.hn_beta_vec
                if node.depth + 1 == self.c_d_max:
                    node.children[x[i-node.depth-1]].h_g = 0.0
                    node.children[x[i-node.depth-1]].leaf = True
            tmp1 = self._update_posterior_recursion(node.children[x[i-node.depth-1]],x,i)
            tmp2 = (1 - node.h_g) * self._update_posterior_leaf(node,x,i) + node.h_g * tmp1
            node.h_g = node.h_g * tmp1 / tmp2
            return tmp2
        else:  # leaf node
            return self._update_posterior_leaf(node,x,i)
    
    def update_posterior(self,x):
        """Update the hyperparameters using traning data.

        Parameters
        ----------
        x : numpy ndarray
            1-dimensional int array
        """
        _check.nonneg_ints(x,'x',DataFormatError)
        if x.max() >= self.c_k:
            raise(DataFormatError(f"x.max() must smaller than c_k:{self.c_k}"))
        x = np.ravel(x)

        if self.hn_root is None:
            self.hn_root = _Node(0,self.c_k,self.hn_g)
            self.hn_root.h_beta_vec[:] = self.hn_beta_vec

        for i in range(x.shape[0]):
            self._update_posterior_recursion(self.hn_root,x,i)
        return self

    def _map_recursion_add_nodes(self,node:_Node):
        if node.depth == self.c_d_max:  # leaf node
            node.h_g = 0.0
            node.leaf = True
            node.map_leaf = True
        else:  # inner node
            for i in range(self.c_k):
                node.children[i] = _Node(node.depth+1,self.c_k,self.hn_g)
                node.children[i].h_beta_vec[:] = self.hn_beta_vec
                self._map_recursion_add_nodes(node.children[i])

    def _map_recursion(self,node:_Node):
        if node.depth == self.c_d_max:
            node.map_leaf = True
            return 1.0
        else:
            tmp1 = 1.0-node.h_g
            tmp_vec = np.empty(self.c_k)
            for i in range(self.c_k):
                if node.children[i] is not None:
                    tmp_vec[i] = self._map_recursion(node.children[i])
                else:
                    node.children[i] = _Node(node.depth+1,self.c_k,self.hn_g)
                    node.children[i].h_beta_vec[:] = self.hn_beta_vec
                    if 1.0 - node.h_g > node.h_g * self.hn_g ** ((self.c_k ** (self.c_d_max - node.depth) - 1)/(self.c_k-1)-1):
                        node.children[i].map_leaf = True
                        tmp_vec[i] = 1.0 - node.h_g
                    else:
                        self._map_recursion_add_nodes(node.children[i])
                        tmp_vec[i] = node.h_g * self.hn_g ** ((self.c_k ** (self.c_d_max - node.depth) - 1)/(self.c_k-1)-1)
            if tmp1 > node.h_g*tmp_vec.prod():
                node.map_leaf = True
                return tmp1
            else:
                node.map_leaf = False
                return node.h_g*tmp_vec.prod()

    def _copy_map_tree_recursion(self,copyed_node:_Node,original_node:_Node):
        copyed_node.h_g = original_node.h_g
        copyed_node.h_beta_vec[:] = original_node.h_beta_vec
        if np.all(original_node.h_beta_vec > 1):
            copyed_node.theta_vec[:] = (original_node.h_beta_vec - 1) / (np.sum(original_node.h_beta_vec) - self.c_k)
        else:
            warnings.warn("MAP estimate of theta_vec doesn't exist for the current h_beta_vec.",ResultWarning)
            copyed_node.theta_vec = None

        if original_node.map_leaf:
            copyed_node.leaf = True
        else:
            for i in range(self.c_k):
                copyed_node.children[i] = _Node(copyed_node.depth+1,self.c_k)
                self._copy_map_tree_recursion(copyed_node.children[i],original_node.children[i])

    def estimate_params(self,loss="0-1",visualize=True,filename=None,format=None):
        """Estimate the parameter under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default ``\"0-1\"``.
            This function supports only ``\"0-1\"``.
        visualize : bool, optional
            If ``True``, the estimated context tree model will be visualized, by default ``True``.
            This visualization requires ``graphviz``.
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).


        Returns
        -------
        map_root : contexttree._Node
            The root node of the estimated context tree model 
            that also contains the estimated parameters in each node.

        See Also
        --------
        graphviz.Digraph
        """

        if loss == "0-1":
            if self.hn_root is None:
                self.hn_root = _Node(0,self.c_k,self.hn_g)
                self.hn_root.h_beta_vec[:] = self.hn_beta_vec
            self._map_recursion(self.hn_root)
            map_root = _Node(0,self.c_k)
            self._copy_map_tree_recursion(map_root,self.hn_root)
            if visualize:
                import graphviz
                tree_graph = graphviz.Digraph(filename=filename,format=format)
                tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
                self._visualize_model_recursion(tree_graph, map_root, 0, None, None, 1.0, True, False)
                # Can we show the image on the console without saving the file?
                tree_graph.view()
            return map_root
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports only \"0-1\"."))
    
    def _visualize_model_recursion(self,tree_graph,node:_Node,node_id,parent_id,sibling_num,p_s,map_tree,h_params):
        tmp_id = node_id
        tmp_p_s = p_s
        
        # add node information
        label_string = f'hn_g={node.h_g:.2f}\\lp_s={tmp_p_s:.2f}\\l'
        if map_tree and not node.leaf:
            label_string += 'theta_vec\\l=None\\l'
        else:
            if h_params:
                label_string += f'hn_beta_vec\\l={np.array2string(node.h_beta_vec,precision=2,max_line_width=11)}\\l'
            elif np.all(node.h_beta_vec > 1):
                theta_vec_hat = (node.h_beta_vec - 1) / (np.sum(node.h_beta_vec) - self.c_k)
                label_string += f'theta_vec\\l={np.array2string(theta_vec_hat,precision=2,max_line_width=11)}\\l'
            else:
                warnings.warn("MAP estimate of theta_vec doesn't exist for the current hn_beta_vec.",ResultWarning)
                label_string += 'theta_vec\\l=None\\l'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_s))}')
        if tmp_p_s > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        for i in range(self.c_k):
            if node.children[i] is not None:
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,i,tmp_p_s*node.h_g,map_tree,h_params)
        
        return node_id

    def _visualize_model_recursion_none(self,tree_graph,depth,node_id,parent_id,sibling_num,p_s,h_params):
        tmp_id = node_id
        tmp_p_s = p_s
        
        # add node information
        if depth == self.c_d_max:
            label_string = 'hn_g=0.0\\l'
        else:
            label_string = f'hn_g={self.hn_g:.2f}\\l'    
        label_string += f'p_s={tmp_p_s:.2f}\\l'
        if h_params:
            label_string += f'hn_beta_vec\\l={np.array2string(self.hn_beta_vec,precision=2,max_line_width=11)}\\l'
        elif np.all(self.hn_beta_vec > 1):
            theta_vec_hat = (self.hn_beta_vec - 1) / (np.sum(self.hn_beta_vec) - self.c_k)
            label_string += f'theta_vec\\l={np.array2string(theta_vec_hat,precision=2,max_line_width=11)}\\l'
        else:
            warnings.warn("MAP estimate of theta_vec doesn't exist for the current hn_beta_vec.",ResultWarning)
            label_string += 'theta_vec\\l=None\\l'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_s))}')
        if tmp_p_s > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if depth < self.c_d_max:
            for i in range(self.c_k):
                node_id = self._visualize_model_recursion_none(tree_graph,depth+1,node_id+1,tmp_id,i,tmp_p_s*self.hn_g,h_params)
        
        return node_id

    def visualize_posterior(self,filename=None,format=None,h_params=False):
        """Visualize the posterior distribution for the parameter.
        
        This method requires ``graphviz``.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).
        h_params : bool, optional
            If ``True``, hyperparameters at each node will be visualized. 
            if ``False``, estimated parameters at each node will be visulaized.

        Examples
        --------
        >>> from bayesml import contexttree
        >>> gen_model = contexttree.GenModel(c_k=2,c_d_max=3,h_g=0.75)
        >>> gen_model.gen_params()
        >>> x = gen_model.gen_sample(500)
        >>> learn_model = contexttree.LearnModel(c_k=2,c_d_max=3,h0_g=0.75)
        >>> learn_model.update_posterior(x)
        >>> learn_model.visualize_posterior()

        .. image:: ./images/contexttree_posterior.png

        See Also
        --------
        graphviz.Digraph
        """
        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            if self.hn_root is None:
                self._visualize_model_recursion_none(tree_graph, 0, 0, None, None, 1.0, h_params)
            else:
                self._visualize_model_recursion(tree_graph, self.hn_root, 0, None, None, 1.0, False, h_params)
            # Can we show the image on the console without saving the file?
            tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy.ndarray}
            * ``"p_theta_vec"`` : the value of ``self.p_theta_vec``
        """
        return {"p_theta_vec":self.p_theta_vec}

    def _calc_pred_dist_recursion(self,node:_Node,x,i):
        if node.depth < self.c_d_max and i-1-node.depth >= 0:  # inner node
            if node.children[x[i-node.depth-1]] is None:
                node.children[x[i-node.depth-1]] = _Node(
                    node.depth+1,
                    self.c_k,
                    self.hn_g,
                )
                node.children[x[i-node.depth-1]].h_beta_vec[:] = self.hn_beta_vec
                if node.depth + 1 == self.c_d_max:
                    node.children[x[i-node.depth-1]].h_g = 0.0
                    node.children[x[i-node.depth-1]].leaf = True
            tmp1 = self._calc_pred_dist_recursion(node.children[x[i-node.depth-1]],x,i)
            tmp2 = (1 - node.h_g) * node.h_beta_vec / node.h_beta_vec.sum() + node.h_g * tmp1
            return tmp2
        else:  # leaf node
            return node.h_beta_vec / node.h_beta_vec.sum()

    def calc_pred_dist(self,x):
        """Calculate the parameters of the predictive distribution.
        
        Parameters
        ----------
        x : numpy ndarray
            1-dimensional int array
        """
        _check.nonneg_int_vec(x,'x',DataFormatError)
        if x.max() >= self.c_k:
            raise(DataFormatError(f"x.max() must smaller than c_k:{self.c_k}"))
        i = x.shape[0] - 1

        if self.hn_root is None:
            self.p_theta_vec[:] = self.hn_beta_vec / self.hn_beta_vec.sum()
        else:
            self.p_theta_vec[:] = self._calc_pred_dist_recursion(self.hn_root,x,i)
        return self

    def make_prediction(self,loss="KL"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"KL\".
            This function supports \"KL\" and \"0-1\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
            If the loss function is \"KL\", the predictive 
            distribution will be returned as a 1-dimensional 
            numpy.ndarray that consists of occurence probabilities.
        """
        if loss == "KL":
            return self.p_theta_vec
        elif loss == "0-1":
            return np.argmax(self.p_theta_vec)
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports \"0-1\" and \"KL\"."))

    def _pred_and_update_leaf(self,node:_Node,x,i):
            tmp = node.h_beta_vec / node.h_beta_vec.sum()
            node.h_beta_vec[x[i]] += 1
            return tmp

    def _pred_and_update_recursion(self,node:_Node,x,i):
        if node.depth < self.c_d_max and i-1-node.depth >= 0:  # inner node
            if node.children[x[i-node.depth-1]] is None:
                node.children[x[i-node.depth-1]] = _Node(
                    node.depth+1,
                    self.c_k,
                    self.hn_g,
                )
                node.children[x[i-node.depth-1]].h_beta_vec[:] = self.hn_beta_vec
                if node.depth + 1 == self.c_d_max:
                    node.children[x[i-node.depth-1]].h_g = 0.0
                    node.children[x[i-node.depth-1]].leaf = True
            tmp1 = self._pred_and_update_recursion(node.children[x[i-node.depth-1]],x,i)
            tmp2 = (1 - node.h_g) * self._pred_and_update_leaf(node,x,i) + node.h_g * tmp1
            node.h_g = node.h_g * tmp1[x[i]] / tmp2[x[i]]
            return tmp2
        else:  # leaf node
            return self._pred_and_update_leaf(node,x,i)

    def pred_and_update(self,x,loss="KL"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x : numpy.ndarray
            1-dimensional int array
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"KL\".
            This function supports \"KL\", and \"0-1\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        _check.nonneg_int_vec(x,'x',DataFormatError)
        if x.max() >= self.c_k:
            raise(DataFormatError(f"x.max() must smaller than c_k:{self.c_k}"))
        i = x.shape[0] - 1

        if self.hn_root is None:
            self.hn_root = _Node(0,self.c_k,self.hn_g)
            self.hn_root.h_beta_vec[:] = self.hn_beta_vec

        self.p_theta_vec[:] = self._pred_and_update_recursion(self.hn_root,x,i)
        prediction = self.make_prediction(loss=loss)
        return prediction
