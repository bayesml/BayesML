# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Document Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Wenbin Yu <ywb827748728@163.com>
import warnings
import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from sklearn import tree as sklearn_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from .. import base
from .._exceptions import ParameterFormatError, DataFormatError, CriteriaError, ResultWarning, ParameterFormatWarning
from .. import _check
from .. import bernoulli, categorical, normal, multivariate_normal, linearregression, poisson, exponential

_CMAP = plt.get_cmap("Blues")
MODELS = {
    bernoulli,
    categorical,
    normal,
    # multivariate_normal,
    linearregression,
    poisson,
    exponential,
    }
DISCRETE_MODELS = {
    bernoulli,
    categorical,
    poisson,
    }
CONTINUOUS_MODELS = {
    normal,
    # multivariate_normal,
    linearregression,
    exponential,
    }
CLF_MODELS = {
    bernoulli,
    categorical,
    }
REG_MODELS = {
    normal,
    linearregression,
    exponential,
    poisson,
    }

THRESHOLD_TYPES = {'even','random'}

def _make_thresholds(x): # num_children must be 2
    tmp_x = np.unique(x)
    tmp_th = 0
    n_l = 1
    sum_l = tmp_x[0]
    sum_sq_l = tmp_x[0] * tmp_x[0]
    n_r = len(tmp_x)-1
    sum_r = tmp_x[1:].sum()
    sum_sq_r = (tmp_x[1:]*tmp_x[1:]).sum()
    tmp_min = (sum_sq_l - sum_l*sum_l/n_l) + (sum_sq_r - sum_r*sum_r/n_r)
    for i in range(1,len(tmp_x)-1):
        x_i = tmp_x[i]
        x_i_sq = x_i * x_i

        n_l += 1
        sum_l += x_i
        sum_sq_l += x_i_sq
        
        n_r -= 1
        sum_r -= x_i
        sum_sq_r -= x_i_sq

        metric = (sum_sq_l - sum_l*sum_l/n_l) + (sum_sq_r - sum_r*sum_r/n_r)
        if metric < tmp_min:
            tmp_min = metric
            tmp_th = i    
    return np.array([tmp_x.min(),(tmp_x[tmp_th]+tmp_x[tmp_th+1])/2,tmp_x.max()])

class _Node:
    def __init__(self,
                 depth,
                 k_candidates=None,
                 h_g=0.5,
                 k=None,
                 sub_model=None,
                 children=None,
                 ranges=None,
                 thresholds=None,
                 leaf=False,
                 map_leaf=False,
                 log_children_marginal_likelihood=None,
                 log_marginal_likelihood=None,
                 ):
        self.depth = depth
        self.children = children
        self.k_candidates = k_candidates
        self.h_g = h_g
        self.k = k
        self.sub_model = sub_model
        self.ranges = ranges
        self.thresholds = thresholds
        self.leaf = leaf
        self.map_leaf = map_leaf
        self.log_children_marginal_likelihood = log_children_marginal_likelihood
        self.log_marginal_likelihood = log_marginal_likelihood
        self._is_no_sample = False

class GenModel(base.Generative):
    """ The stochastice data generative model and the prior distribution

    Parameters
    ----------
    c_dim_continuous : int
        A non-negative integer
    c_dim_categorical : int
        A non-negative integer
    c_num_children_vec : numpy.ndarray, optional
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``, by default [2,2,...,2].
        The first ``c_dim_continuous`` elements represent 
        the numbers of children of continuous features at 
        inner nodes. The other ``c_dim_categorial`` elements 
        represent those of categorical features.
        If a single integer is input, it will be broadcasted.
    c_max_depth : int, optional
        A positive integer, by default 2
    c_num_assignment_vec : numpy.ndarray, optional
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``. 
        The first ``c_dim_continuous`` elements represent 
        the maximum assignment numbers of continuous features 
        on a path. The other ``c_dim_categorial`` elements 
        represent those of categorical features. If it 
        has a negative element (e.g., -1), the corresponding 
        feature will be assigned any number of times. 
        By default [-1,...,-1].
    c_ranges : numpy.ndarray, optional
        A numpy.ndarray whose size is (c_dim_continuous,2).
        A threshold for the ``k``-th continuous feature will be 
        generated between ``c_ranges[k,0]`` and ``c_ranges[k,1]``. 
        By default, [[-3,3],[-3,3],...,[-3,3]].
    SubModel : class, optional
        bernoulli, categorical, poisson, normal, exponential, 
        or linearregression, by default bernoulli
    sub_constants : dict, optional
        constants for self.SubModel.GenModel, by default {}
    root : metatree._Node, optional
        A root node of a meta-tree, 
        by default a tree consists of only one node.
    h_k_weight_vec : numpy.ndarray, optional
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``, 
        by default [1,...,1].
    h_g : float, optional
        A real number in :math:`[0, 1]`, by default 0.5
    sub_h_params : dict, optional
        h_params for self.SubModel.GenModel, by default {}
    h_metatree_list : list of metatree._Node, optional
        Root nodes of meta-trees, by default []
    h_metatree_prob_vec : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]` 
        that represents prior distribution of h_metatree_list, 
        by default uniform distribution
        Sum of its elements must be 1.0.
    seed : {None, int}, optional
        A seed to initialize numpy.random.default_rng(),
        by default None

    Attributes
    ----------
    c_dim_features: int
        c_dim_continuous + c_dim_categorical
    """
    def __init__(
            self,
            c_dim_continuous,
            c_dim_categorical,
            c_max_depth=2,
            c_num_children_vec=2,
            c_num_assignment_vec=None,
            c_ranges=None,
            SubModel=bernoulli,
            sub_constants={},
            root=None,
            h_k_weight_vec = None,
            h_g=0.5,
            sub_h_params={},
            h_metatree_list=[],
            h_metatree_prob_vec=None,
            seed=None,
            ):
        # constants
        self.c_dim_continuous = _check.nonneg_int(c_dim_continuous,'c_dim_continuous',ParameterFormatError)
        self.c_dim_categorical = _check.nonneg_int(c_dim_categorical,'c_dim_categorical',ParameterFormatError)
        _check.pos_int(
            self.c_dim_continuous+self.c_dim_categorical,
            'c_dim_continuous+c_dim_categorical',
            ParameterFormatError)
        self.c_dim_features = self.c_dim_continuous+self.c_dim_categorical
        
        self.c_max_depth = _check.pos_int(c_max_depth,'c_max_depth',ParameterFormatError)
        
        _check.pos_ints(c_num_children_vec,'c_num_children_vec',ParameterFormatError)
        if np.any(c_num_children_vec<2):
            raise(ParameterFormatError(
                'All the elements of c_num_children_vec must be greater than or equal to 2: '
                +f'c_num_children_vec={c_num_children_vec}.'
            ))
        self.c_num_children_vec = np.ones(self.c_dim_continuous+self.c_dim_categorical,dtype=int)*2
        self.c_num_children_vec[:] = c_num_children_vec
        
        self.c_num_assignment_vec = -np.ones(self.c_dim_features,dtype=int)
        if c_num_assignment_vec is not None:
            _check.ints(c_num_assignment_vec,'c_num_assignment_vec',ParameterFormatError)
            if np.all(c_num_assignment_vec==0):
                raise(ParameterFormatError(
                    'At least one element of c_num_assignment_vec must be non-zero: '
                    +f'c_num_assignment_vec={c_num_assignment_vec}.'
                ))
            self.c_num_assignment_vec[:] = c_num_assignment_vec
        
        self.c_ranges = np.zeros([self.c_dim_continuous,2])
        self.c_ranges[:,0] -= 3
        self.c_ranges[:,1] += 3
        if c_ranges is not None:
            _check.float_vecs(c_ranges,'c_ranges',ParameterFormatError)
            self.c_ranges[:] = c_ranges
            if np.any(self.c_ranges[:,0] > self.c_ranges[:,1]):
                raise(ParameterFormatError(
                    'self.c_ranges[:,1] must be greater than or equal to self.c_ranges[:,0]'
                ))
        
        if SubModel not in MODELS:
            raise(ParameterFormatError(
                "SubModel must be bernoulli, categorical"
                +"poisson, normal, exponential, or linearregression."
            ))
        self.SubModel = SubModel

        self.sub_constants = self.SubModel.GenModel(**sub_constants).get_constants()
        
        self.rng = np.random.default_rng(seed)

        # h_params
        self.h_k_weight_vec = np.ones(self.c_dim_features)
        self.h_g = 0.5
        self.sub_h_params = {}
        self.h_metatree_list = []
        self.h_metatree_prob_vec = None

        self.set_h_params(
            h_k_weight_vec,
            h_g,
            sub_h_params,
            h_metatree_list,
            h_metatree_prob_vec,
        )

        # params
        self._root_k_candidates = []
        for i in range(self.c_dim_features):
            if self.c_num_assignment_vec[i] < 0:
                self._root_k_candidates.append(i)
            else:
                for j in range(self.c_num_assignment_vec[i]):
                    self._root_k_candidates.append(i)
        self.root = _Node(
            0,
            self._root_k_candidates,
            self.h_g,
            sub_model=self.SubModel.GenModel(
                seed=self.rng,
                **self.sub_constants,
                **self.sub_h_params),
            ranges=self.c_ranges,
            leaf=True
            )

        self.set_params(root)

    def get_constants(self):
        """Get constants of GenModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_dim_continuous"`` : the value of ``self.c_dim_continuous``
            * ``"c_dim_categorical"`` : the value of ``self.c_dim_categorical``
            * ``"c_num_children_vec"`` : the value of ``self.c_num_children_vec``
            * ``"c_max_depth"`` : the value of ``self.c_max_depth``
            * ``"c_num_assignment_vec"`` : the value of ``self.c_num_assignment_vec``
            * ``"c_ranges"`` : the value of ``self.c_ranges``
        """
        return {"c_dim_continuous":self.c_dim_continuous,
                "c_dim_categorical":self.c_dim_categorical,
                "c_num_children_vec":self.c_num_children_vec,
                "c_max_depth":self.c_max_depth,
                "c_num_assignment_vec":self.c_num_assignment_vec,
                "c_ranges":self.c_ranges,
                "sub_constants":self.sub_constants}
    
    def _make_children(self,node:_Node):
        child_k_candidates = node.k_candidates.copy()
        if self.c_num_assignment_vec[node.k] > 0:
            child_k_candidates.remove(node.k)
        node.leaf = False
        for i in range(self.c_num_children_vec[node.k]):
            if node.children[i] is None:
                node.children[i] = _Node(
                    node.depth+1,
                    k_candidates=child_k_candidates,
                    h_g=self.h_g,
                    sub_model=self.SubModel.GenModel(
                        seed=self.rng,
                        **self.sub_constants,
                        **self.sub_h_params),
                    ranges=np.array(node.ranges)
                    )
            else:
                node.children[i].k_candidates = child_k_candidates
                node.children[i].ranges = np.array(node.ranges)
            if node.thresholds is not None:
                node.children[i].ranges[node.k,0] = node.thresholds[i]
                node.children[i].ranges[node.k,1] = node.thresholds[i+1]
    
    def _gen_thresholds(self,node:_Node,threshold_type):
        node.thresholds = np.empty(self.c_num_children_vec[node.k]+1)
        if threshold_type == 'random':
            tmp = self.rng.dirichlet(np.ones(self.c_num_children_vec[node.k]))
            tmp *= (node.ranges[node.k,1] - node.ranges[node.k,0])
            node.thresholds[0] = node.ranges[node.k,0]
            for i in range(self.c_num_children_vec[node.k]):
                node.thresholds[i+1] = node.thresholds[i]+tmp[i]
        if threshold_type == 'even':
            node.thresholds[:] = np.linspace(node.ranges[node.k,0],node.ranges[node.k,1],self.c_num_children_vec[node.k]+1)

    def _gen_params_recursion(self,node:_Node,h_node:_Node,feature_fix,threshold_fix,threshold_type):
        if h_node is None:
            node.h_g = 0 if node.depth == self.c_max_depth else self.h_g
            node.sub_model.set_h_params(**self.sub_h_params)
            if node.depth == self.c_max_depth or not node.k_candidates or self.rng.random() > self.h_g:  # leaf node
                node.sub_model.gen_params()
                node.leaf = True
            else:  # inner node
                flag = node.k is None
                if not feature_fix or flag:
                    node.k = self.rng.choice(node.k_candidates,
                                             p=self.h_k_weight_vec[node.k_candidates]/self.h_k_weight_vec[node.k_candidates].sum())
                    node.children = [None for i in range(self.c_num_children_vec[node.k])]
                if node.k < self.c_dim_continuous and (not threshold_fix or flag):
                    self._gen_thresholds(node,threshold_type)
                else:
                    node.thresholds = None
                self._make_children(node)
                for i in range(self.c_num_children_vec[node.k]):
                    self._gen_params_recursion(node.children[i],None,feature_fix,threshold_fix,threshold_type)
        else:
            node.h_g = 0 if node.depth == self.c_max_depth else h_node.h_g
            try:
                sub_h_params = h_node.sub_model.get_h_params()
            except:
                sub_h_params = h_node.sub_model.get_hn_params()
            node.sub_model.set_h_params(*sub_h_params.values())
            if node.depth == self.c_max_depth or not node.k_candidates or self.rng.random() > h_node.h_g:  # leaf node
                node.sub_model.gen_params()
                node.leaf = True
            else:  # inner node
                node.k = h_node.k
                node.children = [None for i in range(self.c_num_children_vec[node.k])]
                node.thresholds = np.array(h_node.thresholds) if node.k < self.c_dim_continuous else None
                self._make_children(node)
                for i in range(self.c_num_children_vec[node.k]):
                    self._gen_params_recursion(node.children[i],h_node.children[i],feature_fix,threshold_fix,threshold_type)

    def _gen_params_recursion_feature_and_tree_fix(self,node:_Node,threshold_fix,threshold_type):
        node.h_g = 0 if node.depth == self.c_max_depth else self.h_g
        node.sub_model.set_h_params(**self.sub_h_params)
        if node.leaf:  # leaf node
            node.sub_model.gen_params()
            node.leaf = True
        else:  # inner node
            if node.k < self.c_dim_continuous and (not threshold_fix):
                self._gen_thresholds(node,threshold_type)
            else:
                node.thresholds = None
            child_k_candidates = node.k_candidates.copy()
            if self.c_num_assignment_vec[node.k] > 0:
                child_k_candidates.remove(node.k)
            node.leaf = False
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    node.children[i].k_candidates = child_k_candidates
                    node.children[i].ranges = np.array(node.ranges)
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._gen_params_recursion_feature_and_tree_fix(node.children[i],threshold_fix,threshold_type)

    def _set_params_recursion(self,node:_Node,original_node:_Node):
        if original_node.leaf:  # leaf node
            try:
                sub_params = original_node.sub_model.get_params()
            except:
                try:
                    sub_params = original_node.sub_model.estimate_params(loss='0-1',dict_out=True)
                except:
                    sub_params = original_node.sub_model.estimate_params(dict_out=True)
            node.sub_model.set_params(**sub_params)
            if node.depth == self.c_max_depth:
                node.h_g = 0
            node.leaf = True
        else:
            node.k = original_node.k
            node.children = [None for i in range(self.c_num_children_vec[node.k])]
            node.thresholds = np.array(original_node.thresholds) if node.k < self.c_dim_continuous else None
            self._make_children(node)
            for i in range(self.c_num_children_vec[node.k]):
                self._set_params_recursion(node.children[i],original_node.children[i])
    
    def _gen_sample_recursion(self,node:_Node,x_continuous,x_categorical):
        if node.leaf:  # leaf node
            if self.SubModel is linearregression:
                _, y = node.sub_model.gen_sample(sample_size=1,x=x_continuous)
                return y
            elif self.SubModel is categorical:
                return node.sub_model.gen_sample(sample_size=1,onehot=False)
            else:
                return node.sub_model.gen_sample(sample_size=1)
        else:
            if node.k < self.c_dim_continuous:
                index = 0
                for i in range(self.c_num_children_vec[node.k]-1):
                    if x_continuous[node.k] < node.thresholds[i+1]:
                        break
                    index += 1
            else:
                index = x_categorical[node.k-self.c_dim_continuous]
            return self._gen_sample_recursion(node.children[index],x_continuous,x_categorical)
    
    def _visualize_model_recursion(self,tree_graph,node:_Node,node_id,parent_id,parent_k,sibling_num,p_s):
        tmp_id = node_id
        tmp_p_s = p_s

        # add node information
        if node.leaf:
            label_string = 'k=None\\l'
        else:
            label_string = f'k={node.k}\\l'
            if node.k < self.c_dim_continuous:
                label_string += f'thresholds=\\l{np.array2string(node.thresholds[1:-1],precision=2)}\\l'
        label_string += f'h_g={node.h_g:.2f}\\lp_s={tmp_p_s:.2f}\\lsub_params={{'
        if node.leaf:
            sub_params = node.sub_model.get_params()
            for key,value in sub_params.items():
                try:
                    label_string += f'\\l{key}:{value:.2f}'
                except:
                    try:
                        label_string += f'\\l{key}:{np.array2string(value,precision=2,max_line_width=1)}'
                    except:
                        label_string += f'\\l{key}:{value}'
            label_string += '}\\l'
        else:
            label_string += '\\lNone}\\l'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_s))}')
        if tmp_p_s > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            if parent_k < self.c_dim_continuous:
                if node.ranges[parent_k,0] <= self.c_ranges[parent_k,0] + 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[*,{node.ranges[parent_k,1]:.2f})')
                elif node.ranges[parent_k,1] >= self.c_ranges[parent_k,1] - 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{node.ranges[parent_k,0]:.2f},*)')
                else:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{node.ranges[parent_k,0]:.2f},{node.ranges[parent_k,1]:.2f})')
            else:
                tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if node.leaf != True:
            for i in range(self.c_num_children_vec[node.k]):
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,node.k,i,tmp_p_s*node.h_g)
        
        return node_id

    def _set_h_g_recursion(self,node:_Node):
        node.h_g = 0 if node.depth == self.c_max_depth else self.h_g
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_h_g_recursion(node.children[i])

    def _set_sub_h_params_recursion(self,node:_Node):
        node.sub_model.set_h_params(**self.sub_h_params)
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_sub_h_params_recursion(node.children[i])

    def _set_h_params_recursion(self,node:_Node,original_node:_Node):
        if original_node is None:
            node.h_g = 0 if node.depth == self.c_max_depth else self.h_g
            node.sub_model.set_h_params(**self.sub_h_params)
            if not node.leaf:
                for i in range(self.c_num_children_vec[node.k]):
                    self._set_h_params_recursion(node.children[i],None)
        else:
            node.h_g = 0 if node.depth == self.c_max_depth else original_node.h_g
            try:
                sub_h_params = original_node.sub_model.get_h_params()
            except:
                sub_h_params = original_node.sub_model.get_hn_params()
            node.sub_model.set_h_params(*sub_h_params.values())
            if original_node.leaf or node.depth == self.c_max_depth:  # leaf node
                node.leaf = True
                node.h_g = 0
            else:
                node.k = original_node.k
                node.children = [None for i in range(self.c_num_children_vec[node.k])]
                if node.k < self.c_dim_continuous:
                    node.thresholds = np.array(original_node.thresholds)
                else:
                    node.thresholds = None
                self._make_children(node)
                for i in range(self.c_num_children_vec[node.k]):
                    self._set_h_params_recursion(node.children[i],original_node.children[i])

    def set_h_params(self,
            h_k_weight_vec = None,
            h_g=None,
            sub_h_params=None,
            h_metatree_list=None,
            h_metatree_prob_vec=None
            ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h_k_weight_vec : numpy.ndarray, optional
            A vector of positive real numbers whose length is 
            ``c_dim_continuous+c_dim_categorical``, 
            by default None.
        h_g : float, optional
            A real number in :math:`[0, 1]`, by default None
        sub_h_params : dict, optional
            h_params for self.SubModel.GenModel, by default None
        h_metatree_list : list of metatree._Node, optional
            Root nodes of meta-trees, by default None
        h_metatree_prob_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]` 
            that represents prior distribution of h_metatree_list, 
            by default None.
            Sum of its elements must be 1.0.
        """
        if h_k_weight_vec is not None:
            _check.nonneg_float_vec(h_k_weight_vec,'h_k_weight_vec',ParameterFormatError)
            _check.shape_consistency(
                h_k_weight_vec.shape[0],'h_k_weight_vec.shape[0]',
                self.c_dim_features,'self.c_dim_features',
                ParameterFormatError
                )
            self.h_k_weight_vec[:] = h_k_weight_vec

        if h_g is not None:
            self.h_g = _check.float_in_closed01(h_g,'h_g',ParameterFormatError)
            if self.h_metatree_list:
                for h_root in self.h_metatree_list:
                    self._set_h_g_recursion(h_root)

        if sub_h_params is not None:
            new_sub_h_params = {}
            for key in sub_h_params:
                if key.startswith('h0_'):
                    new_key = key.replace('h0_','h_',1)
                elif key.startswith('hn_'):
                    new_key = key.replace('hn_','h_',1)
                else:
                    new_key = key
                new_sub_h_params[new_key] = sub_h_params[key]
            self.sub_h_params = self.SubModel.GenModel(
                seed=self.rng,
                **self.sub_constants,
                **new_sub_h_params).get_h_params()
            if self.h_metatree_list:
                for h_root in self.h_metatree_list:
                    self._set_sub_h_params_recursion(h_root)

        if h_metatree_list is not None:
            if not isinstance(h_metatree_list,list):
                raise(ParameterFormatError(
                    "h_metatree_list must be a list"
                ))
            if h_metatree_list:
                for h_root in h_metatree_list:
                    if type(h_root) is not _Node:
                        raise(ParameterFormatError(
                            "all elements of h_metatree_list must be instances of metatree._Node or empty"
                        ))
            diff = len(h_metatree_list) - len(self.h_metatree_list)
            if diff < 0:
                del self.h_metatree_list[diff:]
            elif diff > 0:
                for i in range(diff):
                    self.h_metatree_list.append(
                        _Node(
                            0,
                            self._root_k_candidates,
                            self.h_g,
                            sub_model=self.SubModel.GenModel(
                                seed=self.rng,
                                **self.sub_constants,
                                **self.sub_h_params),
                            ranges=self.c_ranges,
                            )
                    )
            for i in range(len(self.h_metatree_list)):
                self._set_h_params_recursion(self.h_metatree_list[i],h_metatree_list[i])
            if h_metatree_prob_vec is not None:
                self.h_metatree_prob_vec = np.array(
                    _check.float_vec_sum_1(
                        h_metatree_prob_vec,
                        'h_metatree_prob_vec',
                        ParameterFormatError
                    )
                )
            else:
                if h_metatree_list:
                    metatree_num = len(self.h_metatree_list)
                    self.h_metatree_prob_vec = np.ones(metatree_num) / metatree_num
                else:
                    self.h_metatree_prob_vec = None
        elif h_metatree_prob_vec is not None:
            self.h_metatree_prob_vec = np.array(
                _check.float_vec_sum_1(
                    h_metatree_prob_vec,
                    'h_metatree_prob_vec',
                    ParameterFormatError
                )
            )

        if type(self.h_metatree_prob_vec) is np.ndarray:             
            if self.h_metatree_prob_vec.shape[0] != len(self.h_metatree_list):
                raise(ParameterFormatError(
                    "Length of h_metatree_list and dimension of h_metatree_prob_vec must be the same."
                ))
        elif self.h_metatree_prob_vec is None:
            if len(self.h_metatree_list) > 0:
                raise(ParameterFormatError(
                    "Length of h_metatree_list must be zero when self.h_metatree_prob_vec is None."
                ))
        else:
            raise(ParameterFormatError(
                "self.h_metatree_prob_vec must be None or a numpy.ndarray."
            ))
        return self

    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h_params : dict of {str: float, list, dict, numpy.ndarray}
            * ``"h_k_weight_vec"`` : the value of ``self.h_k_weight_vec``
            * ``"h_g"`` : the value of ``self.h_g``
            * ``"sub_h_params"`` : the value of ``self.sub_h_params``
            * ``"h_metatree_list"`` : the value of ``self.h_metatree_list``
            * ``"h_metatree_prob_vec"`` : the value of ``self.h_metatree_prob_vec``
        """
        return {"h_k_weight_vec":self.h_k_weight_vec,
                "h_g":self.h_g, 
                "sub_h_params":self.sub_h_params, 
                "h_metatree_list":self.h_metatree_list,
                "h_metatree_prob_vec":self.h_metatree_prob_vec}
    
    def gen_params(self,feature_fix=False,threshold_fix=False,tree_fix=False,threshold_type='even'):
        """Generate the parameter from the prior distribution.

        The generated vaule is set at ``self.root``.

        Parameters
        ----------
        feature_fix : bool, optional
            If ``True``, feature assignment indices will be fixed, by default ``False``.
        threshold_fix : bool, optional
            If ``True``, thresholds for continuous features will be fixed, by default ``False``. 
            If ``feature_fix`` is ``False``, ``threshold_fix`` must be ``False``. 
        tree_fix : bool, optional
            If ``True``, tree shape will be fixed, by default ``False``. 
            If ``feature_fix`` is ``False``, ``tree_fix`` must be ``False``.
        threshold_type : {'even', 'random'}, optional
            A type of threshold generating procedure, by default ``'even'``
            If ``'even'``, self.c_ranges will be recursively divided by equal intervals. 
            if ``'random'``, self.c_ranges will be recursively divided by at random intervals.
        """
        if not threshold_type in THRESHOLD_TYPES:
            raise(ParameterFormatError(
                'threshold_type must be "even" or "random"'
            ))
        if feature_fix:
            if tree_fix:
                self._gen_params_recursion_feature_and_tree_fix(self.root,threshold_fix,threshold_type)
            else:
                warnings.warn(
                    "If feature_fix=True, tree will be generated according to "
                    +"self.h_g not any element of self.h_metatree_list.",ResultWarning)
                self._gen_params_recursion(self.root,None,True,threshold_fix,threshold_type)
        else:
            if threshold_fix or tree_fix:
                warnings.warn("If feature_fix=False, threshold and tree cannot be fixed.",ResultWarning)
            if self.h_metatree_list:
                tmp_root = self.rng.choice(self.h_metatree_list,p=self.h_metatree_prob_vec)
                self._gen_params_recursion(self.root,tmp_root,False,False,threshold_type)
            else:
                self._gen_params_recursion(self.root,None,False,False,threshold_type)
        return self

    def set_params(self,root=None):
        """Set the parameter of the sthocastic data generative model.

        Parameters
        ----------
        root : metatree._Node, optional
            A root node of a meta-tree, by default None.
        """
        if root is not None:
            if type(root) is not _Node:
                raise(ParameterFormatError(
                    "root must be an instance of metatree._Node"
                ))
            self._set_params_recursion(self.root,root)
        return self

    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : dict of {str:metatree._Node}
            * ``"root"`` : The value of ``self.root``.
        """
        return {"root":self.root}

    def gen_sample(self,sample_size=None,x_continuous=None,x_categorical=None):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default ``None``
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].

        Returns
        -------
        x_continuous : numpy ndarray
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``.
            Each element x_categorical[i,j] must satisfies 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            1 dimensional array whose size is ``sample_size``.
        """                        
        if x_continuous is not None:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous = x_continuous.reshape(-1,self.c_dim_continuous)

            sample_size = x_continuous.shape[0]

            if x_categorical is not None:
                _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
                _check.shape_consistency(
                    x_categorical.shape[-1],'x_categorical.shape[-1]',
                    self.c_dim_categorical,'self.c_dim_categorical',
                    ParameterFormatError
                    )
                x_categorical = x_categorical.reshape(-1,self.c_dim_categorical)
                _check.shape_consistency(
                    x_categorical.shape[0],'x_categorical.shape[0]',
                    x_continuous.shape[0],'x_continuous.shape[0]',
                    ParameterFormatError
                    )
                for i in range(self.c_dim_categorical):
                    if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                        raise(DataFormatError(
                            f"x_categorical[:,{i}].max() must smaller than "
                            +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                            +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            else:
                x_categorical = np.empty([sample_size,self.c_dim_categorical],dtype=int)
                for i in range(self.c_dim_categorical):
                    x_categorical[:,i] = self.rng.choice(
                        self.c_num_children_vec[self.c_dim_continuous+i],
                        sample_size)

        elif x_categorical is not None:
            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            x_categorical = x_categorical.reshape(-1,self.c_dim_categorical)
            for i in range(self.c_dim_categorical):
                if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[:,{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))

            sample_size = x_categorical.shape[0]
            
            x_continuous = np.empty([sample_size,self.c_dim_continuous],dtype=float)
            for i in range(self.c_dim_continuous):
                x_continuous[:,i] = ((self.c_ranges[i,1]-self.c_ranges[i,0])
                                        * self.rng.random(sample_size)
                                        + self.c_ranges[i,0])
        elif sample_size is not None:
            sample_size = _check.pos_int(sample_size,'sample_size',DataFormatError)

            x_continuous = np.empty([sample_size,self.c_dim_continuous],dtype=float)
            for i in range(self.c_dim_continuous):
                x_continuous[:,i] = ((self.c_ranges[i,1]-self.c_ranges[i,0])
                                        * self.rng.random(sample_size)
                                        + self.c_ranges[i,0])

            x_categorical = np.empty([sample_size,self.c_dim_categorical],dtype=int)
            for i in range(self.c_dim_categorical):
                x_categorical[:,i] = self.rng.choice(
                    self.c_num_children_vec[self.c_dim_continuous+i],
                    sample_size)
        else:
            raise(DataFormatError("Either of sample_size, x_continuous, and x_categorical must be given as a input."))

        if self.SubModel in DISCRETE_MODELS:
            y = np.empty(sample_size,dtype=int)
        elif self.SubModel in CONTINUOUS_MODELS:
            y = np.empty(sample_size,dtype=float)
        
        for i in range(sample_size):
            y[i] = self._gen_sample_recursion(self.root,x_continuous[i],x_categorical[i])

        return x_continuous,x_categorical,y
        
    def save_sample(self,filename,sample_size,x=None):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"x\".

        Parameters
        ----------
        filename : str
            The filename to which the sample is saved.
            ``.npz`` will be appended if it isn't there.
        sample_size : int, optional
            A positive integer, by default ``None``
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        
        See Also
        --------
        numpy.savez_compressed
        """
        x_continuous,x_categorical,y = self.gen_sample(sample_size,x_continuous,x_categorical)
        np.savez_compressed(filename,x_continuous=x_continuous,x_categorical=x_categorical,y=y)

    def _plot_2d_threshold_recursion_continuous(self,ax,node:_Node,index):
        if not node.leaf:
            if node.k == index[0]:
                ax.vlines(x=node.thresholds[1:-1],ymin=node.ranges[index[1],0],ymax=node.ranges[index[1],1],colors='red')
            else:
                ax.hlines(y=node.thresholds[1:-1],xmin=node.ranges[index[0],0],xmax=node.ranges[index[0],1],colors='red')
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    self._plot_2d_threshold_recursion_continuous(ax,node.children[i],index)

    def _plot_1d_threshold_recursion_continuous(self,ax,node:_Node,ymin,ymax):
        if not node.leaf:
            ax.vlines(x=node.thresholds[1:-1],ymin=ymin,ymax=ymax,colors='red')
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    self._plot_1d_threshold_recursion_continuous(ax,node.children[i],ymin,ymax)

    def _plot_1d_threshold_recursion_categorical(self,ax,node:_Node,ymin,ymax):
        if not node.leaf:
            ax.vlines(
                x=np.linspace(0,
                    self.c_num_children_vec[node.k]-1,
                    2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                ymin=ymin,
                ymax=ymax,
                colors='red')
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    self._plot_1d_threshold_recursion_categorical(ax,node.children[i],ymin,ymax)

    def _plot_2d_threshold_recursion_mix(self,ax,node:_Node,categorical_index,index):
        if not node.leaf:
            if node.k == index[0]:
                if categorical_index is None:
                    ax.vlines(x=node.thresholds[1:-1],ymin=0-0.2,ymax=self.c_num_children_vec[index[1]]-1+0.2,colors='red')
                else:
                    ax.vlines(x=node.thresholds[1:-1],ymin=categorical_index-0.2,ymax=categorical_index+0.2,colors='red')
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is not None:
                        self._plot_2d_threshold_recursion_mix(ax,node.children[i],categorical_index,index)
            else:
                ax.hlines(
                    y=np.linspace(0,
                        self.c_num_children_vec[node.k]-1,
                        2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                    xmin=node.ranges[index[0],0],
                    xmax=node.ranges[index[0],1],
                    colors='red')
                for i in range(self.c_num_children_vec[node.k]):
                    if node.children[i] is not None:
                        self._plot_2d_threshold_recursion_mix(ax,node.children[i],i,index)

    def _plot_2d_threshold_recursion_categorical(self,ax,node:_Node,categorical_index,index):
        if not node.leaf:
            if node.k == index[0]:
                if categorical_index is None:
                    ax.vlines(
                        x=np.linspace(0,
                            self.c_num_children_vec[node.k]-1,
                            2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                        ymin=-0.2,
                        ymax=self.c_num_children_vec[index[1]]-1+0.2,
                        colors='red')
                else:
                    ax.vlines(
                        x=np.linspace(0,
                            self.c_num_children_vec[node.k]-1,
                            2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                        ymin=max(categorical_index-0.5,-0.2),
                        ymax=min(categorical_index+0.5,self.c_num_children_vec[index[1]]-1+0.2),
                        colors='red')
            else:
                if categorical_index is None:
                    ax.hlines(
                        y=np.linspace(0,
                            self.c_num_children_vec[node.k]-1,
                            2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                        xmin=-0.2,
                        xmax=self.c_num_children_vec[index[0]]-1+0.2,
                        colors='red')
                else:
                    ax.hlines(
                        y=np.linspace(0,
                            self.c_num_children_vec[node.k]-1,
                            2*(self.c_num_children_vec[node.k]-1)+1)[1:-1:2],
                        xmin=max(categorical_index-0.5,-0.2),
                        xmax=min(categorical_index+0.5,self.c_num_children_vec[index[0]]-1+0.2),
                        colors='red')
            for i in range(self.c_num_children_vec[node.k]):
                if node.children[i] is not None:
                    self._plot_2d_threshold_recursion_categorical(ax,node.children[i],i,index)

    def visualize_model(self,filename=None,format=None,sample_size=100,x_continuous=None,x_categorical=None):
        """Visualize the stochastic data generative model and generated samples.

        Note that values of categorical features will be shown with jitters.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).
        sample_size : int, optional
            A positive integer, by default 100
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].

        Examples
        --------
        >>> from bayesml import metatree
        >>> model = metatree.GenModel(
        >>>     c_dim_continuous=1,
        >>>     c_dim_categorical=1)
        >>> model.gen_params(threshold_type='random')
        >>> model.visualize_model()

        .. image:: ./images/metatree_example1.png
        .. image:: ./images/metatree_example2.png

        See Also
        --------
        graphviz.Digraph
        """
        _check.pos_int(sample_size,'sample_size',DataFormatError)

        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            self._visualize_model_recursion(
                tree_graph,
                self.root,
                0,
                None,
                None,
                None,
                1.0)
            # Can we show the image on the console without saving the file?
            tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)

        fig, ax = plt.subplots()
        x_continuous,x_categorical,y = self.gen_sample(sample_size,x_continuous,x_categorical)
        if self.c_dim_categorical > 0:
            x_categorical_jitter = x_categorical + 0.2*(self.rng.random(x_categorical.shape)-0.5)

        if self.SubModel in DISCRETE_MODELS:
            y_jitter = y + 0.2*(self.rng.random(y.shape)-0.5)

        if np.count_nonzero(self.c_num_assignment_vec) == 1:
            index = np.flatnonzero(self.c_num_assignment_vec)
            if index[0] >= self.c_dim_continuous:
                if self.SubModel in DISCRETE_MODELS:
                    ax.scatter(x_categorical_jitter[:,index[0]-self.c_dim_continuous],y_jitter)
                else:
                    ax.scatter(x_categorical_jitter[:,index[0]-self.c_dim_continuous],y)
                ymin, ymax = ax.get_ylim()
                self._plot_1d_threshold_recursion_categorical(ax,self.root,ymin,ymax)
                ax.set_xlabel(f'x_categorical[{index[0]-self.c_dim_continuous}]')
                ax.set_ylabel('y')
            else:
                if self.SubModel in DISCRETE_MODELS:
                    ax.scatter(x_continuous[:,index[0]],y_jitter)
                else:
                    ax.scatter(x_continuous[:,index[0]],y)
                ymin, ymax = ax.get_ylim()
                self._plot_1d_threshold_recursion_continuous(ax,self.root,ymin,ymax)
                ax.set_xlabel(f'x_continuous[{index[0]}]')
                ax.set_ylabel('y')
            plt.show()
        elif np.count_nonzero(self.c_num_assignment_vec) == 2:
            index = np.flatnonzero(self.c_num_assignment_vec)
            if np.all(index >= self.c_dim_continuous):
                mappable = ax.scatter(x_categorical_jitter[:,index[0]-self.c_dim_continuous],x_categorical_jitter[:,index[1]-self.c_dim_continuous],c=y)
                self._plot_2d_threshold_recursion_categorical(ax,self.root,None,index)
                ax.set_xlabel(f'x_categorical[{index[0]-self.c_dim_continuous}]')
                ax.set_ylabel(f'x_categorical[{index[1]-self.c_dim_continuous}]')
            elif np.all(index < self.c_dim_continuous):
                mappable = ax.scatter(x_continuous[:,index[0]],x_continuous[:,index[1]],c=y)
                self._plot_2d_threshold_recursion_continuous(ax,self.root,index)
                ax.set_xlabel(f'x_continuous[{index[0]}]')
                ax.set_ylabel(f'x_continuous[{index[1]}]')
            else:
                mappable = ax.scatter(x_continuous[:,index[0]],x_categorical_jitter[:,index[1]-self.c_dim_continuous],c=y)
                self._plot_2d_threshold_recursion_mix(ax,self.root,None,index)
                ax.set_xlabel(f'x_continuous[{index[0]}]')
                ax.set_ylabel(f'x_categorical[{index[1]-self.c_dim_continuous}]')
            fig.colorbar(mappable,label='y')
            plt.show()
        else:
            print(x_continuous)
            print(x_categorical)
            print(y)

class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.

    Parameters
    ----------
    c_dim_continuous : int
        A non-negative integer
    c_dim_categorical : int
        A non-negative integer
    c_max_depth : int, optional
        A positive integer, by default 2
    c_num_children_vec : numpy.ndarray, optional
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``, by default [2,2,...,2].
        The first ``c_dim_continuous`` elements represent 
        the numbers of children of continuous features at 
        inner nodes. The other ``c_dim_categorial`` elements 
        represent those of categorical features.
        If a single integer is input, it will be broadcasted.
    c_num_assignment_vec : numpy.ndarray, optional
        A vector of positive integers whose length is 
        ``c_dim_continuous+c_dim_categorical``. 
        The first ``c_dim_continuous`` elements represent 
        the maximum assignment numbers of continuous features 
        on a path. The other ``c_dim_categorial`` elements 
        represent those of categorical features. If it 
        has a negative element (e.g., -1), the corresponding 
        feature will be assigned any number of times. 
        By default [-1,...,-1].
    c_ranges : numpy.ndarray, optional
        A numpy.ndarray whose size is (c_dim_continuous,2).
        A threshold for the ``k``-th continuous feature will be 
        generated between ``c_ranges[k,0]`` and ``c_ranges[k,1]``. 
        By default, [[-3,3],[-3,3],...,[-3,3]].
    SubModel : class, optional
        bernoulli, categorical, poisson, normal, exponential, 
        or linearregression, by default bernoulli
    sub_constants : dict, optional
        constants for self.SubModel.LearnModel, by default {}
    h0_k_weight_vec : numpy.ndarray, optional
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``, 
        by default [1,...,1].
    h0_g : float, optional
        A real number in :math:`[0, 1]`, by default 0.5
    sub_h0_params : dict, optional
        h0_params for self.SubModel.LearnModel, by default {}
    h0_metatree_list : list of metatree._Node, optional
        Root nodes of meta-trees, by default []
    h0_metatree_prob_vec : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]` 
        that represents prior distribution of h0_metatree_list, 
        by default uniform distribution
        Sum of its elements must be 1.0.

    Attributes
    ----------
    c_dim_features: int
        c_dim_continuous + c_dim_categorical
    hn_k_weight_vec : numpy.ndarray
        A vector of positive real numbers whose length is 
        ``c_dim_continuous+c_dim_categorical``
    hn_g : float
        A real number in :math:`[0, 1]`
    sub_hn_params : dict
        hn_params for self.SubModel.LearnModel
    hn_metatree_list : list of metatree._Node
        Root nodes of meta-trees
    hn_metatree_prob_vec : numpy.ndarray
        A vector of real numbers in :math:`[0, 1]` 
        that represents prior distribution of h0_metatree_list.
        Sum of its elements is 1.0.
    """
    def __init__(
            self,
            c_dim_continuous,
            c_dim_categorical,
            c_max_depth=2,
            c_num_children_vec=2,
            c_num_assignment_vec=None,
            c_ranges=None,
            SubModel=bernoulli,
            sub_constants={},
            h0_k_weight_vec = None,
            h0_g=0.5,
            sub_h0_params={},
            h0_metatree_list=[],
            h0_metatree_prob_vec=None
            ):
        # constants
        self.c_dim_continuous = _check.nonneg_int(c_dim_continuous,'c_dim_continuous',ParameterFormatError)
        self.c_dim_categorical = _check.nonneg_int(c_dim_categorical,'c_dim_categorical',ParameterFormatError)
        _check.pos_int(
            self.c_dim_continuous+self.c_dim_categorical,
            'c_dim_continuous+c_dim_categorical',
            ParameterFormatError)
        self.c_dim_features = self.c_dim_continuous+self.c_dim_categorical
        
        self.c_max_depth = _check.pos_int(c_max_depth,'c_max_depth',ParameterFormatError)
        
        _check.pos_ints(c_num_children_vec,'c_num_children_vec',ParameterFormatError)
        if np.any(c_num_children_vec<2):
            raise(ParameterFormatError(
                'All the elements of c_num_children_vec must be greater than or equal to 2: '
                +f'c_num_children_vec={c_num_children_vec}.'
            ))
        self.c_num_children_vec = np.ones(self.c_dim_continuous+self.c_dim_categorical,dtype=int)*2
        self.c_num_children_vec[:] = c_num_children_vec
        
        self.c_num_assignment_vec = -np.ones(self.c_dim_features,dtype=int)
        if c_num_assignment_vec is not None:
            _check.ints(c_num_assignment_vec,'c_num_assignment_vec',ParameterFormatError)
            if np.all(c_num_assignment_vec==0):
                raise(ParameterFormatError(
                    'At least one element of c_num_assignment_vec must be non-zero: '
                    +f'c_num_assignment_vec={c_num_assignment_vec}.'
                ))
            self.c_num_assignment_vec[:] = c_num_assignment_vec
        
        self.c_ranges = np.zeros([self.c_dim_continuous,2])
        self.c_ranges[:,0] -= 3
        self.c_ranges[:,1] += 3
        if c_ranges is not None:
            _check.float_vecs(c_ranges,'c_ranges',ParameterFormatError)
            self.c_ranges[:] = c_ranges
            if np.any(self.c_ranges[:,0] > self.c_ranges[:,1]):
                raise(ParameterFormatError(
                    'self.c_ranges[:,1] must be greater than or equal to self.c_ranges[:,0]'
                ))
        
        if SubModel not in MODELS:
            raise(ParameterFormatError(
                "SubModel must be bernoulli, categorical"
                +"poisson, normal, exponential, or linearregression."
            ))
        self.SubModel = SubModel

        self.sub_constants = self.SubModel.LearnModel(**sub_constants).get_constants()

        self._root_k_candidates = []
        for i in range(self.c_dim_features):
            if self.c_num_assignment_vec[i] < 0:
                self._root_k_candidates.append(i)
            else:
                for j in range(self.c_num_assignment_vec[i]):
                    self._root_k_candidates.append(i)

        # h0_params
        self.h0_k_weight_vec = np.ones(self.c_dim_features)
        self.h0_g = 0.5
        self.sub_h0_params = {}
        self.h0_metatree_list = []
        self.h0_metatree_prob_vec = None

        # hn_params
        self.hn_k_weight_vec = np.ones(self.c_dim_features)
        self.hn_g = 0.5
        self.sub_hn_params = {}
        self.hn_metatree_list = []
        self.hn_metatree_prob_vec = None

        self._tmp_x_continuous = np.zeros(self.c_dim_continuous,dtype=float)
        self._tmp_x_categorical = np.zeros(self.c_dim_categorical,dtype=int)

        self.set_h0_params(
            h0_k_weight_vec,
            h0_g,
            sub_h0_params,
            h0_metatree_list,
            h0_metatree_prob_vec,
        )

    def get_constants(self):
        """Get constants of LearnModel.

        Returns
        -------
        constants : dict of {str: int, numpy.ndarray}
            * ``"c_dim_continuous"`` : the value of ``self.c_dim_continuous``
            * ``"c_dim_categorical"`` : the value of ``self.c_dim_categorical``
            * ``"c_num_children_vec"`` : the value of ``self.c_num_children_vec``
            * ``"c_max_depth"`` : the value of ``self.c_max_depth``
            * ``"c_num_assignment_vec"`` : the value of ``self.c_num_assignment_vec``
            * ``"c_ranges"`` : the value of ``self.c_ranges``
        """
        return {"c_dim_continuous":self.c_dim_continuous,
                "c_dim_categorical":self.c_dim_categorical,
                "c_num_children_vec":self.c_num_children_vec,
                "c_max_depth":self.c_max_depth,
                "c_num_assignment_vec":self.c_num_assignment_vec,
                "c_ranges":self.c_ranges,
                "sub_constants":self.sub_constants}

    def _set_h0_g_recursion(self,node:_Node):
        node.h_g = 0 if node.depth == self.c_max_depth else self.h0_g
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_h0_g_recursion(node.children[i])

    def _set_sub_h0_params_recursion(self,node:_Node):
        node.sub_model.set_h0_params(**self.sub_h0_params)
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_sub_h0_params_recursion(node.children[i])

    def _set_h0_params_recursion(self,node:_Node,original_node:_Node):
        if original_node is None:
            node.h_g = 0 if node.depth == self.c_max_depth else self.h0_g
            node.sub_model.set_h0_params(**self.sub_h0_params)
            if not node.leaf:
                for i in range(self.c_num_children_vec[node.k]):
                    self._set_h0_params_recursion(node.children[i],None)
        else:
            node.h_g = 0 if node.depth == self.c_max_depth else original_node.h_g
            try:
                sub_h0_params = original_node.sub_model.get_h_params()
            except:
                sub_h0_params = original_node.sub_model.get_h0_params()
            node.sub_model.set_h0_params(*sub_h0_params.values())
            if original_node.leaf or node.depth == self.c_max_depth:  # leaf node
                node.leaf = True
            else:
                node.k = original_node.k
                node.children = [None for i in range(self.c_num_children_vec[node.k])]
                node.thresholds = np.array(original_node.thresholds) if node.k < self.c_dim_continuous else None
                child_k_candidates = node.k_candidates.copy()
                if self.c_num_assignment_vec[node.k] > 0:
                    child_k_candidates.remove(node.k)
                node.leaf = False
                for i in range(self.c_num_children_vec[node.k]):
                    node.children[i] = _Node(
                        node.depth+1,
                        k_candidates=child_k_candidates,
                        sub_model=self.SubModel.LearnModel(
                            **self.sub_constants,
                            **self.sub_h0_params),
                        ranges=np.array(node.ranges),
                        log_children_marginal_likelihood=np.zeros(2),
                        )
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._set_h0_params_recursion(node.children[i],original_node.children[i])

    def _set_hn_g_recursion(self,node:_Node):
        node.h_g = 0 if node.depth == self.c_max_depth else self.hn_g
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_hn_g_recursion(node.children[i])

    def _set_sub_hn_params_recursion(self,node:_Node):
        node.sub_model.set_hn_params(**self.sub_hn_params)
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                self._set_sub_hn_params_recursion(node.children[i])

    def _set_hn_params_recursion(self,node:_Node,original_node:_Node):
        if original_node is None:
            node.h_g = 0 if node.depth == self.c_max_depth else self.hn_g
            node.sub_model.set_hn_params(**self.sub_hn_params)
            if not node.leaf:
                for i in range(self.c_num_children_vec[node.k]):
                    self._set_hn_params_recursion(node.children[i],None)
        else:
            node.h_g = 0 if node.depth == self.c_max_depth else original_node.h_g
            try:
                sub_hn_params = original_node.sub_model.get_h_params()
            except:
                sub_hn_params = original_node.sub_model.get_hn_params()
            node.sub_model.set_hn_params(*sub_hn_params.values())
            if original_node.leaf or node.depth == self.c_max_depth:  # leaf node
                node.leaf = True
            else:
                node.k = original_node.k
                node.children = [None for i in range(self.c_num_children_vec[node.k])]
                node.thresholds = np.array(original_node.thresholds) if node.k < self.c_dim_continuous else None
                child_k_candidates = node.k_candidates.copy()
                if self.c_num_assignment_vec[node.k] > 0:
                    child_k_candidates.remove(node.k)
                node.leaf = False
                for i in range(self.c_num_children_vec[node.k]):
                    node.children[i] = _Node(
                        node.depth+1,
                        k_candidates=child_k_candidates,
                        sub_model=self.SubModel.LearnModel(
                            **self.sub_constants,
                            **self.sub_h0_params).set_hn_params(**self.sub_hn_params),
                        ranges=np.array(node.ranges),
                        log_children_marginal_likelihood=np.zeros(2),
                        )
                    if node.thresholds is not None:
                        node.children[i].ranges[node.k,0] = node.thresholds[i]
                        node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                    self._set_hn_params_recursion(node.children[i],original_node.children[i])

    def set_h0_params(self,
        h0_k_weight_vec = None,
        h0_g=None,
        sub_h0_params=None,
        h0_metatree_list=None,
        h0_metatree_prob_vec=None
        ):
        """Set the hyperparameters of the prior distribution.

        Parameters
        ----------
        h0_k_weight_vec : numpy.ndarray, optional
            A vector of positive real numbers whose length is 
            ``c_dim_continuous+c_dim_categorical``, 
            by default None.
        h0_g : float, optional
            A real number in :math:`[0, 1]`, by default None
        sub_h0_params : dict, optional
            h0_params for self.SubModel.LearnModel, by default None
        h0_metatree_list : list of metatree._Node, optional
            Root nodes of meta-trees, by default None
        h0_metatree_prob_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]` 
            that represents prior distribution of h0_metatree_list, 
            by default None.
            Sum of its elements must be 1.0.
        """
        if h0_k_weight_vec is not None:
            _check.nonneg_float_vec(h0_k_weight_vec,'h0_k_weight_vec',ParameterFormatError)
            _check.shape_consistency(
                h0_k_weight_vec.shape[0],'h0_k_weight_vec.shape[0]',
                self.c_dim_features,'self.c_dim_features',
                ParameterFormatError
                )
            self.h0_k_weight_vec[:] = h0_k_weight_vec

        if h0_g is not None:
            self.h0_g = _check.float_in_closed01(h0_g,'h0_g',ParameterFormatError)
            if self.h0_metatree_list:
                for h0_root in self.h0_metatree_list:
                    self._set_h0_g_recursion(h0_root)

        if sub_h0_params is not None:
            new_sub_h0_params = {}
            for key in sub_h0_params:
                if key.startswith('h_'):
                    new_key = key.replace('h_','h0_',1)
                elif key.startswith('hn_'):
                    new_key = key.replace('hn_','h0_',1)
                else:
                    new_key = key
                new_sub_h0_params[new_key] = sub_h0_params[key]
            self.sub_h0_params = self.SubModel.LearnModel(
                **self.sub_constants,
                **new_sub_h0_params).get_h0_params()
            if self.h0_metatree_list:
                for h0_root in self.h0_metatree_list:
                    self._set_sub_h0_params_recursion(h0_root)

        if h0_metatree_list is not None:
            if not isinstance(h0_metatree_list,list):
                raise(ParameterFormatError(
                    "h0_metatree_list must be a list"
                ))
            if h0_metatree_list:
                for h0_root in h0_metatree_list:
                    if type(h0_root) is not _Node:
                        raise(ParameterFormatError(
                            "all elements of h0_metatree_list must be instances of metatree._Node or empty"
                        ))
            diff = len(h0_metatree_list) - len(self.h0_metatree_list)
            if diff < 0:
                del self.h0_metatree_list[diff:]
            elif diff > 0:
                for i in range(diff):
                    self.h0_metatree_list.append(
                        _Node(
                            0,
                            self._root_k_candidates,
                            self.h0_g,
                            sub_model=self.SubModel.LearnModel(
                                **self.sub_constants,
                                **self.sub_h0_params),
                            ranges=self.c_ranges,
                            log_children_marginal_likelihood=np.zeros(2),
                            )
                    )
            for i in range(len(self.h0_metatree_list)):
                self._set_h0_params_recursion(self.h0_metatree_list[i],h0_metatree_list[i])
            if h0_metatree_prob_vec is not None:
                self.h0_metatree_prob_vec = np.array(
                    _check.float_vec_sum_1(
                        h0_metatree_prob_vec,
                        'h0_metatree_prob_vec',
                        ParameterFormatError
                    )
                )
            else:
                if h0_metatree_list:
                    metatree_num = len(self.h0_metatree_list)
                    self.h0_metatree_prob_vec = np.ones(metatree_num) / metatree_num
                else:
                    self.h0_metatree_prob_vec = None
        elif h0_metatree_prob_vec is not None:
            self.h0_metatree_prob_vec = np.array(
                _check.float_vec_sum_1(
                    h0_metatree_prob_vec,
                    'h0_metatree_prob_vec',
                    ParameterFormatError
                )
            )

        if type(self.h0_metatree_prob_vec) is np.ndarray:             
            if self.h0_metatree_prob_vec.shape[0] != len(self.h0_metatree_list):
                raise(ParameterFormatError(
                    "Length of h0_metatree_list and dimension of h0_metatree_prob_vec must be the same."
                ))
        elif self.h0_metatree_prob_vec is None:
            if len(self.h0_metatree_list) > 0:
                raise(ParameterFormatError(
                    "Length of h0_metatree_list must be zero when self.h0_metatree_prob_vec is None."
                ))
        else:
            raise(ParameterFormatError(
                "self.h0_metatree_prob_vec must be None or a numpy.ndarray."
            ))

        self.reset_hn_params()
        return self

    def get_h0_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h0_params : dict of {str: float, list, dict, numpy.ndarray}
            * ``"h0_k_weight_vec"`` : the value of ``self.h0_k_weight_vec``
            * ``"h0_g"`` : the value of ``self.h0_g``
            * ``"sub_h0_params"`` : the value of ``self.sub_h0_params``
            * ``"h0_metatree_list"`` : the value of ``self.h0_metatree_list``
            * ``"h0_metatree_prob_vec"`` : the value of ``self.h0_metatree_prob_vec``
        """
        return {"h0_k_weight_vec":self.h0_k_weight_vec,
                "h0_g":self.h0_g, 
                "sub_h0_params":self.sub_h0_params, 
                "h0_metatree_list":self.h0_metatree_list,
                "h0_metatree_prob_vec":self.h0_metatree_prob_vec}
    
    def set_hn_params(self,
        hn_k_weight_vec = None,
        hn_g=None,
        sub_hn_params=None,
        hn_metatree_list=None,
        hn_metatree_prob_vec=None
        ):
        """Set the hyperparameters of the posterior distribution.

        Parameters
        ----------
        hn_k_weight_vec : numpy.ndarray, optional
            A vector of positive real numbers whose length is 
            ``c_dim_continuous+c_dim_categorical``, 
            by default None.
        hn_g : float, optional
            A real number in :math:`[0, 1]`, by default None
        sub_hn_params : dict, optional
            hn_params for self.SubModel.LearnModel, by default None
        hn_metatree_list : list of metatree._Node, optional
            Root nodes of meta-trees, by default None
        hn_metatree_prob_vec : numpy.ndarray, optional
            A vector of real numbers in :math:`[0, 1]` 
            that represents prior distribution of hn_metatree_list, 
            by default None.
            Sum of its elements must be 1.0.
        """
        if hn_k_weight_vec is not None:
            _check.nonneg_float_vec(hn_k_weight_vec,'hn_k_weight_vec',ParameterFormatError)
            _check.shape_consistency(
                hn_k_weight_vec.shape[0],'hn_k_weight_vec.shape[0]',
                self.c_dim_features,'self.c_dim_features',
                ParameterFormatError
                )
            self.hn_k_weight_vec[:] = hn_k_weight_vec

        if hn_g is not None:
            self.hn_g = _check.float_in_closed01(hn_g,'hn_g',ParameterFormatError)
            if self.hn_metatree_list:
                for hn_root in self.hn_metatree_list:
                    self._set_hn_g_recursion(hn_root)

        if sub_hn_params is not None:
            new_sub_hn_params = {}
            for key in sub_hn_params:
                if key.startswith('h_'):
                    new_key = key.replace('h_','hn_',1)
                elif key.startswith('h0_'):
                    new_key = key.replace('h0_','hn_',1)
                else:
                    new_key = key
                new_sub_hn_params[new_key] = sub_hn_params[key]
            self.sub_hn_params = self.SubModel.LearnModel(
                **self.sub_constants,
                **self.sub_h0_params).set_hn_params(**new_sub_hn_params).get_hn_params()
            if self.hn_metatree_list:
                for hn_root in self.hn_metatree_list:
                    self._set_sub_hn_params_recursion(hn_root)

        if hn_metatree_list is not None:
            if not isinstance(hn_metatree_list,list):
                raise(ParameterFormatError(
                    "hn_metatree_list must be a list"
                ))
            if hn_metatree_list:
                for hn_root in hn_metatree_list:
                    if type(hn_root) is not _Node:
                        raise(ParameterFormatError(
                            "all elements of hn_metatree_list must be instances of metatree._Node or empty"
                        ))
            diff = len(hn_metatree_list) - len(self.hn_metatree_list)
            if diff < 0:
                del self.hn_metatree_list[diff:]
            elif diff > 0:
                for i in range(diff):
                    self.hn_metatree_list.append(
                        _Node(
                            0,
                            self._root_k_candidates,
                            self.hn_g,
                            sub_model=self.SubModel.LearnModel(
                                **self.sub_constants,
                                **self.sub_h0_params).set_hn_params(**self.sub_hn_params),
                            ranges=self.c_ranges,
                            log_children_marginal_likelihood=np.zeros(2),
                            )
                    )
            for i in range(len(self.hn_metatree_list)):
                self._set_hn_params_recursion(self.hn_metatree_list[i],hn_metatree_list[i])
            if hn_metatree_prob_vec is not None:
                self.hn_metatree_prob_vec = np.array(
                    _check.float_vec_sum_1(
                        hn_metatree_prob_vec,
                        'hn_metatree_prob_vec',
                        ParameterFormatError
                    )
                )
            else:
                if hn_metatree_list:
                    metatree_num = len(self.hn_metatree_list)
                    self.hn_metatree_prob_vec = np.ones(metatree_num) / metatree_num
                else:
                    self.hn_metatree_prob_vec = None
        elif hn_metatree_prob_vec is not None:
            self.hn_metatree_prob_vec = np.array(
                _check.float_vec_sum_1(
                    hn_metatree_prob_vec,
                    'hn_metatree_prob_vec',
                    ParameterFormatError
                )
            )

        if type(self.hn_metatree_prob_vec) is np.ndarray:             
            if self.hn_metatree_prob_vec.shape[0] != len(self.hn_metatree_list):
                raise(ParameterFormatError(
                    "Length of hn_metatree_list and dimension of hn_metatree_prob_vec must be the same."
                ))
        elif self.hn_metatree_prob_vec is None:
            if len(self.hn_metatree_list) > 0:
                raise(ParameterFormatError(
                    "Length of hn_metatree_list must be zero when self.hn_metatree_prob_vec is None."
                ))
        else:
            raise(ParameterFormatError(
                "self.hn_metatree_prob_vec must be None or a numpy.ndarray."
            ))

        self.calc_pred_dist(
            np.zeros(self.c_dim_continuous,dtype=float),
            np.zeros(self.c_dim_categorical,dtype=int))
        return self

    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: float, list, dict, numpy.ndarray}
            * ``"hn_k_weight_vec"`` : the value of ``self.hn_k_weight_vec``
            * ``"hn_g"`` : the value of ``self.hn_g``
            * ``"sub_hn_params"`` : the value of ``self.sub_hn_params``
            * ``"hn_metatree_list"`` : the value of ``self.hn_metatree_list``
            * ``"hn_metatree_prob_vec"`` : the value of ``self.hn_metatree_prob_vec``
        """
        return {"hn_k_weight_vec":self.hn_k_weight_vec,
                "hn_g":self.hn_g, 
                "sub_hn_params":self.sub_hn_params, 
                "hn_metatree_list":self.hn_metatree_list,
                "hn_metatree_prob_vec":self.hn_metatree_prob_vec}
    
    def _copy_tree_from_sklearn_tree(self,new_node:_Node, original_tree,node_id):
        if original_tree.children_left[node_id] != sklearn_tree._tree.TREE_LEAF and new_node.k_candidates:  # inner node
            new_node.k = original_tree.feature[node_id]
            new_node.children = [None,None]
            if new_node.k < self.c_dim_continuous:
                new_node.thresholds = np.array(
                    [new_node.ranges[new_node.k,0],
                     original_tree.threshold[node_id],
                     new_node.ranges[new_node.k,1]])
            else:
                new_node.thresholds = None
            child_k_candidates = new_node.k_candidates.copy()
            if self.c_num_assignment_vec[new_node.k] > 0:
                child_k_candidates.remove(new_node.k)
            new_node.children[0] = _Node(
                new_node.depth+1,
                child_k_candidates,
                h_g=self.h0_g,
                sub_model=self.SubModel.LearnModel(
                    **self.sub_constants,
                    **self.sub_h0_params).set_hn_params(**self.sub_hn_params),
                ranges=np.array(new_node.ranges),
                log_children_marginal_likelihood=np.zeros(2),
                )
            if new_node.thresholds is not None:
                new_node.children[0].ranges[new_node.k,1] = new_node.thresholds[1]
            self._copy_tree_from_sklearn_tree(new_node.children[0],original_tree,original_tree.children_left[node_id])
            new_node.children[1] = _Node(
                new_node.depth+1,
                child_k_candidates,
                h_g=self.h0_g,
                sub_model=self.SubModel.LearnModel(
                    **self.sub_constants,
                    **self.sub_h0_params).set_hn_params(**self.sub_hn_params),
                ranges=np.array(new_node.ranges),
                log_children_marginal_likelihood=np.zeros(2),
                )
            if new_node.thresholds is not None:
                new_node.children[1].ranges[new_node.k,0] = new_node.thresholds[1]
            self._copy_tree_from_sklearn_tree(new_node.children[1],original_tree,original_tree.children_right[node_id])
        else:
            new_node.h_g = 0.0
            new_node.leaf = True

    def _update_posterior_leaf_batch(self,node:_Node,y):
        node.sub_model._update_posterior(y)
        return node.sub_model.calc_log_marginal_likelihood()

    def _update_posterior_recursion_batch(self,node:_Node,x_continuous,x_categorical,y):
        if node.leaf:  # leaf node
            node.log_marginal_likelihood = self._update_posterior_leaf_batch(node,y)
            return node.log_marginal_likelihood
        else:  # inner node
            if node.k < self.c_dim_continuous:
                for i in range(self.c_num_children_vec[node.k]):
                    if i == 0:
                        indices = x_continuous[:,node.k] < node.thresholds[i+1]
                    elif i == self.c_num_children_vec[node.k]-1:
                        indices = node.thresholds[i] <= x_continuous[:,node.k]
                    else:
                        indices = (node.thresholds[i] <= x_continuous[:,node.k]) & (x_continuous[:,node.k] < node.thresholds[i+1])
                    
                    if np.any(indices):
                        node.log_children_marginal_likelihood[i] = \
                            self._update_posterior_recursion_batch(
                                node.children[i],
                                x_continuous[indices],
                                x_categorical[indices],
                                y[indices],
                            )
                    else:
                        node.log_children_marginal_likelihood[i] = 0.0
            else:
                for i in range(self.c_num_children_vec[node.k]):
                    indices = x_categorical[:,node.k-self.c_dim_continuous] == i
                    if np.any(indices):
                        node.log_children_marginal_likelihood[i] = \
                            self._update_posterior_recursion_batch(
                                node.children[i],
                                x_continuous[indices],
                                x_categorical[indices],
                                y[indices],
                            )
                    else:
                        node.log_children_marginal_likelihood[i] = 0.0
            tmp1 = np.log(node.h_g) + node.log_children_marginal_likelihood.sum()
            node.log_marginal_likelihood = self._update_posterior_leaf_batch(node,y)
            tmp2 = np.logaddexp(np.log(1 - node.h_g) + node.log_marginal_likelihood, tmp1)
            node.h_g = np.exp(tmp1 - tmp2)
            return tmp2

    def _update_posterior_leaf_lr_batch(self,node:_Node,x_continuous,y):
        node.sub_model._update_posterior(x_continuous,y)
        return node.sub_model.calc_log_marginal_likelihood()

    def _update_posterior_recursion_lr_batch(self,node:_Node,x_continuous,x_categorical,y):
        if node.leaf:  # leaf node
            node.log_marginal_likelihood = self._update_posterior_leaf_lr_batch(node,x_continuous,y)
            return node.log_marginal_likelihood
        else:  # inner node
            if node.k < self.c_dim_continuous:
                for i in range(self.c_num_children_vec[node.k]):
                    if i == 0:
                        indices = x_continuous[:,node.k] < node.thresholds[i+1]
                    elif i == self.c_num_children_vec[node.k]-1:
                        indices = node.thresholds[i] <= x_continuous[:,node.k]
                    else:
                        indices = (node.thresholds[i] <= x_continuous[:,node.k]) & (x_continuous[:,node.k] < node.thresholds[i+1])
                    
                    if np.any(indices):
                        node.log_children_marginal_likelihood[i] = \
                            self._update_posterior_recursion_lr_batch(
                                node.children[i],
                                x_continuous[indices],
                                x_categorical[indices],
                                y[indices],
                            )
                    else:
                        node.log_children_marginal_likelihood[i] = 0.0
            else:
                for i in range(self.c_num_children_vec[node.k]):
                    indices = x_categorical[:,node.k-self.c_dim_continuous] == i
                    if np.any(indices):
                        node.log_children_marginal_likelihood[i] = \
                            self._update_posterior_recursion_lr_batch(
                                node.children[i],
                                x_continuous[indices],
                                x_categorical[indices],
                                y[indices],
                            )
                    else:
                        node.log_children_marginal_likelihood[i] = 0.0
            tmp1 = np.log(node.h_g) + node.log_children_marginal_likelihood.sum()
            node.log_marginal_likelihood = self._update_posterior_leaf_lr_batch(node,x_continuous,y)
            tmp2 = np.logaddexp(np.log(1 - node.h_g) + node.log_marginal_likelihood, tmp1)
            node.h_g = np.exp(tmp1 - tmp2)
            return tmp2

    def _compare_metatree_recursion(self,node1:_Node,node2:_Node):
        if node1.leaf:
            if node2.leaf:
                return True
            else:
                return False
        else:
            if node2.leaf:
                return False
            elif node1.k < self.c_dim_continuous:
                if node1.k != node2.k or not np.allclose(node1.thresholds,node2.thresholds):
                    return False
                else:
                    for i in range(self.c_num_children_vec[node1.k]):
                        if not self._compare_metatree_recursion(node1.children[i],node2.children[i]):
                            return False
                    return True
            else:
                if node1.k != node2.k:
                    return False
                else:
                    for i in range(self.c_num_children_vec[node1.k]):
                        if not self._compare_metatree_recursion(node1.children[i],node2.children[i]):
                            return False
                    return True
    
    def _marge_metatrees(self,metatree_list,metatree_prob_vec):
        num_metatrees = len(metatree_list)
        for i in range(num_metatrees):
            for j in range(i+1,num_metatrees):
                if self._compare_metatree_recursion(metatree_list[i],metatree_list[j]):
                    metatree_list[i] = None
                    metatree_prob_vec[j] += metatree_prob_vec[i]
                    metatree_prob_vec[i] = -1
                    break
        metatree_list = [tmp for tmp in metatree_list if tmp != None]
        metatree_prob_vec = metatree_prob_vec[metatree_prob_vec > -0.5]
        return metatree_list,metatree_prob_vec

    def _MTRF(self,x_continuous,x_categorical,y,n_estimators=100,**kwargs):
        """make metatrees

        Parameters
        ----------
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            values of objective variable whose dtype may be int or float
        n_estimators : int, optional
            number of trees in sklearn.RandomForestClassifier or 
            sklearn.RandomForestRegressor, by default 100

        Returns
        -------
        metatree_list : list of metatree._Node
            Each element is a root node of metatree.
        metatree_prob_vec : numpy ndarray
        """
        if np.any(self.c_num_children_vec != 2):
            raise(ParameterFormatError(
                "MTRF is supported only when all the elements of c_num_children_vec is 2."))
        if self.SubModel in CLF_MODELS:
            randomforest = RandomForestClassifier(
                n_estimators=n_estimators,max_depth=self.c_max_depth,**kwargs)
        elif self.SubModel in REG_MODELS:
            randomforest = RandomForestRegressor(
                n_estimators=n_estimators,max_depth=self.c_max_depth,**kwargs)
        else:
            raise(ParameterFormatError(
                'MTRF is supported when self.SubModel = '
                +'bernoulli, categorical, poisson, '
                +'normal, exponential, linearregression.'
                ))

        x = np.empty([y.shape[0],self.c_dim_features])
        x[:,:self.c_dim_continuous] = x_continuous
        x[:,self.c_dim_continuous:] = x_categorical

        randomforest.fit(x,y)
        
        tmp_metatree_list = [
            _Node(
                0,
                self._root_k_candidates,
                self.hn_g,
                sub_model=self.SubModel.LearnModel(
                    **self.sub_constants,
                    **self.sub_h0_params).set_hn_params(**self.sub_hn_params),
                ranges=self.c_ranges,
                log_children_marginal_likelihood=np.zeros(2),
                )
            for i in range(n_estimators)
            ]
        tmp_metatree_prob_vec = np.ones(n_estimators) / n_estimators
        for i in range(n_estimators):
            self._copy_tree_from_sklearn_tree(tmp_metatree_list[i],randomforest.estimators_[i].tree_, 0)

        return self._marge_metatrees(tmp_metatree_list,tmp_metatree_prob_vec)

    def _given_MT(self,x_continuous,x_categorical,y):
        """make metatrees

        Parameters
        ----------
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            values of objective variable whose dtype may be int or float

        Returns
        -------
        metatree_list : list of metatree._Node
            Each element is a root node of metatree.
        metatree_prob_vec : numpy ndarray
        """
        if not self.hn_metatree_list:
            raise(ParameterFormatError("given_MT is supported only when len(self.hn_metatree_list) > 0."))
        log_metatree_posteriors = np.log(self.hn_metatree_prob_vec)
        if self.SubModel is linearregression:
            for i,metatree in enumerate(self.hn_metatree_list):
                log_metatree_posteriors[i] += self._update_posterior_recursion_lr_batch(metatree,x_continuous,x_categorical,y)
        else:
            for i,metatree in enumerate(self.hn_metatree_list):
                log_metatree_posteriors[i] += self._update_posterior_recursion_batch(metatree,x_continuous,x_categorical,y)
        self.hn_metatree_prob_vec[:] = np.exp(log_metatree_posteriors - log_metatree_posteriors.max())
        self.hn_metatree_prob_vec[:] /= self.hn_metatree_prob_vec.sum()

    def _check_sample_x(self,x_continuous,x_categorical):
        if self.c_dim_continuous > 0 and self.c_dim_categorical > 0:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous = x_continuous.reshape([-1,self.c_dim_continuous])
            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            x_categorical = x_categorical.reshape([-1,self.c_dim_categorical])
            for i in range(self.c_dim_categorical):
                if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[:,{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            _check.shape_consistency(
                x_continuous.shape[0],'x_continuous.shape[0]',
                x_categorical.shape[0],'x_categorical.shape[0]',
                ParameterFormatError
                )

        elif self.c_dim_continuous > 0:
            _check.float_vecs(x_continuous,'x_continuous',DataFormatError)
            _check.shape_consistency(
                x_continuous.shape[-1],'x_continuous.shape[-1]',
                self.c_dim_continuous,'self.c_dim_continuous',
                ParameterFormatError
                )
            x_continuous = x_continuous.reshape([-1,self.c_dim_continuous])
            x_categorical = np.empty([x_continuous.shape[0],0]) # dummy

        elif self.c_dim_categorical > 0:
            _check.nonneg_int_vecs(x_categorical,'x_categorical',DataFormatError)
            _check.shape_consistency(
                x_categorical.shape[-1],'x_categorical.shape[-1]',
                self.c_dim_categorical,'self.c_dim_categorical',
                ParameterFormatError
                )
            x_categorical = x_categorical.reshape([-1,self.c_dim_categorical])
            for i in range(self.c_dim_categorical):
                if x_categorical[:,i].max() >= self.c_num_children_vec[self.c_dim_continuous+i]:
                    raise(DataFormatError(
                        f"x_categorical[:,{i}].max() must smaller than "
                        +f"self.c_num_children_vec[{self.c_dim_continuous+i}]: "
                        +f"{self.c_num_children_vec[self.c_dim_continuous+i]}"))
            x_continuous = np.empty([x_categorical.shape[0],0]) # dummy

        return x_continuous,x_categorical

    def _check_sample_y(self,x_continuous,y):
        if self.SubModel is linearregression:
            self.SubModel.LearnModel(**self.sub_constants)._check_sample(x_continuous,y)
        else:
            self.SubModel.LearnModel(**self.sub_constants)._check_sample(y)
        return np.ravel(y)

    def _check_sample(self,x_continuous,x_categorical,y):
        x_continuous, x_categorical = self._check_sample_x(x_continuous,x_categorical)
        y = self._check_sample_y(x_continuous,y)
        _check.shape_consistency(
            x_continuous.shape[0],'x_continuous.shape[0] and x_categorical.shape[0]',
            y.shape[0],'y.shape[0]',
            ParameterFormatError
            )
        return x_continuous,x_categorical,y
    
    def _MTMCMC(
            self,
            x_continuous,
            x_categorical,
            y,
            burn_in=100,
            num_metatrees=500,
            g_max=0.0,
            rho=0.99,
            phi=0.999,
            p_obj=0.3,
            threshold_type='1d_kmeans',
            seed=None,
            ):
        # check condition
        if np.any(self.c_num_children_vec != self.c_num_children_vec[0]):
            raise(ParameterFormatError(
                'MTMCMC is supported only when all the elements of '
                + 'self.c_num_children_vec are the same. '
                + f'self.c_num_children_vec = {self.c_num_children_vec}.'
            ))
        self._num_children = self.c_num_children_vec[0]

        if np.any(self.c_num_assignment_vec > 0):
            raise(ParameterFormatError(
                'MTMCMC is supported only when all the elements of '
                + 'self.c_num_assignment_vec are not positive. '
                + f'self.c_num_assignment_vec = {self.c_num_assignment_vec}.'
            ))
        
        if not np.allclose(self.h0_k_weight_vec,self.h0_k_weight_vec[0]):
            raise(ParameterFormatError(
                'MTMCMC is supported only when all the elements of '
                + 'self.h0_k_weight_vec are the same. '
                + f'self.h0_k_weight_vec = {self.h0_k_weight_vec}.'
            ))

        # add variables
        self.rng = np.random.default_rng(seed)
        self._hn_node_sub_model = self.SubModel.LearnModel(
            **self.sub_constants,
            **self.sub_h0_params,
        ).set_hn_params(**self.sub_hn_params)
        if not threshold_type in {'1d_kmeans', 'sample_midpoint'}:
            raise(ParameterFormatError(
                'threshold_type must be "1d_kmeans" or "sample_midpoint".'
            ))
        self._threshold_type = threshold_type
        self._tmp_root = _Node(
            0,
            self._root_k_candidates,
            self.hn_g,
            self.rng.choice(self._root_k_candidates),
            sub_model=copy.deepcopy(self._hn_node_sub_model),
            ranges=self.c_ranges,
            log_children_marginal_likelihood=np.zeros(self._num_children),
            )
        self._tmp_metatree_list = [self._tmp_root]
        self._tmp_metatree_count_list = [1]
        if self.SubModel is linearregression:
            self._l_last = self._generate_truncated_and_update_lr(None,self._tmp_root,True,x_continuous,x_categorical,y)
        else:
            self._l_last = self._generate_truncated_and_update(None,self._tmp_root,True,x_continuous,x_categorical,y)
        
        if self.c_dim_features == 1:
            return self._tmp_metatree_list, np.array([1])

        self._num_proposed = 1
        self._num_accepted = 1
        self._rho = rho
        self._phi = phi
        self._g_max = g_max
        self._denominator = 0.0
        self._p_obj = p_obj
        self._numerator = self._denominator * p_obj
        self._g_list = []
        self._l_list = []
        self._g_max_denominator = 0
        self._g_max_accumulated = g_max * self._g_max_denominator

        # procedure
        print(f'brun_in: {burn_in}')
        while self._num_proposed < burn_in:
            self._mh_step_truncated(
                x_continuous,
                x_categorical,
                y,
            )
            self._update_g_max()
        print()
        tmp = self._num_accepted-1
        self._tmp_metatree_count_list[-1] = 1
        print(f'burn_in + num_metatrees: {burn_in + num_metatrees}')
        while self._num_proposed < burn_in + num_metatrees:
            self._mh_step_truncated(
                x_continuous,
                x_categorical,
                y,
            )
        print()
        
        # output
        _tmp_metatree_prob_vec = np.array(self._tmp_metatree_count_list[tmp:],dtype=float)
        _tmp_metatree_prob_vec /= _tmp_metatree_prob_vec.sum()
        return self._marge_metatrees(self._tmp_metatree_list[tmp:],_tmp_metatree_prob_vec)

    def _REMTMCMC(
            self,
            x_continuous,
            x_categorical,
            y,
            burn_in=100,
            num_metatrees=500,
            num_chains=8,
            g_max=0.9,
            beta_vec=None,
            num_interval=10,
            num_exchange=8,
            threshold_type='1d_kmeans',
            seed=None,
            ):
        # check condition
        if np.any(self.c_num_children_vec != self.c_num_children_vec[0]):
            raise(ParameterFormatError(
                'MTMCMC is supported only when all the elements of '
                + 'self.c_num_children_vec are the same. '
                + f'self.c_num_children_vec = {self.c_num_children_vec}.'
            ))
        self._num_children = self.c_num_children_vec[0]

        if np.any(self.c_num_assignment_vec > 0):
            raise(ParameterFormatError(
                'MTMCMC is supported only when all the elements of '
                + 'self.c_num_assignment_vec are not positive. '
                + f'self.c_num_assignment_vec = {self.c_num_assignment_vec}.'
            ))
        
        if not np.allclose(self.h0_k_weight_vec,self.h0_k_weight_vec[0]):
            raise(ParameterFormatError(
                'MTMCMC is supported only when all the elements of '
                + 'self.h0_k_weight_vec are the same. '
                + f'self.h0_k_weight_vec = {self.h0_k_weight_vec}.'
            ))

        # add variables
        self.rng = np.random.default_rng(seed)
        self._hn_node_sub_model = self.SubModel.LearnModel(
            **self.sub_constants,
            **self.sub_h0_params,
        ).set_hn_params(**self.sub_hn_params)
        self._num_chains = _check.pos_int(num_chains,'num_chains',ParameterFormatError)
        if not threshold_type in {'1d_kmeans', 'sample_midpoint'}:
            raise(ParameterFormatError(
                'threshold_type must be "1d_kmeans" or "sample_midpoint".'
            ))
        self._threshold_type = threshold_type
        self._tmp_roots =[]
        for i in range(self._num_chains):
            self._tmp_roots.append(
                _Node(
                    0,
                    self._root_k_candidates,
                    self.hn_g,
                    self.rng.choice(self._root_k_candidates),
                    sub_model=copy.deepcopy(self._hn_node_sub_model),
                    ranges=self.c_ranges,
                    log_children_marginal_likelihood=np.zeros(self._num_children),
                )
            )
        self._tmp_metatree_lists = [[self._tmp_roots[i]] for i in range(self._num_chains)]
        self._tmp_metatree_count_lists = [[1] for i in range(self._num_chains)]
        self._l_lasts = np.zeros(self._num_chains)
        if self.SubModel is linearregression:
            for i in range(self._num_chains):
                self._l_lasts[i] = self._generate_truncated_and_update_lr(None,self._tmp_roots[i],True,x_continuous,x_categorical,y)
        else:
            for i in range(self._num_chains):
                self._l_lasts[i] = self._generate_truncated_and_update(None,self._tmp_roots[i],True,x_continuous,x_categorical,y)
        
        if self.c_dim_features == 1:
            return self._tmp_metatree_lists[self._num_chains-1], np.array([1])

        self._g_max = g_max
        self._l_list = []
        self._l_new = np.zeros(self._num_chains)
        self._beta_vec = (np.arange(self._num_chains)+1) / self._num_chains
        if beta_vec is not None:
            self._beta_vec[:] = beta_vec
            if np.any(self._beta_vec < 0) or np.any(self._beta_vec > 1):
                raise(ParameterFormatError(
                    'All the elements of beta_vec must be in [0,1] '
                    + f'beta_vec = {beta_vec}.'
                ))
        self._num_interval = _check.pos_int(num_interval,'num_interval',ParameterFormatError)
        self._num_exchange = _check.pos_int(num_exchange,'num_exchange',ParameterFormatError)    
        self._exchange_list = []    

        # procedure
        print(f'brun_in: {burn_in}')
        for i in range(burn_in):
            self._remh_step_truncated_memory_efficient(
                x_continuous,
                x_categorical,
                y,
            )
            if self._num_chains > 1 and i % self._num_interval == 0:
                self._replica_exchange_memory_efficient()
            print(f'\r{i}', end='')
        print()
        tmp = len(self._tmp_metatree_count_lists[self._num_chains-1])-1
        for i in range(self._num_chains):
            self._tmp_metatree_count_lists[i][-1] = 1
        print(f'num_metatrees: {num_metatrees}')
        for i in range(num_metatrees):
            self._remh_step_truncated_memory_efficient(
                x_continuous,
                x_categorical,
                y,
            )
            if self._num_chains > 1 and i % self._num_interval == 0:
                self._replica_exchange_memory_efficient()
            print(f'\r{i}', end='')
        print()
        
        # output
        _tmp_metatree_prob_vec = np.array(self._tmp_metatree_count_lists[self._num_chains-1][tmp:],dtype=float)
        _tmp_metatree_prob_vec /= _tmp_metatree_prob_vec.sum()
        return self._marge_metatrees(self._tmp_metatree_lists[self._num_chains-1][tmp:],_tmp_metatree_prob_vec)

    def _generate_truncated_and_update_lr(
            self,
            last:_Node,
            new:_Node,
            flag:bool,
            x_continuous,
            x_categorical,
            y,
            ):
        # always
        if flag:
            new.sub_model=copy.deepcopy(self._hn_node_sub_model)
            new.sub_model._update_posterior(x_continuous,y)
            new.log_marginal_likelihood = new.sub_model.calc_log_marginal_likelihood()
        else:
            new.sub_model = copy.deepcopy(last.sub_model)
            new.log_marginal_likelihood = last.log_marginal_likelihood

        # leaf node
        if (new.depth == self.c_max_depth 
            or (np.allclose(x_continuous,x_continuous[0]) 
                and np.allclose(x_categorical,x_categorical[0]))):
            new.h_g = 0
            new.leaf = True
            return new.log_marginal_likelihood

        # inner node
        if flag:
            new.k = self.rng.choice(new.k_candidates)
        elif self.rng.random() > min(last.h_g,self._g_max):
            flag = True
            last.division_flag = False
            new.division_flag = False
            new.k = self.rng.choice([k for k in new.k_candidates if k != last.k])
        else:
            last.division_flag = True
            new.division_flag = True
            new.k = last.k

        new.ranges[:,0] = x_continuous.min(axis=0)
        new.ranges[:,1] = x_continuous.max(axis=0)
        self._make_children_for_mcmc(new,x_continuous)

        if new.k < self.c_dim_continuous:
            for i in range(self._num_children):
                if i == 0:
                    indices = x_continuous[:,new.k] < new.thresholds[i+1]
                elif i == self._num_children-1:
                    indices = new.thresholds[i] <= x_continuous[:,new.k]
                else:
                    indices = (new.thresholds[i] <= x_continuous[:,new.k]) & (x_continuous[:,new.k] < new.thresholds[i+1])
                
                if np.any(indices):
                    new.log_children_marginal_likelihood[i] = \
                        self._generate_truncated_and_update_lr(
                            None if flag else last.children[i],
                            new.children[i],
                            flag,
                            x_continuous[indices],
                            x_categorical[indices],
                            y[indices],
                        )
                else:
                    new.log_children_marginal_likelihood[i] = 0.0
                    new.children[i].leaf = True
                    new.children[i].log_marginal_likelihood = 0.0
                    new.children[i].sub_model = copy.deepcopy(self._hn_node_sub_model)
        else:
            for i in range(self._num_children):
                indices = x_categorical[:,new.k-self.c_dim_continuous] == i
                if np.any(indices):
                    new.log_children_marginal_likelihood[i] = \
                        self._generate_truncated_and_update_lr(
                            None if flag else last.children[i],
                            new.children[i],
                            flag,
                            x_continuous[indices],
                            x_categorical[indices],
                            y[indices],
                        )
                else:
                    new.log_children_marginal_likelihood[i] = 0.0
                    new.children[i].leaf = True
                    new.children[i].log_marginal_likelihood = 0.0
                    new.children[i].sub_model = copy.deepcopy(self._hn_node_sub_model)
        tmp1 = np.log(new.h_g) + new.log_children_marginal_likelihood.sum()
        tmp2 = np.logaddexp(np.log(1 - new.h_g) + new.log_marginal_likelihood, tmp1)
        new.h_g = np.exp(tmp1 - tmp2)
        return tmp2

    def _generate_truncated_and_update(
            self,
            last:_Node,
            new:_Node,
            flag:bool,
            x_continuous,
            x_categorical,
            y,
            ):
        # always
        if flag:
            new.sub_model=copy.deepcopy(self._hn_node_sub_model)
            new.sub_model._update_posterior(y)
            new.log_marginal_likelihood = new.sub_model.calc_log_marginal_likelihood()
        else:
            new.sub_model = copy.deepcopy(last.sub_model)
            new.log_marginal_likelihood = last.log_marginal_likelihood

        # leaf node
        if (new.depth == self.c_max_depth 
            or (np.allclose(x_continuous,x_continuous[0]) 
                and np.allclose(x_categorical,x_categorical[0]))):
            new.h_g = 0
            new.leaf = True
            return new.log_marginal_likelihood

        # inner node
        if flag:
            new.k = self.rng.choice(new.k_candidates)
        elif self.rng.random() > min(last.h_g,self._g_max):
            flag = True
            last.division_flag = False
            new.division_flag = False
            new.k = self.rng.choice([k for k in new.k_candidates if k != last.k])
        else:
            last.division_flag = True
            new.division_flag = True
            new.k = last.k

        new.ranges[:,0] = x_continuous.min(axis=0)
        new.ranges[:,1] = x_continuous.max(axis=0)
        self._make_children_for_mcmc(new,x_continuous)

        if new.k < self.c_dim_continuous:
            for i in range(self._num_children):
                if i == 0:
                    indices = x_continuous[:,new.k] < new.thresholds[i+1]
                elif i == self._num_children-1:
                    indices = new.thresholds[i] <= x_continuous[:,new.k]
                else:
                    indices = (new.thresholds[i] <= x_continuous[:,new.k]) & (x_continuous[:,new.k] < new.thresholds[i+1])
                
                if np.any(indices):
                    new.log_children_marginal_likelihood[i] = \
                        self._generate_truncated_and_update(
                            None if flag else last.children[i],
                            new.children[i],
                            flag,
                            x_continuous[indices],
                            x_categorical[indices],
                            y[indices],
                        )
                else:
                    new.log_children_marginal_likelihood[i] = 0.0
                    new.children[i].leaf = True
                    new.children[i].log_marginal_likelihood = 0.0
                    new.children[i].sub_model = copy.deepcopy(self._hn_node_sub_model)
        else:
            for i in range(self._num_children):
                indices = x_categorical[:,new.k-self.c_dim_continuous] == i
                if np.any(indices):
                    new.log_children_marginal_likelihood[i] = \
                        self._generate_truncated_and_update(
                            None if flag else last.children[i],
                            new.children[i],
                            flag,
                            x_continuous[indices],
                            x_categorical[indices],
                            y[indices],
                        )
                else:
                    new.log_children_marginal_likelihood[i] = 0.0
                    new.children[i].leaf = True
                    new.children[i].log_marginal_likelihood = 0.0
                    new.children[i].sub_model = copy.deepcopy(self._hn_node_sub_model)
        tmp1 = np.log(new.h_g) + new.log_children_marginal_likelihood.sum()
        tmp2 = np.logaddexp(np.log(1 - new.h_g) + new.log_marginal_likelihood, tmp1)
        new.h_g = np.exp(tmp1 - tmp2)
        return tmp2

    def _mh_step_truncated(self,x_continuous,x_categorical,y):
        self._tmp_root = _Node(
            0,
            self._root_k_candidates,
            self.hn_g,
            self.rng.choice(self._root_k_candidates),
            sub_model=copy.deepcopy(self._hn_node_sub_model),
            ranges=self.c_ranges,
            log_children_marginal_likelihood=np.zeros(self._num_children),
            )

        if self.SubModel is linearregression:
            _l_new = self._generate_truncated_and_update_lr(
                self._tmp_metatree_list[-1],
                self._tmp_root,
                False,
                x_continuous,
                x_categorical,
                y,
            )
        else:
            _l_new = self._generate_truncated_and_update(
                self._tmp_metatree_list[-1],
                self._tmp_root,
                False,
                x_continuous,
                x_categorical,
                y,
            )

        _t_posteror_new = self._calc_truncated_posterior_lean(self._tmp_root)
        _t_posteror_last = self._calc_truncated_posterior_lean(self._tmp_metatree_list[-1])

        if self.rng.random() < np.exp(_l_new-_t_posteror_last-self._l_last+_t_posteror_new):
            # accept
            self._tmp_metatree_list.append(self._tmp_root)
            self._tmp_metatree_count_list.append(1)
            self._l_last = _l_new
            self._l_list.append(_l_new)
            self._num_proposed += 1
            self._num_accepted += 1
            self._numerator *= self._rho
            self._numerator += 1
            self._denominator *= self._rho
            self._denominator += 1
            print(f'\r{self._num_proposed}(accepted:{self._num_accepted})', end='')
        else:
            # reject
            self._tmp_metatree_count_list[-1] += 1
            self._l_list.append(self._l_last)
            self._num_proposed += 1
            self._numerator *= self._rho
            self._denominator *= self._rho
            self._denominator += 1
            print(f'\r{self._num_proposed}(accepted:{self._num_accepted})', end='')

    def _remh_step_truncated_memory_efficient(self,x_continuous,x_categorical,y):
        for i in range(self._num_chains-1):
            self._tmp_roots[i] = _Node(
                0,
                self._root_k_candidates,
                self.hn_g,
                self.rng.choice(self._root_k_candidates),
                sub_model=copy.deepcopy(self._hn_node_sub_model),
                ranges=self.c_ranges,
                log_children_marginal_likelihood=np.zeros(self._num_children),
            )

            if self.SubModel is linearregression:
                _l_new = self._generate_truncated_and_update_lr(
                    self._tmp_metatree_lists[i][-1],
                    self._tmp_roots[i],
                    False,
                    x_continuous,
                    x_categorical,
                    y,
                )
            else:
                _l_new = self._generate_truncated_and_update(
                    self._tmp_metatree_lists[i][-1],
                    self._tmp_roots[i],
                    False,
                    x_continuous,
                    x_categorical,
                    y,
                )

            _t_posteror_new = self._calc_truncated_posterior_lean(self._tmp_roots[i])
            _t_posteror_last = self._calc_truncated_posterior_lean(self._tmp_metatree_lists[i][-1])

            if self.rng.random() < np.exp((_l_new-self._l_lasts[i])*self._beta_vec[i]-_t_posteror_last+_t_posteror_new):
                # accept
                self._tmp_metatree_lists[i][-1] = self._tmp_roots[i]
                self._tmp_metatree_count_lists[i][-1] = 1
                self._l_lasts[i] = _l_new
            else:
                # reject
                self._tmp_metatree_count_lists[i][-1] += 1

        self._tmp_roots[-1] = _Node(
            0,
            self._root_k_candidates,
            self.hn_g,
            self.rng.choice(self._root_k_candidates),
            sub_model=copy.deepcopy(self._hn_node_sub_model),
            ranges=self.c_ranges,
            log_children_marginal_likelihood=np.zeros(self._num_children),
        )

        if self.SubModel is linearregression:
            _l_new = self._generate_truncated_and_update_lr(
                self._tmp_metatree_lists[-1][-1],
                self._tmp_roots[-1],
                False,
                x_continuous,
                x_categorical,
                y,
            )
        else:
            _l_new = self._generate_truncated_and_update(
                self._tmp_metatree_lists[-1][-1],
                self._tmp_roots[-1],
                False,
                x_continuous,
                x_categorical,
                y,
            )

        _t_posteror_new = self._calc_truncated_posterior_lean(self._tmp_roots[-1])
        _t_posteror_last = self._calc_truncated_posterior_lean(self._tmp_metatree_lists[-1][-1])

        if self.rng.random() < np.exp((_l_new-self._l_lasts[-1])*self._beta_vec[-1]-_t_posteror_last+_t_posteror_new):
            # accept
            self._tmp_metatree_lists[-1].append(self._tmp_roots[-1])
            self._tmp_metatree_count_lists[-1].append(1)
            self._l_lasts[-1] = _l_new
        else:
            # reject
            self._tmp_metatree_count_lists[-1][-1] += 1

        self._l_list.append(self._l_lasts[-1])

    def _replica_exchange_memory_efficient(self):
        self._exchange_list.append(-1)
        for i in range(self._num_exchange):
            j = self.rng.choice(self._num_chains-1)
            if self.rng.random() < np.exp(
                    self._l_lasts[j]*self._beta_vec[j+1]
                    +self._l_lasts[j+1]*self._beta_vec[j]
                    -self._l_lasts[j]*self._beta_vec[j]
                    -self._l_lasts[j+1]*self._beta_vec[j+1]
                ):
                if j == self._num_chains-2:
                    self._exchange_list.append(j)
                    self._tmp_metatree_lists[j+1].append(self._tmp_metatree_lists[j][-1])
                    self._tmp_metatree_lists[j][-1] = self._tmp_metatree_lists[j+1][-2]
                    self._tmp_metatree_count_lists[j][-1] = 1
                    self._tmp_metatree_count_lists[j+1][-1] -= 1
                    self._tmp_metatree_count_lists[j+1].append(1)
                    tmp = self._l_lasts[j]
                    self._l_lasts[j] = self._l_lasts[j+1]
                    self._l_lasts[j+1] = tmp
                    self._l_list.append(self._l_lasts[j+1])
                else:
                    self._exchange_list.append(j)
                    tmp = self._tmp_metatree_lists[j+1][-1]
                    self._tmp_metatree_lists[j+1][-1] = self._tmp_metatree_lists[j][-1]
                    self._tmp_metatree_lists[j][-1] = tmp
                    self._tmp_metatree_count_lists[j][-1] = 1
                    self._tmp_metatree_count_lists[j+1][-1] = 1
                    tmp = self._l_lasts[j]
                    self._l_lasts[j] = self._l_lasts[j+1]
                    self._l_lasts[j+1] = tmp

    def _calc_truncated_posterior_lean(self,node:_Node):
        if node.k is None:
            return 0.0
        elif node.division_flag:
            tmp = 0
            for i in range(self._num_children):
                tmp += self._calc_truncated_posterior_lean(node.children[i])
            return np.log(min(node.h_g,self._g_max)) + tmp
        else:
            return np.log(1.0-min(node.h_g,self._g_max))

    def _update_g_max(self):
        p_hat = self._numerator / self._denominator
        if p_hat > self._p_obj:
            g_max_new = self._g_max * self._p_obj / p_hat
        else:
            g_max_new = 1.0 - (1.0-self._g_max) * (1.0-self._p_obj) / (1.0-p_hat)
        self._g_max_accumulated *= self._phi
        self._g_max_accumulated += g_max_new
        self._g_max_denominator *= self._phi
        self._g_max_denominator += 1
        self._g_max = self._g_max_accumulated / self._g_max_denominator
        self._g_list.append(self._g_max)

    def _make_children_for_mcmc(self,node:_Node,x):
        child_k_candidates = node.k_candidates.copy()
        if self.c_num_assignment_vec[node.k] > 0:
            child_k_candidates.remove(node.k)
        if node.k < self.c_dim_continuous:
            if node.ranges[node.k,0] == node.ranges[node.k,1]:
                node.thresholds = np.ones(self._num_children+1) * node.ranges[node.k,0]
            elif self._threshold_type == '1d_kmeans' and self._num_children == 2:
                node.thresholds = _make_thresholds(x[:,node.k])
            else:
                node.thresholds = np.arange(
                    node.ranges[node.k,0],
                    node.ranges[node.k,1]+(node.ranges[node.k,1]-node.ranges[node.k,0])/self._num_children/2,
                    (node.ranges[node.k,1]-node.ranges[node.k,0])/self._num_children,
                )
        else:
            node.thresholds = None
        node.children = [None for i in range(self._num_children)]
        for i in range(self._num_children):
            node.children[i] = _Node(
                node.depth+1,
                child_k_candidates,
                h_g=self.h0_g,
                ranges=np.array(node.ranges),
                log_children_marginal_likelihood=np.zeros(self._num_children),
                )
            if node.thresholds is not None:
                node.children[i].ranges[node.k,0] = node.thresholds[i]
                node.children[i].ranges[node.k,1] = node.thresholds[i+1]


    def update_posterior(self,x_continuous=None,x_categorical=None,y=None,alg_type='MTRF',**kwargs):
        """Update the hyperparameters of the posterior distribution using traning data.

        Parameters
        ----------
        x_continuous : numpy ndarray, optional
            2 dimensional float array whose size is ``(sample_size,c_dim_continuous)``, 
            by default None.
        x_categorical : numpy ndarray, optional
            2 dimensional int array whose size is ``(sample_size,c_dim_categorical)``, 
            by default None. Each element x_categorical[i,j] must satisfy 
            0 <= x_categorical[i,j] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            values of objective variable whose dtype may be int or float
        alg_type : {'MTRF', 'given_MT'}, optional
            type of algorithm, by default 'MTRF'
        **kwargs : dict, optional
            optional parameters of algorithms, by default {}
        """
        x_continuous,x_categorical,y = self._check_sample(x_continuous,x_categorical,y)

        if alg_type == 'MTRF':
            self.hn_metatree_list, self.hn_metatree_prob_vec \
                = self._MTRF(x_continuous,x_categorical,y,**kwargs)
            self._given_MT(x_continuous,x_categorical,y)
        elif alg_type == 'given_MT':
            self._given_MT(x_continuous,x_categorical,y)            
        elif alg_type == 'MTMCMC':
            self.hn_metatree_list, self.hn_metatree_prob_vec \
                = self._MTMCMC(x_continuous,x_categorical,y,**kwargs)
        elif alg_type == 'REMTMCMC':
            self.hn_metatree_list, self.hn_metatree_prob_vec \
                = self._REMTMCMC(x_continuous,x_categorical,y,**kwargs)
        return self

    def _map_recursion_add_nodes(self,node:_Node):
        if node.depth == self.c_max_depth or not node.k_candidates:  # leaf node
            node.h_g = 0.0
            node.sub_model = self.SubModel.LearnModel(
                **self.sub_constants,
                **self.sub_h0_params).set_hn_params(**self.sub_hn_params)
            node.leaf = True
            node.map_leaf = True
        else:  # inner node
            node.k = node.k_candidates[self.hn_k_weight_vec[node.k_candidates].argmax()]
            node.children = [None for i in range(self.c_num_children_vec[node.k])]
            if node.k < self.c_dim_continuous:
                node.thresholds = np.linspace(
                    node.ranges[node.k,0],
                    node.ranges[node.k,1],
                    self.c_num_children_vec[node.k]+1
                    )
            else:
                node.thresholds = None
            child_k_candidates = node.k_candidates.copy()
            if self.c_num_assignment_vec[node.k] > 0:
                child_k_candidates.remove(node.k)
            # node.leaf = False # To distinguish the leaf with no sample from map leaf, node.leaf must be left as it is.
            for i in range(self.c_num_children_vec[node.k]):
                node.children[i] = _Node(
                    node.depth+1,
                    child_k_candidates,
                    self.hn_g,
                    ranges=np.array(node.ranges)
                    )
                if node.thresholds is not None:
                    node.children[i].ranges[node.k,0] = node.thresholds[i]
                    node.children[i].ranges[node.k,1] = node.thresholds[i+1]
                self._map_recursion_add_nodes(node.children[i])

    def _map_recursion(self,node:_Node):
        if node.leaf:
            if node.depth == self.c_max_depth or not node.k_candidates:
                node.map_leaf = True
                return 1.0
            else:
                sum_nodes = 0
                num_nodes = 1
                rest_num_children_vec = np.sort(self.c_num_children_vec[node.k_candidates])
                for i in range(min(self.c_max_depth-node.depth,len(node.k_candidates))):
                    sum_nodes += num_nodes
                    num_nodes *= rest_num_children_vec[i]
                if 1.0 - node.h_g > node.h_g * self.hn_g ** (sum_nodes-1):
                    node.map_leaf = True
                    return 1.0 - node.h_g
                else:
                    self._map_recursion_add_nodes(node)
                    return node.h_g * self.hn_g ** (sum_nodes-1)
        else:
            tmp1 = 1.0-node.h_g
            tmp2 = node.h_g
            for i in range(self.c_num_children_vec[node.k]):
                tmp2 *= self._map_recursion(node.children[i])
            if tmp1 > tmp2:
                node.map_leaf = True
                return tmp1
            else:
                node.map_leaf = False
                return tmp2

    def _copy_map_tree_recursion(self,copied_node:_Node,original_node:_Node):
        copied_node.h_g = original_node.h_g
        if original_node.map_leaf:
            copied_node.sub_model = copy.deepcopy(original_node.sub_model)
            copied_node.leaf = True
        else:
            copied_node.k = original_node.k
            copied_node.children = [None for i in range(self.c_num_children_vec[copied_node.k])]
            if copied_node.k < self.c_dim_continuous:
                copied_node.thresholds = np.array(original_node.thresholds)
            else:
                copied_node.thresholds = None
            child_k_candidates = copied_node.k_candidates.copy()
            if self.c_num_assignment_vec[copied_node.k] > 0:
                child_k_candidates.remove(copied_node.k)
            copied_node.leaf = False
            for i in range(self.c_num_children_vec[copied_node.k]):
                copied_node.children[i] = _Node(
                    copied_node.depth+1,
                    child_k_candidates,
                    ranges=np.array(copied_node.ranges),
                    )
                if copied_node.thresholds is not None:
                    copied_node.children[i].ranges[copied_node.k,0] = copied_node.thresholds[i]
                    copied_node.children[i].ranges[copied_node.k,1] = copied_node.thresholds[i+1]
                self._copy_map_tree_recursion(copied_node.children[i],original_node.children[i])

    def estimate_params(self,loss="0-1",visualize=True,filename=None,format=None):
        r"""Estimate the parameter under the given criterion.

        The approximate MAP meta-tree 
        :math:`M_{T,\boldsymbol{k}_b} = \mathrm{argmax} p(M_{T,\boldsymbol{k}_{b'}} | \boldsymbol{x}^n, y^n)` 
        will be returned.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default ``"0-1"``.
            This function supports only ``"0-1"``.
        visualize : bool, optional
            If ``True``, the estimated metatree will be visualized, by default ``True``.
            This visualization requires ``graphviz``.
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``"pdf"``, ``"png"``, ...).


        Returns
        -------
        map_root : metatree._Node
            The root node of the estimated meta-tree 
            that also contains the estimated parameters in each node.

        See Also
        --------
        graphviz.Digraph

        Warnings
        --------
        Multiple metatrees can represent equivalent model classes. 
        This function does not take such duplication into account.
        """

        if loss == "0-1":
            map_index = 0
            map_prob = -1.0
            map_root = _Node(
                0,
                self._root_k_candidates,
                self.hn_g,
                ranges=self.c_ranges,
                leaf=True,
                )
            if self.hn_metatree_list:
                for i,metatree in enumerate(self.hn_metatree_list):
                    prob = self.hn_metatree_prob_vec[i] * self._map_recursion(metatree)
                    if prob > map_prob:
                        map_index = i
                        map_prob = prob
                self._copy_map_tree_recursion(map_root,self.hn_metatree_list[map_index])
            else:
                warnings.warn(
                    "self.hn_metatree_list is empty. "
                    +"Therefore, one of the most likely model tree will be returned.",
                    ResultWarning)
                self._map_recursion(map_root)
            if visualize:
                import graphviz
                tree_graph = graphviz.Digraph(filename=filename,format=format)
                tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
                self._visualize_model_recursion(
                    tree_graph,
                    map_root,
                    0,
                    None,
                    None,
                    None,
                    1.0,
                    map_prob,
                    True,
                    False,
                    )
                tree_graph.view()
            return map_root
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports only \"0-1\"."))
    
    def _visualize_model_recursion(self,tree_graph,node:_Node,node_id,parent_id,parent_k,sibling_num,p_s,approx_posterior,map_tree,h_params):
        tmp_id = node_id
        tmp_p_s = p_s

        # add node information
        if node.leaf:
            label_string = 'k=None\\l'
        else:
            label_string = f'k={node.k}\\l'
            if node.k < self.c_dim_continuous:
                label_string += f'thresholds=\\l{np.array2string(node.thresholds[1:-1],precision=2)}\\l'
        label_string += f'hn_g={node.h_g:.2f}\\lp_s={tmp_p_s:.2f}\\l'
        if node.sub_model is not None:
            if h_params:
                label_string += 'sub_hn_params={'
                sub_params = node.sub_model.get_hn_params()
            else:
                label_string += 'sub_params={'
                try:
                    sub_params = node.sub_model.estimate_params(loss='0-1',dict_out=True)
                except:
                    sub_params = node.sub_model.estimate_params(dict_out=True)

            for key,value in sub_params.items():
                try:
                    label_string += f'\\l{key}:{value:.2f}'
                except:
                    try:
                        label_string += f'\\l{key}:{np.array2string(value,precision=2,max_line_width=1)}'
                    except:
                        label_string += f'\\l{key}:{value}'
            label_string += '}\\l'
        else:
            label_string += 'sub_model=\\lNone\\l'
            
        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_s))}')
        if tmp_p_s > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            if parent_k < self.c_dim_continuous:
                if node.ranges[parent_k,0] <= self.c_ranges[parent_k,0] + 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[*,{node.ranges[parent_k,1]:.2f})')
                elif node.ranges[parent_k,1] >= self.c_ranges[parent_k,1] - 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{node.ranges[parent_k,0]:.2f},*)')
                else:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{node.ranges[parent_k,0]:.2f},{node.ranges[parent_k,1]:.2f})')
            else:
                tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        elif map_tree:
            tree_graph.edge(f'{tmp_id}', f'{tmp_id}', label=f'approximate\\lmodel tree\\lposterior\\l{approx_posterior:.2f}\\l',color='invis',fontname='helvetica')
        else:
            tree_graph.edge(f'{tmp_id}', f'{tmp_id}', label=f'approximate\\lmeta-tree\\lposterior\\l{approx_posterior:.2f}\\l',color='invis',fontname='helvetica')
        
        if not node.leaf:
            for i in range(self.c_num_children_vec[node.k]):
                node_id = self._visualize_model_recursion(tree_graph,node.children[i],node_id+1,tmp_id,node.k,i,tmp_p_s*node.h_g,approx_posterior,map_tree,h_params)
        
        return node_id

    def _visualize_model_recursion_none(self,tree_graph,depth,k_candidates,ranges,node_id,parent_id,parent_k,sibling_num,p_s,h_params):
        tmp_id = node_id
        tmp_p_s = p_s
        
        # add node information
        if depth == self.c_max_depth or not k_candidates:
            label_string = 'k=None\\l'
        else:
            k = k_candidates[self.hn_k_weight_vec[k_candidates].argmax()]
            label_string = f'k={k}\\l'
            if k < self.c_dim_continuous:
                thresholds = np.linspace(ranges[k,0],ranges[k,1],self.c_num_children_vec[k]+1)
                label_string += f'thresholds=\\l{np.array2string(thresholds[1:-1],precision=2)}\\l'
            else:
                thresholds = None
            child_k_candidates = k_candidates.copy()
            if self.c_num_assignment_vec[k] > 0:
                child_k_candidates.remove(k)
        label_string += f'hn_g={self.hn_g:.2f}\\lp_s={tmp_p_s:.2f}\\l'

        sub_model = self.SubModel.LearnModel(
            **self.sub_constants,
            **self.sub_h0_params).set_hn_params(**self.sub_hn_params)
        if h_params:
            label_string += 'sub_hn_params={'
            sub_params = sub_model.get_hn_params()
        else:
            label_string += 'sub_params={'
            try:
                sub_params = sub_model.estimate_params(loss='0-1',dict_out=True)
            except:
                sub_params = sub_model.estimate_params(dict_out=True)

        for key,value in sub_params.items():
            try:
                label_string += f'\\l{key}:{value:.2f}'
            except:
                try:
                    label_string += f'\\l{key}:{np.array2string(value,precision=2,max_line_width=1)}'
                except:
                    label_string += f'\\l{key}:{value}'
        label_string += '}\\l'

        tree_graph.node(name=f'{tmp_id}',label=label_string,fillcolor=f'{rgb2hex(_CMAP(tmp_p_s))}')
        if tmp_p_s > 0.65:
            tree_graph.node(name=f'{tmp_id}',fontcolor='white')
        
        # add edge information
        if parent_id is not None:
            if parent_k < self.c_dim_continuous:
                if ranges[parent_k,0] <= self.c_ranges[parent_k,0] + 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[*,{ranges[parent_k,1]:.2f})')
                elif ranges[parent_k,1] >= self.c_ranges[parent_k,1] - 1.0E-8:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{ranges[parent_k,0]:.2f},*)')
                else:
                    tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'[{ranges[parent_k,0]:.2f},{ranges[parent_k,1]:.2f})')
            else:
                tree_graph.edge(f'{parent_id}', f'{tmp_id}', label=f'{sibling_num}')
        
        if depth < self.c_max_depth and k_candidates:
            for i in range(self.c_num_children_vec[k]):
                child_ranges = np.array(ranges)
                if thresholds is not None:
                    child_ranges[k,0] = thresholds[i]
                    child_ranges[k,1] = thresholds[i+1]
                node_id = self._visualize_model_recursion_none(tree_graph,depth+1,child_k_candidates,child_ranges,node_id+1,tmp_id,k,i,tmp_p_s*self.hn_g,h_params)
        
        return node_id

    def visualize_posterior(self,filename=None,format=None,num_metatrees=3,h_params=False):
        """Visualize the posterior distribution for the parameter.
        
        This method requires ``graphviz``.

        Parameters
        ----------
        filename : str, optional
            Filename for saving the figure, by default ``None``
        format : str, optional
            Rendering output format (``\"pdf\"``, ``\"png\"``, ...).
        num_metatrees : int, optional
            Number of metatrees to be visualized, by default 3.
        h_params : bool, optional
            If ``True``, hyperparameters at each node will be visualized. 
            if ``False``, estimated parameters at each node will be visulaized.

        Examples
        --------
        >>> from bayesml import metatree
        >>> gen_model = metatree.GenModel(
        >>>     c_dim_continuous=1,
        >>>     c_dim_categorical=1)
        >>> gen_model.gen_params(threshold_type='random')
        >>> x_continuous,x_categorical,y = gen_model.gen_sample(200)
        >>> learn_model = metatree.LearnModel(
        >>>     c_dim_continuous=1,
        >>>     c_dim_categorical=1)
        >>> learn_model.update_posterior(x_continuous,x_categorical,y)
        >>> learn_model.visualize_posterior(num_metatrees=2)

        .. image:: ./images/metatree_posterior2.png

        See Also
        --------
        graphviz.Digraph
        """
        try:
            import graphviz
            tree_graph = graphviz.Digraph(filename=filename,format=format)
            tree_graph.attr("node",shape="box",fontname="helvetica",style="rounded,filled")
            if not self.hn_metatree_list:
                warnings.warn(
                    "self.hn_metatree_list is empty. "
                    +"Therefore, one of the most likely meta-tree will be visualized.",
                    ResultWarning)
                self._visualize_model_recursion_none(
                    tree_graph,
                    0,
                    self._root_k_candidates,
                    self.c_ranges,
                    0,
                    None,
                    None,
                    None,
                    1.0,
                    h_params,
                    )
            else:
                node_id = -1
                indices = np.argsort(self.hn_metatree_prob_vec)[::-1]
                for i in range(min(num_metatrees,len(self.hn_metatree_list))):
                    node_id = self._visualize_model_recursion(
                        tree_graph,
                        self.hn_metatree_list[indices[i]],
                        node_id+1,
                        None,
                        None,
                        None,
                        1.0,
                        self.hn_metatree_prob_vec[indices[i]],
                        False,
                        h_params,
                        )
            # Can we show the image on the console without saving the file?
            tree_graph.view()
        except ImportError as e:
            print(e)
        except graphviz.CalledProcessError as e:
            print(e)
    
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        This model does not have a simple parametric expression of the predictive distribution.
        Therefore, this function returns ``None``.

        Returns
        -------
        ``None``
        """
        return None
    
    def _calc_pred_dist_recursion(self,node:_Node,x_continuous,x_categorical):
        node.sub_model.calc_pred_dist()
        if not node.leaf:  # inner node
            if node.k < self.c_dim_continuous:
                index = 0
                for i in range(self.c_num_children_vec[node.k]-1):
                    if x_continuous[node.k] < node.thresholds[i+1]:
                        break
                    index += 1
            else:
                index = x_categorical[node.k-self.c_dim_continuous]
            self._calc_pred_dist_recursion(node.children[index],x_continuous,x_categorical)

    def _calc_pred_dist_recursion_lr(self,node:_Node,x_continuous,x_categorical):
        node.sub_model._calc_pred_dist(x_continuous)
        if not node.leaf:  # inner node
            if node.k < self.c_dim_continuous:
                index = 0
                for i in range(self.c_num_children_vec[node.k]-1):
                    if x_continuous[node.k] < node.thresholds[i+1]:
                        break
                    index += 1
            else:
                index = x_categorical[node.k-self.c_dim_continuous]
            self._calc_pred_dist_recursion_lr(node.children[index],x_continuous,x_categorical)

    def calc_pred_dist(self,x_continuous=None,x_categorical=None):
        """Calculate the parameters of the predictive distribution.
        
        Parameters
        ----------
        x_continuous : numpy ndarray, optional
            A float vector whose length is ``self.c_dim_continuous``, 
            by default None.
        x_categorical : numpy ndarray, optional
            A int vector whose length is ``self.c_dim_categorical``, 
            by default None. Each element x_categorical[i] must satisfy 
            0 <= x_categorical[i] < self.c_num_children_vec[self.c_dim_continuous+i].
        """
        x_continuous,x_categorical = self._check_sample_x(x_continuous,x_categorical)

        self._tmp_x_continuous[:] = x_continuous
        self._tmp_x_categorical[:] = x_categorical
        if self.SubModel is linearregression:
            for root in self.hn_metatree_list:
                self._calc_pred_dist_recursion_lr(root,self._tmp_x_continuous,self._tmp_x_categorical)
        else:
            for root in self.hn_metatree_list:
                self._calc_pred_dist_recursion(root,self._tmp_x_continuous,self._tmp_x_categorical)
        return self

    def _make_prediction_recursion_squared(self,node:_Node):
        if node.leaf:  # leaf node
            return node.sub_model.make_prediction(loss='squared')
        else:  # inner node
            if node.k < self.c_dim_continuous:
                index = 0
                for i in range(self.c_num_children_vec[node.k]-1):
                    if self._tmp_x_continuous[node.k] < node.thresholds[i+1]:
                        break
                    index += 1
            else:
                index = self._tmp_x_categorical[node.k-self.c_dim_continuous]
            return ((1 - node.h_g) * node.sub_model.make_prediction(loss='squared')
                    + node.h_g * self._make_prediction_recursion_squared(node.children[index]))

    def _make_prediction_recursion_kl(self,node:_Node):
        if node.leaf:  # leaf node
            return node.sub_model.make_prediction(loss='KL')
        else:  # inner node
            if node.k < self.c_dim_continuous:
                index = 0
                for i in range(self.c_num_children_vec[node.k]-1):
                    if self._tmp_x_continuous[node.k] < node.thresholds[i+1]:
                        break
                    index += 1
            else:
                index = self._tmp_x_categorical[node.k-self.c_dim_continuous]
            return ((1 - node.h_g) * node.sub_model.make_prediction(loss='KL')
                    + node.h_g * self._make_prediction_recursion_kl(node.children[index]))

    def make_prediction(self,loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        if loss == "squared":
            if self.SubModel is categorical:
                tmp_pred_vec = np.empty([len(self.hn_metatree_list),self.sub_constants['c_degree']])
            else:
                tmp_pred_vec = np.empty(len(self.hn_metatree_list))
            for i,metatree in enumerate(self.hn_metatree_list):
                tmp_pred_vec[i] = self._make_prediction_recursion_squared(metatree)
            return self.hn_metatree_prob_vec @ tmp_pred_vec
        elif loss == "0-1":
            if self.SubModel in CLF_MODELS:
                degree = 2 if self.SubModel is bernoulli else self.sub_constants['c_degree']
                tmp_pred_dist_vec = np.empty([len(self.hn_metatree_list),degree])
                for i,metatree in enumerate(self.hn_metatree_list):
                    tmp_pred_dist_vec[i] = self._make_prediction_recursion_kl(metatree)
                return np.argmax(self.hn_metatree_prob_vec @ tmp_pred_dist_vec)
            else:
                raise(CriteriaError("Unsupported loss function! \"0-1\" is supported "
                                    +"only when self.SubModel is bernoulli or categorical."))
        elif loss == "KL":
            if self.SubModel in CLF_MODELS:
                degree = 2 if self.SubModel is bernoulli else self.sub_constants['c_degree']
                tmp_pred_dist_vec = np.empty([len(self.hn_metatree_list),degree])
                for i,metatree in enumerate(self.hn_metatree_list):
                    tmp_pred_dist_vec[i] = self._make_prediction_recursion_kl(metatree)
                return self.hn_metatree_prob_vec @ tmp_pred_dist_vec
            else:
                raise(CriteriaError("Unsupported loss function! \"KL\" is supported "
                                    +"only when self.SubModel is bernoulli or categorical."))
        else:
            raise(CriteriaError("Unsupported loss function! "
                                +"This function supports \"squared\", \"0-1\", and \"KL\"."))

    def pred_and_update(self,x_continuous=None,x_categorical=None,y=None,loss="squared"):
        """Predict a new data point and update the posterior sequentially.

        Parameters
        ----------
        x_continuous : numpy ndarray, optional
            A float vector whose length is ``self.c_dim_continuous``, 
            by default None.
        x_categorical : numpy ndarray, optional
            A int vector whose length is ``self.c_dim_categorical``, 
            by default None. Each element x_categorical[i] must satisfy 
            0 <= x_categorical[i] < self.c_num_children_vec[self.c_dim_continuous+i].
        y : numpy ndarray
            values of objective variable whose dtype may be int or float
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"squared\".
            This function supports \"squared\", \"0-1\", and \"KL\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
        self.calc_pred_dist(x_continuous,x_categorical)
        prediction = self.make_prediction(loss=loss)
        self.update_posterior(x_continuous,x_categorical,y,alg_type='given_MT')
        return prediction

    def _calc_pred_var_recursion(self,node:_Node):
        if node.leaf:  # leaf node
            return (node.sub_model.make_prediction(loss='squared'),
                    node.sub_model.calc_pred_var())
        else:  # inner node
            if node.k < self.c_dim_continuous:
                index = 0
                for i in range(self.c_num_children_vec[node.k]-1):
                    if self._tmp_x_continuous[node.k] < node.thresholds[i+1]:
                        break
                    index += 1
            else:
                index = self._tmp_x_categorical[node.k-self.c_dim_continuous]
            
            tmp_mean_child,tmp_var_child = self._calc_pred_var_recursion(node.children[index])
            tmp_mean = node.sub_model.make_prediction(loss='squared')
            tmp_var = node.sub_model.calc_pred_var()

            mix_mean = (1-node.h_g) * tmp_mean + node.h_g * tmp_mean_child
            mix_var = ((1-node.h_g) * ((mix_mean-tmp_mean)*(mix_mean-tmp_mean)+tmp_var)
                       + node.h_g * ((mix_mean-tmp_mean_child)*(mix_mean-tmp_mean_child)+tmp_var_child))

            return mix_mean,mix_var

    def calc_pred_var(self):
        """Calculate the variance of the predictive distribution.
        
        Returns
        -------
        var : float
            The variance of the predictive distribution.
        """
        if self.SubModel not in {normal,linearregression}:
            raise(ParameterFormatError("SubModel must be normal or linearregression."))
        tmp_means = np.empty(len(self.hn_metatree_list))
        tmp_vars = np.empty(len(self.hn_metatree_list))
        for i,metatree in enumerate(self.hn_metatree_list):
            tmp_means[i],tmp_vars[i] = self._calc_pred_var_recursion(metatree)
        mix_mean = self.hn_metatree_prob_vec @ tmp_means
        return self.hn_metatree_prob_vec @ ((tmp_means-mix_mean)*(tmp_means-mix_mean)+tmp_vars)

    def _calc_feature_importances_recursion(self,node:_Node):
        if node.leaf:
            return 0
        else:
            tmp_feature_importances = np.zeros(self.c_dim_features)
            for i in range(self.c_num_children_vec[node.k]):
                tmp_feature_importances += self._calc_feature_importances_recursion(node.children[i])
            
            for i in range(self.c_num_children_vec[node.k]):
                tmp_feature_importances[node.k] += node.children[i].log_marginal_likelihood
            tmp_feature_importances[node.k] -= node.log_marginal_likelihood
            
            return node.h_g * tmp_feature_importances

    def calc_feature_importances(self):
        """Calculate the feature importances
        
        Returns
        -------
        feature_importances : numpy.ndarray
            The feature importances.
        """
        feature_importances = np.zeros(self.c_dim_features)
        for i,metatree in enumerate(self.hn_metatree_list):
            feature_importances += self.hn_metatree_prob_vec[i] * self._calc_feature_importances_recursion(metatree)
        return feature_importances

    def _calc_pred_density_recursion(self,node:_Node,y):
        if node.leaf:  # leaf node
            return node.sub_model._calc_pred_density(y)
        else:  # inner node
            if node.k < self.c_dim_continuous:
                index = 0
                for i in range(self.c_num_children_vec[node.k]-1):
                    if self._tmp_x_continuous[node.k] < node.thresholds[i+1]:
                        break
                    index += 1
            else:
                index = self._tmp_x_categorical[node.k-self.c_dim_continuous]
            return ((1 - node.h_g) * node.sub_model._calc_pred_density(y)
                    + node.h_g * self._calc_pred_density_recursion(node.children[index],y))

    def calc_pred_density(self,y):
        """Calculate the values of the probability density function of the predictive distribution.
        
        Parameters
        ----------
        y : numpy.ndarray
            A float vector
        
        Returns
        -------
        p_y : numpy.ndarray
            The values of the probability density function of the predictive distribution.
        """
        y = _check.floats(y,'y',DataFormatError)
        tmp = 0.0
        for i,metatree in enumerate(self.hn_metatree_list):
            tmp += self.hn_metatree_prob_vec[i] * self._calc_pred_density_recursion(metatree,y)
        return tmp
