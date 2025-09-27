# Document Author
# Shota Saito <shota.s@gunma-u.ac.jp>
# Yuta Nakahara <y.nakahara@waseda.jp>
r"""
Stochastic Data Generative Model
--------------------------------

* :math:`\boldsymbol{x}=[x_1, \ldots, x_p, x_{p+1}, \ldots , x_{p+q}]` : an explanatory variable. The first :math:`p` variables are continuous. The other :math:`q` variables are categorical. 
* :math:`\mathcal{Y}` : a space of an objective variable
* :math:`y \in \mathcal{Y}` : an objective variable
* :math:`D_\mathrm{max} \in \mathbb{N}` : the maximum depth of trees
* :math:`T_\mathrm{max}` : the perfect tree where all the inner nodes have the same number of child nodes and all the leaf nodes have the same depth of :math:`D_\mathrm{max}`
* :math:`\mathcal{S}_\mathrm{max}` : the set of all the nodes of :math:`T_\mathrm{max}`
* :math:`s \in \mathcal{S}_\mathrm{max}` : a node of a tree
* :math:`\mathcal{I}_\mathrm{max} \subset \mathcal{S}_\mathrm{max}` : the set of all the inner nodes of :math:`T_\mathrm{max}`
* :math:`\mathcal{L}_\mathrm{max} \subset \mathcal{S}_\mathrm{max}` : the set of all the leaf nodes of :math:`T_\mathrm{max}`
* :math:`\mathcal{T}` : the set of all the pruned subtrees of :math:`T_\mathrm{max}`
* :math:`T \in \mathcal{T}` : a pruned subtree of :math:`T_\mathrm{max}`
* :math:`\mathcal{I}_T` : the set of all the inner nodes of :math:`T`
* :math:`\mathcal{L}_T` : the set of all the leaf nodes of :math:`T`
* :math:`\boldsymbol{k}=(k_s)_{s \in \mathcal{I}_\mathrm{max}}` : indices of the features assigned to inner nodes, i.e., :math:`k_s \in \{1, 2,\ldots,p+q\}`. If :math:`k_s \leq p`, the node :math:`s` has a threshold.
* :math:`\mathcal{K}=\{ 1, 2, \ldots , p+q \}^{|\mathcal{I}_\mathrm{max}|}` : the set of all :math:`\boldsymbol{k}`
* :math:`\boldsymbol{\theta}=(\theta_s)_{s \in \mathcal{S}}` : parameters assigned to the nodes
* :math:`s_{\boldsymbol{k},T}(\boldsymbol{x}) \in \mathcal{L}_T` : a leaf node which :math:`\boldsymbol{x}` reaches under :math:`T` and :math:`\boldsymbol{k}`

.. math::
    p(y | \boldsymbol{x}, \boldsymbol{\theta}, T, \boldsymbol{k})=p(y | \theta_{s_{\boldsymbol{k},T}(\boldsymbol{x})})

Prior Distribution
------------------

* :math:`g_s \in [0,1]` : a hyperparameter assigned to each node :math:`s \in \mathcal{S}_\mathrm{max}`. For any leaf node :math:`s` of :math:`T_\mathrm{max}`, we assume :math:`g_s=0`.

.. math::
    p(\boldsymbol{k}) &= \frac{1}{|\mathcal{K}|} = \left( \frac{1}{p+q} \right)^{|\mathcal{I}_\mathrm{max}|}, \\
    p(T) &= \prod_{s \in \mathcal{I}_T} g_s \prod_{s' \in \mathcal{L}_T} (1-g_{s'}).

The prior distribution of the parameter :math:`\theta_s` is assumed to be a conjugate prior distribution for :math:`p(y | \theta_s)` and independent for each node.

Posterior Distribution
----------------------

The posterior distribution is approximated as follows:

* :math:`n \in \mathbb{N}` : a sample size
* :math:`\boldsymbol{x}^n = \{ \boldsymbol{x}_1, \boldsymbol{x}_2, \ldots, \boldsymbol{x}_n \}`
* :math:`\boldsymbol{x}_{s, \boldsymbol{k}}` : the explanatory variables of the data points that pass through :math:`s` under :math:`\boldsymbol{k}`.
* :math:`y^n = \{ y_1, y_2, \ldots, y_n \}`
* :math:`y_{s, \boldsymbol{k}}` : the objective variables of the data points that pass through :math:`s` under :math:`\boldsymbol{k}`.

First, the posterior distribution :math:`p(\boldsymbol{k}, T, \boldsymbol{\theta} | \boldsymbol{x}^n, y^n)` can be decomposed as follows:

.. math::
    p(\boldsymbol{k}, T, \boldsymbol{\theta} | \boldsymbol{x}^n, y^n) = p(\boldsymbol{k} | \boldsymbol{x}^n, y^n) p(T | \boldsymbol{x}^n, y^n, \boldsymbol{k}) p(\boldsymbol{\theta} | \boldsymbol{x}^n, y^n, \boldsymbol{k}, T).

For :math:`\boldsymbol{\theta}`, we can exactly calculate the posterior distribution :math:`p(\boldsymbol{\theta} | \boldsymbol{x}^n, y^n, \boldsymbol{k}, T)` because we assumed the conjugate prior distribution.

Also for :math:`T`, we can exactly calculate the posterior distribution :math:`p(T | \boldsymbol{x}^n, y^n, \boldsymbol{k})` by using the concept called a meta-tree. 
The meta-tree is not a tree but a set of trees where all the trees have the same feature assignment :math:`\boldsymbol{k}` to their inner nodes. 
The posterior distribution of the trees over the meta-tree defined by :math:`\boldsymbol{k}` is as follows:

.. math::
    p(T | \boldsymbol{x}^n, y^n, \boldsymbol{k}) = \prod_{s \in \mathcal{I}_T} g_{s|\boldsymbol{x}^n, y^n, \boldsymbol{k}} \prod_{s' \in \mathcal{L}_T} (1-g_{s'|\boldsymbol{x}^n, y^n, \boldsymbol{k}}),

where :math:`g_{s|\boldsymbol{x}^n, y^n, \boldsymbol{k}} \in [0,1]` can be calculated from :math:`\boldsymbol{x}^n`, :math:`y^n`, and :math:`\boldsymbol{k}` as follows:

.. math::
    g_{s|\boldsymbol{x}^n, y^n, \boldsymbol{k}} =
    \begin{cases}
        \frac{g_s \prod_{s' \in \mathrm{Ch}(s)}q(y_{s', \boldsymbol{k}}|\boldsymbol{x}_{s', \boldsymbol{k}}, s', \boldsymbol{k})}{q(y_{s, \boldsymbol{k}}|\boldsymbol{x}_{s, \boldsymbol{k}}, s, \boldsymbol{k})}, & s \in \mathcal{I}_\mathrm{max},\\
        g_s, & \mathrm{otherwise},
    \end{cases}

where :math:`\mathrm{Ch}(s)` denotes the set of child nodes of :math:`s` on :math:`T_\mathrm{max}` and :math:`q(y_{s, \boldsymbol{k}}|\boldsymbol{x}_{s, \boldsymbol{k}}, s, \boldsymbol{k})` is defined for any :math:`s \in \mathcal{S}_\mathrm{max}` as follows.

.. math::
    &q(y_{s, \boldsymbol{k}}|\boldsymbol{x}_{s, \boldsymbol{k}}, s, \boldsymbol{k}) =
    \begin{cases}
        (1-g_s) f(y_{s, \boldsymbol{k}} | \boldsymbol{x}_{s, \boldsymbol{k}}, s, \boldsymbol{k}) \\
        \qquad {}+ g_s \prod_{s' \in \mathrm{Ch}(s)} q(y_{s', \boldsymbol{k}} | \boldsymbol{x}_{s', \boldsymbol{k}}, s', \boldsymbol{k}), & s \in \mathcal{I}_\mathrm{max},\\
        f(y_{s, \boldsymbol{k}} | \boldsymbol{x}_{s, \boldsymbol{k}}, s, \boldsymbol{k}), & \mathrm{otherwise}.
    \end{cases}

Here, :math:`f(y_{s, \boldsymbol{k}} | \boldsymbol{x}_{s, \boldsymbol{k}}, s, \boldsymbol{k})` is defined as follows:

.. math::
    f(y_{s, \boldsymbol{k}} | \boldsymbol{x}_{s, \boldsymbol{k}}, s, \boldsymbol{k}) = \int p(y_{s, \boldsymbol{k}} | \boldsymbol{x}_{s, \boldsymbol{k}}, \theta_s) p(\theta_s) \mathrm{d}\theta_s.

For :math:`\boldsymbol{k}`, there are two algirithms to approximate the posterior distribution :math:`p(\boldsymbol{k} | \boldsymbol{x}^n, y^n)`: the meta-tree random forest (MTRF) and the meta-tree Markov chain Monte Carlo (MTMCMC) method. 

Approximation by MTRF
~~~~~~~~~~~~~~~~~~~~~

In MTRF, we first construct a set of feature assignment vectors :math:`\mathcal{K}' = \{\boldsymbol{k}_1, \boldsymbol{k}_2, \ldots, \boldsymbol{k}_B\}` by using the usual (non-Bayesian) random forest algorithm.
Next, for :math:`\boldsymbol{k} \in \mathcal{K}`, we approximate the posterior distribution :math:`p(\boldsymbol{k} | \boldsymbol{x}^n, y^n)` as follows:

.. math::
    p(\boldsymbol{k} | \boldsymbol{x}^n, y^n) \approx \tilde{p}(\boldsymbol{k} | \boldsymbol{x}^n, y^n) \propto \begin{cases}
        q(y_{s_\lambda, \boldsymbol{k}}|\boldsymbol{x}_{s_\lambda, \boldsymbol{k}}, s_\lambda, \boldsymbol{k}), & \boldsymbol{k} \in \mathcal{K}',\\
        0, & \mathrm{otherwise}.
    \end{cases}

where :math:`s_{\lambda}` is the root node of :math:`T_\mathrm{max}`.

The predictive distribution is approximated as follows:

.. math::
    p(y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n) = \sum_{\boldsymbol{k} \in \mathcal{K}'} \tilde{p}(\boldsymbol{k} | \boldsymbol{x}^n, y^n) q(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s_\lambda, \boldsymbol{k}),

where :math:`q(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s_\lambda, \boldsymbol{k})` is calculated in a similar manner to :math:`q(y_{s_\lambda, \boldsymbol{k}}|\boldsymbol{x}_{s_\lambda, \boldsymbol{k}}, s_\lambda, \boldsymbol{k})`.

The expectation of the predictive distribution is approximated as follows.

.. math::
    \mathbb{E}_{p(y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n)} [Y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n] = \sum_{\boldsymbol{k} \in \mathcal{K}'} \tilde{p}(\boldsymbol{k} | \boldsymbol{x}^n, y^n) \mathbb{E}_{q(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s_\lambda, \boldsymbol{k})} [Y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}],

where the expectation for :math:`q` is recursively given as follows.

.. math::
    &\mathbb{E}_{q(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s, \boldsymbol{k})} [Y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}] \\
    &= \begin{cases}
    (1-g_{s|\boldsymbol{x}^n, y^n, \boldsymbol{k}}) \mathbb{E}_{f(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s, \boldsymbol{k})} [Y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}] \\
    \qquad + g_{s|\boldsymbol{x}^n, y^n, \boldsymbol{k}} \mathbb{E}_{q(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s_\mathrm{child}, \boldsymbol{k})} [Y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}] ,& s \in \mathcal{I}_\mathrm{max},\\
    \mathbb{E}_{f(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s, \boldsymbol{k})} [Y_{n+1} | \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}],& (\mathrm{otherwise}).
    \end{cases}

Here, :math:`f(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s, \boldsymbol{k})` is calculated in a similar manner to :math:`f(y_{s, \boldsymbol{k}} | \boldsymbol{x}_{s, \boldsymbol{k}}, s, \boldsymbol{k})` and :math:`s_\mathrm{child}` is the child node of :math:`s` on the path from the root node to the leaf node :math:`s_{\boldsymbol{k},T_\mathrm{max}}(\boldsymbol{x}_{n+1})`.

Approximation by MTMCMC
~~~~~~~~~~~~~~~~~~~~~~~

In MTMCMC method, we generate a sample :math:`\boldsymbol{k}` from the posterior distribution :math:`p(\boldsymbol{k} | \boldsymbol{x}^n, y^n)` by a MCMC method, and the posterior distribution is approximated by the empirical distribution of this sample.
Let :math:`\{\boldsymbol{k}^{(t)}\}_{t=1}^{t_\mathrm{end}}` be the obtained sample. 

The predictive distribution is approximated as follows:

.. math::
    p(y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n) = \frac{1}{t_\mathrm{end}} \sum_{t=1}^{t_\mathrm{end}} q(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s_\lambda, \boldsymbol{k}^{(t)}).

The expectation of the predictive distribution is approximated as follows:

.. math::
    \mathbb{E}_{p(y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n)} [Y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n] = \frac{1}{t_\mathrm{end}} \sum_{t=1}^{t_\mathrm{end}} \mathbb{E}_{q(y_{n+1}|\boldsymbol{x}_{n+1},\boldsymbol{x}^n, y^n, s_\lambda, \boldsymbol{k}^{(t)})} [Y_{n+1}| \boldsymbol{x}_{n+1}, \boldsymbol{x}^n, y^n, \boldsymbol{k}^{(t)}].

References
----------

* Dobashi, N.; Saito, S.; Nakahara, Y.; Matsushima, T. Meta-Tree Random Forest: Probabilistic Data-Generative Model and Bayes Optimal Prediction. *Entropy* 2021, 23, 768. https://doi.org/10.3390/e23060768
* Nakahara, Y.; Saito, S.; Kamatsuka, A.; Matsushima, T. Probability Distribution on Full Rooted Trees. *Entropy* 2022, 24, 328. https://doi.org/10.3390/e24030328
* Nakahara, Y.; Saito, S.; Ichijo, N.; Kazama, K.; Matsushima, T. Bayesian Decision Theory on Decision Trees: Uncertainty Evaluation and Interpretability. *Proceedings of The 28th International Conference on Artificial Intelligence and Statistics*, in *Proceedings of Machine Learning Research* 2025, 258:1045-1053 Available from https://proceedings.mlr.press/v258/nakahara25a.html.
"""
from ._metatree import GenModel
from ._metatree import LearnModel

__all__ = ["GenModel", "LearnModel"]