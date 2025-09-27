<img src="../logos/BayesML_logo.png" width="200">

# Docstringの運用規則 Ver.3

<div style="text-align:right;float:right">
作成：中原
</div>

* [numpyスタイル](https://numpydoc.readthedocs.io/en/latest/format.html)を採用する
  * 関連するライブラリの多くが採用しており，利用者もそちらに慣れていると思われる．我々も慣れておいたほうが良い．
* VSCodeのautoDocstringという拡張機能が便利
* パッケージの説明は`__init__.py`の冒頭に書く
  * 書くべき項目は以下．
    * データ生成確率モデル（尤度関数）
    * 事前分布
    * 事後分布
    * （事後）予測分布
  * 書き方は以下
    * `r"""` で始めると Raw docstring となり、バックスラッシュを含めることができる。
    * 変数とその取りうる値を箇条書きで書く
      * 例，``` * :math:` x \in \mathcal{X}\` ```
    * 密度関数などの式を別行立てで書く．
      * 例，``` .. math:: e^{i \pi} + 1 = 0 ```
* クラスやメソッドの説明では，
  * 必ずshort summaryを`"""`の直後に概要を1行書く．
  * 引数がある場合は必ずPrametersを書く．その際，引数が理論的に満たすべき条件を書く．
    * 例，分散は正の実数
  * 戻り値がある場合は必ずReturnsを書く
  * `visualize_model`, `visualize_posterior`は必ずExampleを書く
* bernoulliパッケージの`__init__.py`, `_bernoulli.py`を参考にしてほしい

## 例1

```
r"""
The Bernoulli distribution with the beta prior distribution.

The stochastic data generative model is as follows:

* :math:`x \in \{ 0, 1\}`: a data point
* :math:`p \in [0, 1]`: a parameter 

.. math:: \text{Bern}(x|p) = p^x (1-p)^{1-x}.

The prior distribution is as follows:

* :math:`\alpha_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_0 \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`B(\cdot,\cdot): \mathbb{R}_{>0} \times \mathbb{R}_{>0} \to \times \mathbb{R}_{>0}`: the Beta function

.. math:: \text{Beta}(p|\alpha_0,\beta_0) = \frac{1}{B(\alpha_0, \beta_0)} p^{\alpha_0} (1-p)^{\beta_0}.

The posterior distribution is as follows:

* :math:`x^n = (x_1, x_2, \dots , x_n) \in \{ 0, 1\}^n`: given data
* :math:`\alpha_n \in \mathbb{R}_{>0}`: a hyperparameter
* :math:`\beta_n \in \mathbb{R}_{>0}`: a hyperparameter

.. math:: \text{Beta}(p|\alpha_n,\beta_n) = \frac{1}{B(\alpha_n, \beta_n)} p^{\alpha_n} (1-p)^{\beta_n},

where the updating rule of the hyperparameters is

.. math::
    \alpha_n = \alpha_0 + \sum_{i=1}^n I \{ x_i = 1 \},\\
    \beta_n = \beta_0 + \sum_{i=1}^n I \{ x_i = 0 \}.

The predictive distribution is as follows:

* :math:`x_n \in \{ 0, 1\}`: a new data point
* :math:`\alpha_n \in \mathbb{R}_{>0}`: the hyperparameter of the posterior
* :math:`\beta_n \in \mathbb{R}_{>0}`: the hyperparameter of the posterior

.. math::
    p(x|\alpha_n, \beta_n) = \begin{cases}
    \frac{\alpha_n}{\alpha_n + \beta_n}, & x = 1,\\
    \frac{\beta_n}{\alpha_n + \beta_n}, & x = 0.
    \end{cases}
"""
```

## 例2

```python
    def set_h_params(self,h_alpha,h_beta):
        """Set the hyperparameters of the prior distribution.
        
        Parameters
        ----------
        h_alpha : float
            A positive real number
        h_beta : float
            A positive real number
        """
```

## 定型文（各メソッドの記述）

```python
class GenModel(base.Generative):
    """The stochastic data generative model and the prior distribution
```

```python
    def set_h_params(self):
        """Set the hyperparameters of the prior distribution.
```

```python
    def get_h_params(self):
        """Get the hyperparameters of the prior distribution.
        
        Returns
        -------
        h_params : {str:float, np.ndarray}
            * ``"xxx"`` : the value of ``self.xxx``
```

```python
    def gen_params(self):
        """Generate the parameter from the prior distribution.
        
        To confirm the generated vaules, use `self.get_params()`.
        """
```

```python
    def set_params(self):
        """Set the parameter of the sthocastic data generative model.
```

```python
    def get_params(self):
        """Get the parameter of the sthocastic data generative model.

        Returns
        -------
        params : {str:float, numpy.ndarray}
            * ``"xxx"`` : the value of ``self.xxx``
```

```python
    def gen_sample(self,sample_size):
        """Generate a sample from the stochastic data generative model.

        Parameters
        ----------
        sample_size : int
            A positive integer

        Returns
        -------
        x : numpy ndarray
            2-dimensional array whose shape is ``(xxx,yyy)`` 
            Its elements are real numbers.
```

```python
    def save_sample(self,filename,sample_size):
        """Save the generated sample as NumPy ``.npz`` format.

        It is saved as a NpzFile with keyword: \"xxx\", \"xxx\".
        
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
```

```python
    def visualize_model(self,sample_size=100):
        """Visualize the stochastic data generative model and generated samples.

        Parameters
        ----------
        sample_size : int, optional
            A positive integer, by default 100
        
        Examples
        --------
        >>> from bayesml import xxx
```

```python
class LearnModel(base.Posterior,base.PredictiveMixin):
    """The posterior distribution and the predictive distribution.
```

```python
    def set_h0_params(self):
        """Set the hyperparameters of the prior distribution.
```

```python
    def get_h0_params(self):
        """Get the hyperparameters of the prior distribution.

        Returns
        -------
        h0_params : dict of {str: numpy.ndarray}
            * ``"xxx"`` : the value of ``self.xxx``
```

```python
    def set_hn_params(self):
        """Set the hyperparameter of the posterior distribution.
```

```python
    def get_hn_params(self):
        """Get the hyperparameters of the posterior distribution.

        Returns
        -------
        hn_params : dict of {str: numpy.ndarray}
            * ``"xxx"`` : the value of ``self.xxx``
```

```python
    def update_posterior(self):
        """Update the the posterior distribution using traning data.
```

```python
    def estimate_params(self,loss="squared"):
        """Estimate the parameter under the given criterion.

        Note that the criterion is applied to estimating 
        ``xxx``, ``xxx`` and ``xxx`` independently.
        Therefore, a tuple of xxx, xxx and xxx will be returned when loss=\"xxx\"

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"xxx\".
            This function supports \"xxx\", \"xxx\", and \"xxx\".

        Returns
        -------
        Estimates : a tuple of {numpy ndarray, float, None, or rv_frozen}
            * ``xxx`` : the estimate for xxx
            The estimated values under the given loss function. 
            If it is not exist, `np.nan` will be returned.
            If the loss function is \"KL\", the posterior distribution itself 
            will be returned as rv_frozen object of scipy.stats.

        See Also
        --------
        scipy.stats.rv_continuous
        scipy.stats.rv_discrete
        """
```

```python
    def visualize_posterior(self):
        """Visualize the posterior distribution for the parameter.
        
        Examples
        --------
        >>> from bayesml import xxx
```

```python
    def get_p_params(self):
        """Get the parameters of the predictive distribution.

        Returns
        -------
        p_params : dict of {str: numpy.ndarray}
            * ``"xxx"`` : the value of ``self.xxx``
```

```python
    def make_prediction(self,loss="squared"):
        """Predict a new data point under the given criterion.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"xxx\".
            This function supports \"xxx\" and \"xxx\".

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
```

```python
    def pred_and_update(self):
        """Predict a new data point and update the posterior sequentially.

        h0_params will be overwritten by current hn_params 
        before updating hn_params by x.
        
        Parameters
        ----------

        Returns
        -------
        predicted_value : {float, numpy.ndarray}
            The predicted value under the given loss function. 
        """
```

```python
    def estimate_latent_vars(self,x,loss=''):
        """Estimate latent variables under the given criterion.

        Note that the criterion is independently applied to each data point.

        Parameters
        ----------
        loss : str, optional
            Loss function underlying the Bayes risk function, by default \"xxx\".
            This function supports \"xxx\", \"xxx\", and \"xxx\".

        Returns
        -------
        estimates : numpy.ndarray
            The estimated values under the given loss function. 
            If the loss function is \"xxx\", the posterior distribution will be returned 
            as a numpy.ndarray whose elements consist of occurence probabilities.
        """
```

```python
    def estimate_latent_vars_and_update(self):
        """Estimate latent variables and update the posterior sequentially.

        h0_params will be overwritten by current hn_params 
        before updating hn_params by x
        
        Parameters
        ----------

        Returns
        -------
        predicted_value : numpy.ndarray
            The estimated values under the given loss function. 
        """
```

## 定型文（入力値の条件の記述）

### スカラー

#### 実数

```
    xxx : float, optional
        A real number, by default 0.0
```

#### 非負の実数

```
    xxx : float, optional
        A non-negative real number, by default 1.0
```

#### 正の実数

```
    xxx : float, optional
        A positive real number, by default 1.0
```

#### 0以上1以下の実数

```
    xxx : float, optional
        A real number in :math:`[0, 1]`, by default 1/2
```

#### ある値以上の実数

```
    xxx : float, optional
        A real number greater than yyy, by default yyy+1
```

#### 整数

```
    xxx : int, optional
        An integer, by default 0
```

#### 非負の整数

```
    xxx : int, optional
        A non-negative integer, by default 1
```

#### 正の整数

```
    xxx : int, optional
        A positive integer, by default 1
```

#### 実数の組

```
    xxx : float or numpy.ndarray, optional
        Real numbers, 
        by default [0.0, 0.0, ... , 0.0]
        If a single real number is input, it will be broadcasted.
```

#### 非負の実数の組

```
    xxx : float or numpy.ndarray, optional
        Non-negative real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
```

#### 正の実数の組

```
    xxx : float or numpy.ndarray, optional
        Positive real numbers, 
        by default [1.0, 1.0, ... , 1.0]
        If a single real number is input, it will be broadcasted.
```

#### 0以上1以下の実数の組

```
    xxx : float or numpy.ndarray, optional
        Real numbers in :math:`[0, 1]`, 
        by default [1/2, 1/2, ... , 1/2]
        If a single real number is input, it will be broadcasted.
```

#### ある値以上の実数の組

```
    xxx : float or numpy.ndarray, optional
        Real numbers greater than yyy, by default zzz
```

#### 整数の組

```
    xxx : int or numpy.ndarray, optional
        Integers, 
        by default [0, 0, ... , 0]
        If a single integer is input, it will be broadcasted.
```

#### 非負の整数の組

```
    xxx : int or numpy.ndarray, optional
        Non-negative integers, 
        by default [1, 1, ... , 1]
        If a single integer is input, it will be broadcasted.
```

#### 正の整数の組

```
    xxx : int or numpy.ndarray, optional
        Positive integers, 
        by default [1, 1, ... , 1]
        If a single integer is input, it will be broadcasted.
```

### ベクトル

#### 実数のベクトル

```
    xxx : numpy.ndarray, optional
        A vector of real numbers, 
        by default [0.0, 0.0, ... , 0.0]
```

#### 非負の実数のベクトル

```
    xxx : numpy.ndarray, optional
        A vector of non-negative real numbers, 
        by default [1/2, 1/2, ... , 1/2]
```

#### 正の実数のベクトル

```
    xxx : numpy.ndarray, optional
        A vector of positive real numbers, 
        by default [1/2, 1/2, ... , 1/2]
```

#### 和が1となるベクトル

```
    xxx : numpy.ndarray, optional
        A vector of real numbers in :math:`[0, 1]`, 
        by default [1/yyy, 1/yyy, ... , 1/yyy]
        Sum of its elements must be 1.0.
```

#### 実数のベクトルの組

```
    xxx : numpy.ndarray, optional
        Vectors of real numbers, 
        by default zero vectors
        If a single vector is input, will be broadcasted.
```

#### 非負の実数のベクトルの組

```
    xxx : numpy.ndarray, optional
        Vectors of non-negative numbers, 
        by default vectors whose elements are all 1.0
        If a single vector is input, will be broadcasted.
```

#### 正の実数のベクトルの組

```
    xxx : numpy.ndarray, optional
        Vectors of positive numbers, 
        by default vectors whose elements are all 1.0
        If a single vector is input, will be broadcasted.
```

#### 和が1となるベクトルの組

```
    xxx : numpy.ndarray, optional
        Vectors of real numbers in :math:`[0, 1]`, 
        by default vectors such as [1/yyy, 1/yyy, ... , 1/yyy]
        Sum of the elements of each vector must be 1.0.
        If a single vector is input, will be broadcasted.
```

### 行列

#### 正定値対称行列

```
    xxx : numpy.ndarray, optional
        A positive definite symetric matrix, 
        by default the identity matrix
```

#### 各行の和が1の行列

```
    xxx : numpy.ndarray, optional
        A matrix of real numbers in :math:`[0, 1]`, 
        by default a matrix obtained by stacking [1/yyy, 1/yyy, ... , 1/yyy]
        Sum of the elements of each row vector must be 1.0.
        If a single vector is input, will be broadcasted.
```

#### 正定値対称行列の組

```
    xxx : numpy.ndarray, optional
        Positive definite symetric matrices, 
        by default the identity matrices
        If a single matrix is input, it will be broadcasted.
```
