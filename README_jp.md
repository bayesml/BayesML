<!--
Document Author
Yuta Nakahara <yuta.nakahara@aoni.waseda.jp>
-->

---

<div align="center">
<b>メタツリーモデルに対するアルゴリズムがAISTATS 2025に採択！詳細は<a href="https://bayesml.github.io/BayesML/examples/metatree_prediction_interval.html">こちら</a>！</b>
</div>

---

<img src="./doc/logos/BayesML_logo.png" width="600">

## 目的

BayesMLは，ベイズ統計学やベイズ決定理論に基づく機械学習の研究，教育，活用を促進し，社会に広く貢献することを目的としたライブラリです．

## 特徴

BayesMLは以下の特徴を持っています．

* データからの学習による事後分布の更新と，ベイズ基準のもとでの最適推定値の出力というベイズ統計学，ベイズ決定理論の理念がライブラリの構造に反映されています．
* 学習アルゴリズムの多くはデータ生成確率モデルと事前分布の共役性を効果的に用いているため，MCMC法などの汎用的なベイズ学習アルゴリズムと比べて非常に高速で，オンライン学習にも適しています．
* 2~3次元空間上で，データ生成確率モデル，生成された人工データ，学習された事後分布を可視化するメソッドが全てのパッケージに備わっています．そのため，人工データの生成と学習を通じて確率モデル，アルゴリズムの特性を効果的に把握することができます．

詳細は[Webサイト](https://bayesml.github.io/BayesML/ "BayesML's Documentation")をご覧ください．

## ニュース

* メタツリーモデルに対するアルゴリズムがAISTATS 2025に採択されました．サンプルコードは[こちら](https://bayesml.github.io/BayesML/examples/metatree_prediction_interval.html)をご覧ください．

## インストール

以下のコマンドによりインストール可能です．

``` bash
pip install bayesml
```

BayesMLの実行には以下が必要です．

* Python (>= 3.7)
* NumPy (>= 1.20)
* SciPy (>= 1.7)
* MatplotLib (>= 3.5)
* Scikit-learn (>= 1.1)

## 実行例

ベルヌーイ分布に従うデータの生成と，そのデータから学習についての例を示します．

まず，データ生成モデルを作成します．ここでは1の生成確率を表すパラメータ`theta`を0.7としました．

``` python
from bayesml import bernoulli

gen_model = bernoulli.GenModel(theta=0.7)
```

以下のメソッドによって作成されたモデルの特徴を可視化できます．

``` python
gen_model.visualize_model()
```

>theta:0.7  
>x0:[1 1 1 0 1 1 1 0 1 1 1 1 1 1 0 1 1 1 0 1]  
>x1:[1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 0]  
>x2:[1 0 1 1 0 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1]  
>x3:[1 1 1 0 1 1 0 1 0 0 0 0 1 0 1 1 1 1 1 1]  
>x4:[0 0 1 0 0 0 1 1 1 1 1 1 1 1 0 0 1 1 1 1]  
>![bernoulli_example1](./doc/images/README_ex_img1.png)

1の出現頻度が`theta=0.7`程度であることを確認したら，サンプルを生成し変数`x`に保存します．

``` python
x = gen_model.gen_sample(sample_size=20)
```

次に，事後分布学習用のモデルを作成します．

``` python
learn_model = bernoulli.LearnModel()
```

事後分布を可視化するメソッドも存在します．（ここではまだデータからの学習を行っていないので事前分布が表示されます．）

``` python
learn_model.visualize_posterior()
```

>![bernoulli_example2](./doc/images/README_ex_img2.png)

学習を行うことによって事後分布の密度が真のパラメータである`theta=0.7`の周辺に集中することがわかります．

``` python
learn_model.update_posterior(x)
learn_model.visualize_posterior()
```

>![bernoulli_example3](./doc/images/README_ex_img3.png)

オプションとして損失関数を指定することで，その損失関数に基づくベイズリスク関数を評価基準とした最適な推定値が定まります．

``` python
print(learn_model.estimate_params(loss='squared'))
print(learn_model.estimate_params(loss='abs'))
print(learn_model.estimate_params(loss='0-1'))
```

>0.7380952380952381  
>0.7457656349087012  
>0.7631578947368421  

損失関数の設定が異なると，そのもとでの最適な推定値も異なることがわかります．

## パッケージ一覧

現在，以下のモデルに関するパッケージが利用可能です．本ライブラリでは，データ生成確率モデル，事前分布，事後分布（または近似事後分布），予測分布（または近似予測分布）を合わせてモデルと呼んでいます．

* [ベルヌーイモデル](https://bayesml.github.io/BayesML/bayesml.bernoulli.html "Bayesml Bernoulli Model")
* [カテゴリカルモデル](https://bayesml.github.io/BayesML/bayesml.categorical.html "BayesML Categorical Model")
* [ポアソンモデル](https://bayesml.github.io/BayesML/bayesml.poisson.html "BayesML Poisson Model")
* [正規モデル](https://bayesml.github.io/BayesML/bayesml.normal.html "BayesML Normal Model")
* [多変量正規モデル](https://bayesml.github.io/BayesML/bayesml.multivariate_normal.html "BayesML Multivariate Normal Model")
* [指数モデル](https://bayesml.github.io/BayesML/bayesml.exponential.html "BayesML Exponential Model")
* [混合正規モデル](https://bayesml.github.io/BayesML/bayesml.gaussianmixture.html "BayesML Gaussian Mixture Model")
* [線形回帰モデル](https://bayesml.github.io/BayesML/bayesml.linearregression.html "BayesML Lenear Regression Model")
* [メタツリーモデル](https://bayesml.github.io/BayesML/bayesml.metatree.html "BayesML Meta-tree Model")
* [自己回帰モデル](https://bayesml.github.io/BayesML/bayesml.autoregressive.html "BayesML Autoregressive Model")
* [隠れマルコフモデル](https://bayesml.github.io/BayesML/bayesml.hiddenmarkovnormal.html "BayesML Hidden Markov Normal Model")
* [文脈木モデル](https://bayesml.github.io/BayesML/bayesml.contexttree.html "BayesML Context Tree Model")

また，今後はより複雑な階層的モデルを取り扱うパッケージを追加していく予定です．

## コントリビューションの方法

BayesMLへのコントリビューションを考えてくださってありがとうございます．詳細については[こちら](./CONTRIBUTING_jp.md)をご覧ください．

## 参照方法

学術的な成果にBayesMLをご利用いただく際は以下の文献参照をお示しいただければ幸いです．

プレーンテキスト

```
Y. Nakahara, N. Ichijo, K. Shimada, Y. Iikubo, 
S. Saito, K. Kazama, T. Matsushima, BayesML Developers, ``BayesML,'' 
Python package version 0.3.0, 
[Online] https://github.com/bayesml/BayesML
```

BibTeX

``` bibtex
@misc{bayesml,
  author = {Nakahara, Yuta and Ichijo, Naoki and Shimada, Koshi and
            Iikubo, Yuji and Saito, Shota and Kazama, Koki and
            Matsushima, Toshiyasu and {BayesML Developers}},
  title = {{BayesML}},
  howpublished = {Python package version 0.3.0},
  note = {\url{https://github.com/bayesml/BayesML}},
  year = {2025}
}
```
