<img src="../logos/BayesML_logo.png" width="200">

# 変分ベイズ法実装手順

<div style="text-align:right">
作成：中原
</div>

生成モデルの実装なども含めた全体的な実装の流れについては[development_flow.md](./development_flow.md)を参照．ここでは，変分ベイズ法による学習モデルの事後分布更新関数の実装の流れを紹介する．

以下では，未知パラメータ$\theta$, $\eta$を持つ確率モデル$p(x | \theta, \eta)$の近似事後分布$q(\theta, \eta) = q(\theta) q(\eta)$の更新式の実装を例に話を進める．また，事前分布，変分分布はそれぞれ$p(\theta | \alpha_0)$, $p(\eta | \beta_0)$, $q(\theta | \alpha_n)$, $p(\eta | \beta_n)$のように，同じ形でパラメトライズされた分布として表現できるものとする．この形式に沿った実装例としては[_gaussianmixture.py](../../bayesml/gaussianmixture/_gaussianmixture.py)も参照のこと．

## 1. 変分事後分布更新式，変分下界計算式の整理

まずは変分事後分布更新式と変分下界計算式を整理する．特に，更新式，計算式の中で何度も登場する量を把握しておき，そういった量については同じ計算を繰り返すのではなく，計算結果を何らかの変数に保持して使いまわすようにする．多くの場合は，変分分布の期待値$\mathbb{E}_{q(\theta)}[\theta]$や変分分布の対数の期待値$\mathbb{E}_{q(\theta)}[ \ln \theta]$，規格化定数などが様々な箇所で利用されるため，それらを保持しておくとよい．

## 2. 変数の準備

LearnModelの`__init__`関数に以下の変数とその初期化関数を用意する．

* ハイパーパラメータ初期値`h0_params`
* ハイパーパラメータ更新値`hn_params`
* 前節で確認した，各変分分布の期待値などの量
  * 一般的な呼び方ではないが，ここではこれを変分分布の特徴量と呼ぶものとする．
  * これらについては，`_`で始まるプライベートな変数とする．
* 変分下界の各項
  * 変分下界の値を保持する変数はアンダースコア無しの`vl`とし，各項を保持する変数は`_`から始める．

今回の例では以下のような変数を用意するということである．

```python
self.h0_alpha = 0.0
self.h0_beta = 0.0
self.hn_alpha = 0.0
self.hn_beta = 0.0
self._e_theta = 0.0
self._e_ln_theta = 0.0
self._ln_c_hn_alpha = 0.0
self._e_eta = 0.0
self._e_ln_eta = 0.0
self._ln_c_hn_beta = 0.0
self.vl = 0.0
self._vl_p_theta = 0.0
self._vl_p_eta = 0.0
self._vl_p_x = 0.0
self._vl_q_theta = 0.0
self._vl_q_eta = 0.0
```

このほかにも，データの算術平均などの統計量を予め計算しておくと便利な場合がある．そういった変数も`_`から始まる変数として適宜定義すべきである．

## 3. 関数の実装

実装すべき関数は大きく5つある．事後分布更新時に実行される順に以下の5つである．

* 変分ベイズの全体構造を表す関数（各変分分布の更新式を呼ぶ側の関数．反復終了判定などもここに書く）
* `hn_params`のランダム初期値生成関数（`h0_params`の初期値ではないことに注意）
* 各`hn_params`の更新関数
* 各変分分布の特徴量計算関数
* 変分下界計算関数

実行されるのは上記の順だが，実装する順は上記とは異なり，以下のようにするとよい．

1. 変分ベイズの全体構造を表す関数
2. 各変分分布の特徴量計算関数
3. 変分下界計算関数
4. 各`hn_params`の更新関数
5. `hn_params`のランダム初期値生成関数

それぞれについて実装の注意点を以下で述べる．

### 3.1. 変分ベイズの全体構造を表す関数

この関数が`update_posterior()`に相当する．以下や[_gaussianmixture.py](../../bayesml/gaussianmixture/_gaussianmixture.py)のものをコピペして適宜修正してくれればよい．

```python
    def update_posterior(
            self,
            x,
            max_itr=100,
            num_init=10,
            tolerance=1.0E-8,
            ):
        x = self._check_sampla(x) # 別途用意しておいた入力値チェック関数

        # 変分下界やハイパーパラメータの暫定値を保持するための変数
        tmp_vl = 0.0
        tmp_alpha = 0.0
        tmp_beta = 0.0

        convergence_flag = True # 様々な初期値のうち1回でも変分下界が収束したらFalseになる．
        for i in range(num_init):
            self.reset_hn_params()
            self._init_q_beta() # この例ではq_alphaの更新から始めるので，q_betaの法をランダムに初期化しておく
            self._calc_vl()
            print(f'\r{i}. VL: {self.vl}',end='')
            for t in range(max_itr):
                vl_before = self.vl
                self._update_q_alpha()
                self._update_q_beta()
                self._calc_vl()
                print(f'\r{i}. VL: {self.vl} t={t} ',end='')
                if np.abs((self.vl-vl_before)/vl_before) < tolerance: # 収束判定は変分下界の相対誤差で行う
                    convergence_flag = False
                    print(f'(converged)',end='')
                    break
            if i==0 or self.vl > tmp_vl: # 変分下界の最大値が更新されたらその値を保持
                print('*')
                tmp_vl = self.vl
                tmp_alpha = self.hn_alpha
                tmp_beta = self.hn_beta
            else:
                print('')
        if convergence_flag:
            warnings.warn("Algorithm has not converged even once.",ResultWarning)
        
        # 最終的に変分下界が最も大きかったハイパーパラメータを採用する．
        self.hn_alpha = tmp_alpha
        self.hn_beta = tmp_beta
        self._calc_q_theta_features()
        self._calc_q_eta_features()
        return self
```

### 3.2. 各変分分布の特徴量計算関数

* 関数名は`_calc_q_theta_features()`などとする．
* numpy配列の値を更新する場合，左辺側の配列に`[:]`をつけて配列全体のスライスに新たな値を代入するようにする．そうしないとどんどん新しいメモリ空間が確保されてしまい，速度も低下する．（[_gaussianmixture.py](../../bayesml/gaussianmixture/_gaussianmixture.py)も参照のこと）
* これらは`reset_hn_params()`の最後にも呼び出すようにする．

```python
def _calc_q_theta_features(self):
    self._e_theta = 
    self._e_ln_theta = 
    self._ln_c_hn_alpha = 

def _calc_q_eta_features(self):
    self._e_eta = 
    self._e_ln_eta = 
    self._ln_c_hn_eta = 
```

### 3.3. 変分下界計算関数

通常，変分下界はいくつかの項に分解できる．各項を計算する関数を順に実装していくとよい．今回の例であれば，
$$
\begin{align}
\mathrm{VL}(q) = \mathbb{E}[\ln p(\theta)] + \mathbb{E}[\ln p(\eta)] + \mathbb{E}[\ln p(x | \theta, \eta)] - \mathbb{E}[\ln q(\theta)] - \mathbb{E}[\ln q(\eta)]
\end{align}
$$
と書ける（ただし，上記の期待値は変分分布$q(\theta, \eta)$でとっている）．

その他の注意事項は以下の通りである．

* 関数名は`_calc_vl`とする．
* $\mathbb{E}[\ln q(\theta)]$などは変分分布の負のエントロピーに相当するので，scipyの様々な分布のエントロピー計算関数がそのまま使える場合がある．そういった関数は積極的に利用したほうがよい．
* 事前分布の規格化定数などが必要になる場合もある．その場合は別途`_calc_prior_features`などの関数を用意するとよい（多くの場合`_calc_q_theta_features`などを書き換えればすぐに実装できる）．
* 変分下界の計算はどんなに複雑であっても最後は何らかの実数が得られるはずである．パラメータ，ハイパーパラメータの行列やベクトルの次元を変えて，常に実数が得られることをこの時点で確認しておく．

```python
    def _calc_vl(self):
        # E[ln p(x|theta, eta)]
        self._vl_p_x = 

        # E[ln p(theta)]
        self._vl_p_theta = 

        # E[ln p(eta)]
        self._vl_p_eta = 

        # -E[ln q(theta)]
        self._vl_q_theta = 

        # -E[ln q(eta)]
        self._vl_q_eta = 

        self.vl = (self._vl_p_x
                   + self._vl_p_theta
                   + self._vl_p_eta
                   + self._vl_q_theta
                   + self._vl_q_eta)
```

### 3.4. 各`hn_params`の更新関数

* 関数名は`_update_q_theta`などとする．
* 更新直後に必ず`_calc_q_theta_features()`などを呼び，分布の特徴量を最新の状態に保つようにする．
* 既に実装してある変分下界の計算式を使って，各変分分布の更新後に必ず変分下界が上昇することを確認する．
  * 更新式が一つ実装できる毎に必ず確認する．
  * モデルの次数などの定数を変えても変分下界が上昇するか確認する．
  * 極端な初期値を与えても変分下界が上昇するか確認する．
  * 実装中のモデルと異なる生成モデルから生成したデータに対しても変分下界が上昇することを確認する（例えば[0,1]区間上の一様乱数から発生させたデータを混合正規モデルに食わせるなど）．
  * とにかく色んな条件で変分下界の上昇を確認する．
* numpy配列の値を更新する場合左辺側の配列に`[:]`をつけて配列全体のスライスに新たな値を代入するようにする．そうしないとどんどん新しいメモリ空間が確保されてしまい，速度も低下する．（[_gaussianmixture.py](../../bayesml/gaussianmixture/_gaussianmixture.py)も参照のこと）

```python
def _update_q_theta(self):
    self.hn_alpha = 
    self._calc_q_theta_features()

def _update_q_eta(self):
    self.hn_beta = 
    self._calc_q_eta_features()
```

変分ベイズの実装というと真っ先に思い浮かべるのはこの関数の実装だろう．しかし，実装すべき関数全体からするとむしろごく一部に過ぎないことがわかる．

### 3.5. `hn_params`のランダム初期値生成関数

変分ベイズは反復更新型のアルゴリズムなので，何らかの初期値を与えて更新式を回し始める必要がある．この関数については，モデル毎に適切に設計する必要があるため，別途，中原に相談してほしい．一般的な注意としては以下がある．

* 全てのハイパーパラメータに初期値を設定しなければならないわけではない．一つ目の更新式を計算するために既知としておかなければならない量だけに初期値を与えれば十分である．
  * 今回であれば，最初に$q(\theta)$の更新を行うために$q(\eta)$のハイパーパラメータだけに初期値を与えておくなど．
* なるべく未知パラメータの空間をまんべんなく覆うように初期値を設定する．
  * 混合正規分布の場合なら，各混合要素の平均や分散に初期値を与えるのではなく，各データ点の負担率$q(\bm z)$に初期値を与えたほうがよい．
