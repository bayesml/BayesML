<img src="../logos/BayesML_logo.png" width="200">

# 抽象クラス概要 Ver.5
<div style="text-align:right">
作成：中原
</div>

## `class Generative(metaclass=ABCMeta):`

データ生成観測確率モデルとその事前分布の抽象基底クラス．GenModelクラスに継承することで，以下の名前のメソッドの実装を強いる．

* `def set_h_params(self):`
  * 事前分布のハイパーパラメータを設定するためのメソッド．`c_`で始まる変数は変更しなくてよく，それに整合しない場合はエラーを返す．入力されたハイパーパラメータが理論上の仮定（分散共分散行列の正定値性等）を満たさない時はエラーを返すようにする．よく使う入力値チェックは`bayesml/_check.py`に書いておく．GenModelの`get_h_params()`，LearnModelの`get_h0_params()`, `get_hn_params()`で得られる辞書の順序と位置引数の順序を揃えるようにする．Python 3.7以降では辞書の要素の順序が保持されるようになったため，これにより`genmodel.set_h_params(*learnmodel.get_hn_params().vaules())`といったアンパック渡しの機能を活かした使い方が可能となる．
* `def get_h_params(self):`
  * 事前分布のハイパーパラメータを返すメソッド．ハイパーパラメータ名をキーとする辞書を返す．
* `def save_h_params(self):`（抽象クラスではない）
  * 事前分布のハイパーパラメータをファイルに保存するメソッド．`get_h_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def load_h_params(self):`（抽象クラスではない）
  * `save_h_params`で保存したハイパーパラメータを読み込んで`set_h_params`で設定するメソッド．`get_h_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def gen_params(self):`
  * 事前分布からデータ生成観測確率モデルのパラメータを生成するメソッド．
* `def set_params(self):`
  * データ生成観測確率モデルのパラメータを手動で設定するためのメソッド．入力されたパラメータが理論上の仮定（分散共分散行列の正定値性等）を満たさない時はエラーを返すようにする．よく使う入力値チェックは`bayesml/_check.py`に書いておく．
* `def get_params(self):`
  * データ生成観測確率モデルのパラメータを手動で設定するためのメソッド．
* `def save_params(self):`（抽象クラスではない）
  * データ生成観測確率モデルのパラメータをファイルに保存するメソッド．`get_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def load_params(self):`（抽象クラスではない）
  * `save_params`で保存したハイパーパラメータを読み込んで`set_params`で設定するメソッド．`get_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def gen_sample(self):`
  * データ生成観測確率モデルからデータ（標本）を生成し，戻り値として返すためのメソッド．
* `def save_sample(self):`
  * データ生成観測確率モデルからデータ（標本）を生成し，ファイルに保存するためのメソッド．`gen_sample()`の戻り値を[numpy.savez_compressed](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)に渡せばよい．
* `def visualize_model(self):`
  * 設定されたパラメータのデータ生成観測モデルとそこから生成されたデータを可視化し，特徴を把握するためのメソッド．（グラフ等で可視化可能な2, 3次元空間上でのみ動作すればよく，そうでない場合にはエラーを返すようにする）

## `class Posterior(metaclass=ABCMeta):`

データ生成観測確率モデルのパラメータ事後分布の抽象基底クラス．LearnModelクラスに継承することで以下のメソッドの実装を強いる．

* `def set_h0_params(self):`
  * 事後分布のハイパーパラメータの初期値を設定するためのメソッド（`reset_hn_params()`を呼ぶことで，`hn_`で始まる事後分布ハイパーパラメータや`p_`で始まる予測分布パラメータも同時に初期化する）．`c_`で始まる変数は変更しなくてよく，それに整合しない場合はエラーを返す．入力されたハイパーパラメータが理論上の仮定（分散共分散行列の正定値性等）を満たさない時はエラーを返すようにする．よく使う入力値チェックは`bayesml/_check.py`に書いておく．GenModelの`get_h_params()`，LearnModelの`get_h0_params()`, `get_hn_params()`で得られる辞書の順序と位置引数の順序を揃えるようにする．Python 3.7以降では辞書の要素の順序が保持されるようになったため，これにより`learnmodel.set_h_params(*genmodel.get_hn_params().vaules())`といったアンパック渡しの機能を活かした使い方が可能となる．
* `def get_h0_params(self):`
  * 事後分布のハイパーパラメータの初期値を返すメソッド．ハイパーパラメータ名をキーとする辞書を返す．
* `def save_h0_params(self):`（抽象クラスではない）
  * 事後分布のハイパーパラメータの初期値をファイルに保存するメソッド．`get_h0_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def load_h0_params(self):`（抽象クラスではない）
  * `save_h0_params`や`save_hn_params`で保存したハイパーパラメータを読み込んで`set_h0_params`で設定するメソッド．`save_h0_params`や`save_hn_params`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def get_hn_params(self):`
  * データに基づいて更新された事後分布のハイパーパラメータを返すメソッド．ハイパーパラメータ名をキーとする辞書を返す．
* `def set_hn_params(self):`
  * 更新後の事後分布のハイパーパラメータを直接設定するためのメソッド（`calc_pred_dist()`を用いて`p_`で始まる予測分布パラメータも同時に初期化する）．`c_`で始まる変数は変更しなくてよく，それに整合しない場合はエラーを返す．入力されたハイパーパラメータが理論上の仮定（分散共分散行列の正定値性等）を満たさない時はエラーを返すようにする．よく使う入力値チェックは`bayesml/_check.py`に書いておく．GenModelの`get_h_params()`，LearnModelの`get_h0_params()`, `get_hn_params()`で得られる辞書の順序と位置引数の順序を揃えるようにする．Python 3.7以降では辞書の要素の順序が保持されるようになったため，これにより`learnmodel.set_h_params(*genmodel.get_hn_params().vaules())`といったアンパック渡しの機能を活かした使い方が可能となる．
* `def save_hn_params(self):`（抽象クラスではない）
  * データに基づいて更新された事後分布のハイパーパラメータをファイルに保存するメソッド．`get_hn_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def load_hn_params(self):`（抽象クラスではない）
  * `save_hn_params`で保存したハイパーパラメータを読み込んで`set_hn_params`で設定するメソッド．`get_hn_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def reset_hn_params(self):`（抽象クラスではない）
  * 更新後の事後分布ハイパーパラメータ（`hn_`で始まるハイパーパラメータの値）を初期値（`h0_`で始まるハイパーパラメータの値）に設定し直すメソッド．`get_h0_params(self):`, `set_hn_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def overwrite_h0_params(self):`（抽象クラスではない）
  * 事後分布のハイパーパラメータの初期値（`h0_`で始まるハイパーパラメータの値）を更新後の事後分布ハイパーパラメータ（`hn_`で始まるハイパーパラメータの値）で上書きするメソッド．`get_hn_params(self):`, `set_h0_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def update_posterior(self):`
  * データに基づいて事後分布のハイパーパラメータを更新するためのメソッド．in-placeな処理を心がける．データは引数として渡し，変数として保持しない．データが理論上の仮定（整数かどうか等）を満たさない時はエラーを返すようにする．よく使う入力値チェックは`bayesml/_check.py`に書いておく．
* `def estimate_params(self):`
  * データに基づいて更新された事後分布を用いてパラメータを推定するためのメソッド．推定の評価基準をオプション`loss="criteria"`として指定することで，出力値が変わる．事後分布の種類によってはmodeが存在しない場合などもあるので，そういった場合には`None`を返し，警告を表示するようにする．
* `def visualize_posterior(self):`
  * 事後分布を可視化し，特徴を把握するためのメソッド．（グラフ等で可視化可能な2, 3次元空間上でのみ動作すればよく，そうでない場合にはエラーを返すようにする）

## `class PredictiveMixin(metaclass=ABCMeta):`

予測分布の抽象基底クラス．LearnModelクラスに継承することで以下のメソッドの実装を強いる．事後分布の計算は可能だが，予測分布の計算はできない場合もあるので，LearnModelクラスには基本的にPosteriorクラスを継承し，予測分布の計算ができる場合にはPredictiveMixinも継承（多重継承）するようにする．（Mixinというのは多重継承の問題を回避するための概念だが，詳細は中原も勉強中．）

* `def get_p_params(self):`
  * 予測分布のパラメータを返すメソッド．パラメータ名をキーとする辞書を返す．
<!-- * `def set_p_params(self):`
  * 予測分布のパラメータを直接設定するためのメソッド．入力されたパラメータが理論上の仮定（分散共分散行列の正定値性等）を満たさない時はエラーを返すようにする．よく使う入力値チェックは`bayesml/_check.py`に書いておく．
* `def save_p_params(self):`（抽象クラスではない）
  * 予測分布のパラメータをファイルに保存するメソッド．`get_p_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない．
* `def load_p_params(self):`（抽象クラスではない）
  * `save_p_params`で保存したハイパーパラメータを読み込んで`set_p_params`で設定するメソッド．`get_p_params(self):`さえ正しく実装されていれば汎用的に機能するよう`bayesml/base.py`に実装済みなので，基本的に個別のモデルでオーバーライドする必要はない． -->
* `def calc_pred_dist(self):`
  * 事後分布のハイパーパラメータと新規データから予測分布のパラメータを計算するためのメソッド．in-placeな処理を心がける．新規データが理論上の仮定（整数かどうか等）を満たさない時はエラーを返すようにする．よく使う入力値チェックは`bayesml/_check.py`に書いておく．
* `def make_prediction(self):`
  * 予測分布を用いて新規データを予測するためのメソッド．予測の評価基準をオプション`loss="criteria"`として指定することで，出力値が変わる．予測分布の種類によってはmodeが存在しない場合などもあるので，そういった場合には`None`を返し，警告を表示するようにする．
* `def pred_and_update(self):`
  * `calc_pred_dist`, `make_prediction`, `update_posterior`を順に行うメソッド．ベイズ統計，ベイズ決定理論の一つの特徴であるオンライン学習を実現するためのメソッド．
