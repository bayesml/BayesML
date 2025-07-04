<img src="../logos/BayesML_logo.svg" width="200">

# 翻訳手順

<div style="text-align:right">
作成：中原
</div>

## 準備

1. pandocのインストール
   1. 公式サイト https://pandoc.org/installing.html に従い，インストール．
   2. 環境変数の設定を更新するために再起動が必要かも
2. Poeditのインストール
   1. 公式サイト https://poedit.net に従い，インストール．
3. 仮想環境の構築
   1. Anaconda promptで`conda create -n 環境名（bayesml_devなど） python`と入力し，実行
   2. `conda activate 環境名`で仮想環境を起動
   3. `cd`コマンドなどを用いてBayesMLフォルダ（setup.pyの存在する場所）に移動
   4. `pip install -e .`と入力し，実行（ローカルから編集モードでbayesmlがインストールされる）
   5. `pip install sphinx sphinx-book-theme myst-parser sphinx-intl numpydoc nbsphinx notebook`と入力し，実行（webサイト作成関連のライブラリがインストールされる）

## 基本的な流れ

以下では，BayesML/docフォルダ（make.batの存在する場所）をコマンドラインツールの作業ディレクトリとする．

1. POTファイルの生成
   1. `make gettext`
   2. ./_build/gettextにpotファイルが生成される（POTファイルについてはGitでのバージョン管理は不要）．
2. POファイルの生成
   1. `sphinx-intl update -p _build/gettext -l en -l ja`
   2. ./locale/en, ./locale/jaにpoファイルが生成される．
3. 翻訳
   1. Poeditでpoファイルを開き，翻訳する文を選択し，翻訳結果を記入していく．先に右上の事前翻訳ボタンを使うとよいかも．
4. ビルド（macとwindowsで違う）
   1. Windows Anaconda PowerShell Promptの場合（通常のAnaconda Promptではなく，必ずAnaconda PowerShell Promptを用いること）
      1. 英語サイトのビルドコマンド`$env:SPHINX_LANGUAGE = "en"; sphinx-build -D language="en" -b html "." "../docs/en"`
      2. 日本語サイトのビルドコマンド`$env:SPHINX_LANGUAGE = "ja"; sphinx-build -D language="ja" -b html "." "../docs/ja"`
   2. Mac (zsh)の場合
      1. `make html`を実行するだけで英語と日本語両方のサイトがビルドされる
   3. ../docs/html/enに英語サイト，../docs/html/jaに日本語サイトのhtmlが生成される．
5. Gitコミット
   1. 以下は現時点では不要だが，今後gitの履歴をきれいに管理したくなったら作業手順に加えることにする．
   2. `msgcat --no-location --output-file="PO fileへのパス" "PO fileへのパス"`
   3. `msgcat --no-wrap --output-file="PO fileへのパス" "PO fileへのパス"`
   4. poファイルから不要な情報や不要な改行が削除される．

## 新しいモデルを開発するとき

各モデルのソースコードの.pyファイルは英語を主言語として管理し，docstringも英語にしておきたい．そのため，開発は日本語で行ってもよいが，developにマージする前に必ず英語化する．その際，日本語を削除してしまうのはもったいないので，Webサイトの翻訳も同時に行う．したがって，以下のような流れになる．

1. 日本語docstring作成
2. 日本語docstring退避（例えば.pyファイル全体をどこかにコピーしておくなど）
3. 英語docstring作成
4. POTファイル生成（前節1.）
5. POファイル生成（前節2.）
6. 日本語POファイル更新（退避しておいた日本語docstring利用して翻訳）

## Exampleページなどを追加するとき（Web記事のみ追加されるとき）

ExampleなどのWeb記事は日本語を主言語として管理する．最悪，日本語記事しかできていない状態でdevelopにマージしてもよい．記事作成時または後日に以下を行う

1. POTファイル生成（前節1.）
2. POファイル生成（前節2.）
3. 英語POファイルを更新（普通に翻訳）

## 参考サイト

* 全般
  * https://www.sphinx-doc.org/ja/master/usage/advanced/intl.html
  * https://dev.classmethod.jp/articles/sphinx-i18n/
* ディレクトリ構造，言語切替ボタン
  * https://dev.classmethod.jp/articles/implement-sphinx-i18n-switch-button/
* Poeditのダウンロード（公式サイト）
  * https://poedit.net
* git上で差分を確認しやすくする工夫
  * https://weseek.co.jp/tech/3677/
