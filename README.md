# Language_Model
アルバイトの一環で実装したLSTM（GRU）言語モデルです。kanohkさんのプログラム（https://github.com/kanoh-k/WikiNLP/blob/master/rnnlm.ipynb)を基に構築させていただきました。
数字などの特殊文字は事前に除外した言語モデルの構築、およびパープレキシティの評価をpytorchで実装しました。
70~72行の'self.corpus_files','self.valid_files','self.test_files'に学習、開発、テストデータの各パスを入れることでモデルを走らせることが出来ます。
モデルの各種パラメータの初期値は以下の通りです。
・Embeddingの次元数：200次元
・隠れ層：200次元2層
・コスト関数：クロスエントロピー
・ドロップアウト率：0.3
・バッチサイズ：２０
学習の工夫としてGradient clipping, Early stoppingを導入しています。
加えて、264行の’LSTM’の部分を’GRU’に書き換えることでGRU言語モデルを走らせられるようになっています。
また、プログラムの最後にて、学習した言語モデルを利用して文章を自動生成するようになっています。
