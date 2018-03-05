# [Hierarchical Attention Networks for Document Classification](http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf) TensorFlow实现

## 依赖
Python3

Tensorflow 1.4+

## 数据
### 来源 
数据来自DataFoundation的一个比赛[2018“云移杯- 景区口碑评价分值预测](http://www.datafountain.cn/#/competitions/283/intro)，根据评论预测评分，可以当做文档分类任务。
### 形式
每个训练样例由(doc,label)组成，doc是评论，由sent\_num*sent\_length的矩阵组成，每行表示一句话，一句话由字组成（也就是说没有分词，基本单位是字），每个字由唯一的index表示。label是评分，由唯一的index表示。
### 预处理
1. 过滤掉非汉字字符,分句。
2. 制作char2index和label2index字典，并存储为pickle。
3. 将doc和label变成index, 然后pad,存储。

### 运行方法
在`preprocess`目录下，运行
`python preprocess.py`。

## 参数
* `embedding_size`，字向量的大小，默认800。
* `lstm_size_char`，论文中word encoder的大小，代码中实际用的是GRU。 默认300。
* `lstm_size_sent`，sentence encoder的大小，代码中实际用的是GRU。 默认300。
* `batch_size`，默认 32。
* `learning_rate`，默认 0.05。
* `epochs`，默认 10。
* `checkpoint`，从checkpoint中restore参数，默认None。
* `models_dir`，存储checkpoins的目录，默认`models/`
* `test`，如果选择，就在验证集上计算准确率。
* `predict`，如果选择，预测。
* `tiny`，如果选择，采用小规模数据集进行训练。

### 用法
训练：`python main.py`

测试：`python main.py -checkpoint models/*th_epoch_model_*.**.ckpt -test`

预测：`python main.py -checkpoint models/*th_epoch_model_*.**.ckpt -predict`