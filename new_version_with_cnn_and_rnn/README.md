## 说明

文本分类，CNN和RNN模型，

`old_version_1`中得仍然可以使用（Seq2Seq在这里，没有BimSearch），

`new_version_with_cnn_and_rnn_for_classification`是一个最基本的CNN和RNN的分类实现。

其中，CNN是Kim得Sentence CNN，RNN是LSTM-Based

## CNN参数说明

    params = {
        "output_dim":2,         # 输出维度，即类别数量
        "feature_dim":-1,       # 特征数量，推荐使用“字”作为特征
        "embed_dim":64,         # Embedding维度
        # 定义卷集核，（count_of_filter, channels, kernel_height, kernel_width)
        # 具体定义可以看一下theano.tensor.nnet.conv2d函数
        "filters_shapes":[(10, 1, 2, 64), (10, 1, 3, 64), (10, 1, 5, 64), (10, 1, 4, 64)],
        "pooling":"max",        # pooling方式，“max","average"，默认max
        "hidden_dims":[256],    # hidden（MLP）隐层维度，会在前面追加[sum(count_of_filter)]
        "sen_len":30,           # 句子长度
        "batch_size":1024,      # batch大小
        "learning_rate":0.001,  # 最大学习率
        "min_learning_rate":1e-6, # 最小学习率
        "decay":0.95,           # rmr_prop参数
        "epoches":200,          # 迭代轮次
        "save_iter":2,          # 保存间隔
        "drop":True             # 是否使用drop，最后一层FC-Layer（PredictLayer）会加入Drop
    }

## RNN参数说明

    params = {
        "output_dim":2,           # 输出维度，即类别数量
        "word_dim":-1,            # 特征数量，推荐使用“字”作为特征
        "embed_dim":64,           # Embedding维度
        "hidden_dim":128,         # 隐层维度
        "sen_len":30,             # 句子长度
        "batch_size":1024,        # batch大小
        "learning_rate":0.001,    # 最大学习率
        "min_learning_rate":1e-6, # 最小学习率
        "decay":0.95,             # rms_prop参数
        "epoches":200,            # 迭代轮数
        "save_iter":2,            # 保存步长
        "bidirectional":False,    # 双向Encoder，论文我也忘记是那篇了
        "drop":True               # 是否使用drop，最后一层FC-Layer（PredictLayer）会加入Drop
    }

**数据格式**

	label\t句子，句子的特征之间使用空格隔开
	句子逻辑推理时，句子2代表第二句话

## 训练方法

使用CNN: `python cnn.py -t train_data.txt valid_data.txt dic.txt model_save_dir`

使用RNN：`python rnn.py -t train_data.txt valid_data.txt dic.txt model_save_dir`


##测试方法

使用CNN: `python cnn.py -p model_file dic.txt predict_data.txt result.txt`

使用RNN: `python rnn.py -p model_file dic.txt predict_data.txt result.txt`
