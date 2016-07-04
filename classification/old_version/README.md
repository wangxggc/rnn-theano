# rnn-theano
使用Theano实现的一些RNN代码，包括最基本的RNN，LSTM，以及部分最新论文MLSTM等，仅供学习交流使用。

代码风格仿照：https://github.com/dennybritz/rnn-tutorial-gru-lstm

WByW的论文为：[REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION](http://arxiv.org/pdf/1509.06664.pdf)

MLSTM的论文为：[Learning Natural Language Inference with LSTM](http://arxiv.org/pdf/1512.08849.pdf)

工具默认使用**字**作为特征，编码为gb18030，若使用词作为特征，句子间用空格隔开，可以到utils.py中修改源码，具体位置为**47, 48**行。

##Train

**使用方法说明：**
	
	THEANO_FLAGS='device=gpu,cxx=/usr/bin/g++,floatX=float32,force_device=True,cuda.root=/usr/local/cuda' python train.py

**配置，打开train.py**

	MODEL      = LSTMMLstm
	DATA_DIR   = ""
	DATA_TRAIN = DATA_DIR + "/train.gb18030"
	DATA_VALID = DATA_DIR + "/validate.gb18030"
	DATA_TEST  = DATA_DIR + "/test.gb18030"
	DATA_DIC   = DATA_DIR + "/dic_10000.gb18030"
	NEPOCH       = 10
	learningRate = 0.001
	decay        = 0.95
	decay_delt   = 0.02
	batchSize    = 2048
	senLen       = 30
	hiddenDim    = 300
	DEBUG        = False

	输入数据为四个文件，数据格式为有tab隔开的两句话,以及句子的label，如：句子1\t句子2\tlabel分类时，可以将一句话重复两次，即句子1\t句子1\tlabel

	DATA_TRAIN：	为训练文件，
	DATA_VALID：	验证集，格式同DATA_TRAIN
	DATA_TEST：	测试集，格式同DATA_TRAIN
	DATA_DIC：		词典
	NEPOCH：		迭代次数
	learngingRate：学习率
	decay：		RmsProp的decay
	batchSize：	随即梯度下降的batch大小
	senLen：		RNN的递归层数
	hiddenDim：	隐层大小


##Predict

**使用方法**

	THEANO_FLAGS='device=gpu,cxx=/usr/bin/g++,floatX=float32,force_device=True,cuda.root=/usr/local/cuda' python predict.py

**配置，打开predict.py**

	DATA_PREDICT_QA = '' + DIR + '/qa.label'
	DATA_RESULT_QA  = '' + DIR + '/qa.result'
	DATA_DIC        = '' + DIR + '/dic.gb18030'
	
	classDim   = 5
	batchSize  = 10000
	senLen     = 15
	hiddenDim  = 512
	
	DEBUG      = True

	DATA_PREDICT_QA：待预测数据，格式与DATA_TRAIN相同
