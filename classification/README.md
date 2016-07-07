#Classification Model - RNN based on theano
#使用Theano实现的RNN相关分类模型

模型MLSTM论文，[请点击](http://arxiv.org/pdf/1512.08849.pdf)

##基本配置

	使用时，打开setting.py，修改隐层数量，句子长度，batch大小，词典大小等基本参数
	
	g_max_length = 100	# 生成句子的最大长度
	g_n_epoch = 100	# 迭代次数
	g_save_epoch = 2	# 模型保存间隔
	g_learning_rate = 0.001	# 学习率 RmsProp
	g_decay = 0.95	# RmsProp中的decay
	g_batch_size = 64	# batch的大小
	g_sen_len = 15	# 句子的长度
	g_hidden_dim = 256	# 隐层的数量
	g_word_dim = 16531 + 2 # 字典大小 + 2
	G_MODEL = LSTM	# 模型，目前可用的有RNN，LSTM以及MLSTM
	
	
##使用方法

**For training,**
	
	python train.py -t data_train_file data_test_file data_dic_file data_class_weights model_save_file pre_trained_file(optional)
		data_train_file, train data
		data_test_file, test data
		data_class_weights, class weights, format: "label,weights,count,class_name"
    
**For prediction,**
    
	python train.py -p model_file dic_file predict_data_file result_save_file