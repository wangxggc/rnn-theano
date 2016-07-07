#Sequence to sequence - theano

##基本配置

	使用时，打开setting.py，修改隐层数量，句子长度，batch大小，词典大小等基本参数
	
	g_max_length = 100	# 生成句子的最大长度
	g_n_epoch = 100	# 迭代次数
	g_save_epoch = 2	# 模型保存间隔
	g_learning_rate = 0.001	# 学习率 RmsProp
	g_decay = 0.95	# RmsProp中的decsy
	g_batch_size = 64	# batch的大小
	g_sen_len = 15	# 句子的长度
	g_hidden_dim = 256	# 隐层的数量
	g_word_dim = 16531 + 2 # 字典大小 + 2
	G_MODEL = LSTM	# 模型，目前可用的只有LSTM和RNN，使用RNN的时候，train.py中的代码需要部分更改，见方法generate_response和embdding_sentence中的注释
	
##使用方法说明
	
	训练：
 		python train.py -t train_data_file1 train_data_file2 dic_file model_save_file pre_trained_file(optional)"
 			train_data_file1, 前一句话，一行一个文本，句子的字与字或词语词之间用空格隔开
 			train_data_file2, 后一句话，即回答

 	生成：
 		python train.py -g model_file dic_file post_data_file response_save_file"

 	句子向量化，Embdding Sentence：
 		python train.py -e model_file dic_file sentences_file embdding_save_file"