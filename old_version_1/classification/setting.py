from model.lstm_mlstm import LSTMMLstm
from model.lstm import LSTM

DEBUG = False
G_MODEL = LSTM

g_charset = "gb18030"
g_n_epoch = 100	# 迭代次数
g_save_epoch = 10 # 模型保存间隔
g_learning_rate = 0.001	# 学习率 RmsProp
g_decay = 0.95	# RmsProp中的decay
g_batch_size = 1024	# batch的大小
g_sen_len = 20	# 句子的长度
g_hidden_dim = 300	# 隐层的数量
g_class_dim = 31	# 类别数量
g_word_dim = 10000	# 词典大小

