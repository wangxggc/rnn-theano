# /usr/bin/env python
import os
from model.lstm import LSTM
# from model.rnn import RNN

g_charset = "gb18030"
DEBUG = False

PAD_ID, BEG_ID = [0, 1]
PAD, BEG = ["<P>", "<B>"]
DIC_HEAD = ["<P>", "<B>"]

g_max_length = 100	# 生成句子的最大长度

g_n_epoch = 100	# 迭代次数
g_save_epoch = 2	# 模型保存间隔
g_learning_rate = 0.001	# 学习率 RmsProp
g_decay = 0.95	# RmsProp中的decsy
g_batch_size = 64	# batch的大小
g_sen_len = 15	# 句子的长度
g_hidden_dim = 256	# 隐层的数量

g_word_dim = 16531 + 2 # 字典大小 + 2
# g_word_dim = 11007 + 2  # news

G_MODEL = LSTM


