# /usr/bin/env python
import os
from model.lstm import LSTM

g_charset = "gb18030"
DEBUG = False

PAD_ID, BEG_ID = [0, 1]
PAD, BEG = ["<P>", "<B>"]

g_max_length = 100

g_n_epoch = 100
g_save_epoch = 10
g_learning_rate = 0.001
g_decay = 0.95
g_batch_size = 32
g_sen_len = 15
g_hidden_dim = 100


g_word_dim = 11007 + 2
G_MODEL = LSTM

if not os.path.exists("results"):
    os.mkdir("results")


