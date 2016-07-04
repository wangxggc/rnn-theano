# /usr/bin/env python
import sys, os, time
import numpy as np
from datetime import datetime
from utils import *
from models import * 

CHATSET     = "gb18030"
DEBUG         = False

DATA_DIR    = ""
DATA_TRAIN  = DATA_DIR + "/train.dat"
DATA_DIC    = "./data/dic_10000.dat"
LOAD_MODEL_FILE   = ""

g_n_epoch       = 10
g_learning_rate = 0.001
g_decay         = 0.95
g_decay_delt    = 0.02
g_batch_size    = 2048
g_sen_len       = 20
g_hidden_dim    = 300


g_model, g_train = buildModel(MODEL, DATA_TRAIN, DATA_DIC,
                                            learning_rate=g_learning_rate, 
                                            decay      = g_decay,
                                            batch_size = g_batch_size,
                                            sen_len    = g_sen_len,
                                            hidden_dim = g_hidden_dim,
                                            weights    = g_class_weights)



if not LOAD_MODEL_FILE and LOAD_MODEL_FILE != "":
    model.load_model(LOAD_MODEL_FILE)
    print "Load Paraments Done!"
    print "Test Model's Paramenters"


G_LOG_FILE = "%s-%s-%s-%s.log" % (g_batch_size, g_sen_len, g_hidden_dim, g_learning_rate)
G_MODEL_FILE = "%s-%s-%s-%s.model" % (g_batch_size, g_sen_len, g_hidden_dim, g_learning_rate)
print G_MODEL_FILE, "\n", G_LOG_FILE


