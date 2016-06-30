# /usr/bin/env python
import sys, os, time
import numpy as np
from datetime import datetime
from utils import *
from models import * 

MODEL       = LSTMMLstm
DATA_DIR    = "1500W"
DATA_TRAIN  = "./data/" + DATA_DIR + "/train.gb18030"
DATA_VALID  = "./data/" + DATA_DIR + "/valid.gb18030"
DATA_TEST   = "./data/" + DATA_DIR + "/test.gb18030"
DATA_DIC    = "./data/dic_10000.gb18030"



g_n_epoch       = 10
g_learning_rate = 0.001
g_decay         = 0.95
g_decay_delt    = 0.02
g_batch_size    = 2048
g_sen_len       = 20
g_hidden_dim    = 300
g_class_weights = dict([(x.split(",")[0], int(float(x.split(",")[1].strip()))) 
                            for x in open("./data/" + DATA_DIR + "/weights")])

g_model, g_train, g_valid, g_test = buildModel(MODEL, DATA_TRAIN, DATA_VALID, DATA_TEST, DATA_DIC,
                                            learning_rate=g_learning_rate, 
                                            decay      = g_decay,
                                            batch_size = g_batch_size,
                                            sen_len    = g_sen_len,
                                            hidden_dim = g_hidden_dim,
                                            weights    = g_class_weights)

DEBUG         = False
LOAD_MODEL_FILE   = ""

if not LOAD_MODEL_FILE and LOAD_MODEL_FILE != '':
    model.load_model(LOAD_MODEL_FILE)
    print "Load Paraments Done!"
    print "Test Model's Paramenters"
    print testModel(model, validate, batch_size, sen_len)

'''
    log file
'''
ADDINFO = DATA_TRAIN.split("/")[-1] + "-" + DATA_VALID.split("/")[-1] + "-" + DATA_TEST.split("/")[-1]
ADDINFO = ADDINFO.replace(".gb18030", "")

strModel = str(MODEL).split('.')[-1]
ts = datetime.now().strftime("%Y%m%d%H%M")
G_LOG_FILE = 'logs/' + DATA_DIR.replace("/", '.') + '-' + ADDINFO + '-' + strModel + "-%s-%s-%s-%s-%s" % (ts, g_batch_size, g_sen_len, g_hidden_dim, g_learning_rate)

DATA_DIR = ts + "_"  + DATA_DIR.replace("/", '.') + "_" + strModel
os.mkdir("models/" + DATA_DIR)
G_MODEL_FILE = 'models/' + DATA_DIR + '/' + ADDINFO + "-%s-%s-%s-%s" % (g_batch_size, g_sen_len, g_hidden_dim, g_learning_rate)

print G_MODEL_FILE, "\n", G_LOG_FILE


