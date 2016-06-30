# /usr/bin/env python
import sys,random
import os
import time
import numpy as np
from datetime import datetime
from utils import *
#from lstm_attention import LSTMAttention
#from rnn import LSTMAttention
from models import * 

#MODEL      = LSTMAttention
MODEL      = LSTM
#DATA_MODEL = 'models/201605061609_dirty_LSTM/dirty-train-validate-test-LSTM-201605061609-4096-15-512-0.001.002.npz'
#DATA_MODEL = "models/201605111543_s8_LSTM/train-validate-test-2048-15-512-0.001.008.npz"
#DATA_MODEL = "models/201605171912_s9_LSTM/train-validate-test-2048-15-512-0.001.009.npz"
#DATA_MODEL = "models/201605241940_egame_LSTM/train-validate-test-2048-15-512-0.001.009.npz"
DATA_MODEL = "models/201606011123_seqing_LSTM/train-validate-test-2048-15-512-0.001.000.npz"

#DIR = 'predict/neisou'
#DIR = 'predict/piaoliuping'
#DIR = 'predict/maren/comments'
#DIR = "predict/egame/neisou_xianliao"
DIR = "predict/seqing/neisou_xianliao"

DATA_PREDICT_Q  = '' + DIR + '/q.label'
DATA_PREDICT_A  = '' + DIR + '/a.label'
DATA_PREDICT_QA = '' + DIR + '/qa.label'
DATA_RESULT_Q   = '' + DIR + '/q.result'
DATA_RESULT_A   = '' + DIR + '/a.result'
DATA_RESULT_QA  = '' + DIR + '/qa.result'
DATA_DIC        = '' + DIR + '/dic_10000.gb18030'

classDim   = 5
batchSize  = 10000
senLen     = 15
hiddenDim  = 512

DEBUG      = True


model, dataqa = buildTestModel(MODEL, DATA_PREDICT_QA, DATA_MODEL, DATA_DIC, classDim=classDim, batchSize=batchSize, senLen=senLen, hiddenDim=hiddenDim)
dataq = loadData(DATA_PREDICT_Q, DATA_DIC, senLen, shuffle=False)
dataa = loadData(DATA_PREDICT_A, DATA_DIC, senLen, shuffle=False)

def predictData(data, result, model=model):
    sout = open(result, 'w')

    labeled = 0

    for idx in range(len(data[-1])/batchSize + 1):
        _x1, _x2, _maskX1, _maskX2, _y, bs = getMinBatchOrder(data[0], data[1], data[2], data[3], data[4], idx, batchSize)
        _p = predict(model, _x1, _x2, _maskX1, _maskX2)
        _s = weight(model, _x1, _x2, _maskX1, _maskX2)
    
        labeled += bs
        print result, labeled
   
        for i in range(0, bs):
            info = ''
            for _v in _s[i]:
                info += str(_v) + '\t'
            info += str(_p[i])
            sout.write(info + '\n')
        if bs<batchSize:break

    sout.flush()
    sout.close()

if __name__ == '__main__':
    predictData(dataqa, DATA_RESULT_QA)
    predictData(dataq, DATA_RESULT_Q)
    predictData(dataa, DATA_RESULT_A)
