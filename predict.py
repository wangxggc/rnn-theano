# /usr/bin/env python
import sys,random
import os
import time
import numpy as np
from datetime import datetime
from utils import *
from models import * 

MODEL      = LSTM

DATA_MODEL = "models/yourmodel"

DIR = ""


DATA_PREDICT_QA = '' + DIR + '/qa.label'
DATA_RESULT_QA  = '' + DIR + '/qa.result'
DATA_DIC        = '' + DIR + '/dic.gb18030'

classDim   = 5
batchSize  = 10000
senLen     = 15
hiddenDim  = 512

DEBUG      = True


model, dataqa = buildTestModel(MODEL, DATA_PREDICT_QA, DATA_MODEL, DATA_DIC, classDim=classDim, batchSize=batchSize, senLen=senLen, hiddenDim=hiddenDim)

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
