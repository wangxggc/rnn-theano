# /usr/bin/env python
import sys
import os
import time
import numpy as np
from datetime import datetime
from utils import *
#from lstm_attention import LSTMAttention
#from rnn import LSTMAttention
from models import * 

#MODEL      = LSTMAttention
#MODEL      = LSTMWByW
#MODEL      = LSTM

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

MODEL_FILE   = ""

ADDINFO = DATA_TRAIN.split("/")[-1] + "-" + DATA_VALID.split("/")[-1] + "-" + DATA_TEST.split("/")[-1]
ADDINFO = ADDINFO.replace(".gb18030", "")


model, train, validate, test = buildModel(MODEL, DATA_TRAIN, DATA_VALID, DATA_TEST, DATA_DIC,
                                    learningRate=learningRate, 
                                    decay=decay,
                                    batchSize=batchSize,
                                    senLen=senLen,
                                    hiddenDim=hiddenDim)

if not MODEL_FILE and MODEL_FILE != '':
    model.load_model(MODEL_FILE)
    print "Load Paraments Done!"
    print "Test Model's Paramenters"
    print testModel(model, validate, batchSize, senLen)
'''
    log file
'''
strModel = str(MODEL).split('.')[-1]
ts = datetime.now().strftime("%Y%m%d%H%M")
LOG_FILE = 'logs/' + DATA_DIR.replace("/", '.') + '-' + ADDINFO + '-' + strModel + "-%s-%s-%s-%s-%s" % (ts, batchSize, senLen, hiddenDim, learningRate)


DATA_DIR = ts + "_"  + DATA_DIR.replace("/", '.') + "_" + strModel
os.mkdir("models/" + DATA_DIR)
MODEL_FILE = 'models/' + DATA_DIR + '/' + ADDINFO + "-%s-%s-%s-%s" % (batchSize, senLen, hiddenDim, learningRate)

#MODEL_FILE = 'models/' + DATA_DIR + '-' + ADDINFO + '-' + strModel + "-%s-%s-%s-%s-%s" % (ts, batchSize, senLen, hiddenDim, learningRate)

print MODEL_FILE
print LOG_FILE


    

def train(model=model, train=train, test=test,
        learningRate=learningRate, 
        decay=decay,
        batchSize=batchSize,
        senLen=senLen,
        hiddenDim=hiddenDim):
    log = open(LOG_FILE + '.log', 'w')
    num_batches_seen = 0


    for epoch in range(NEPOCH):
        # For each training example...
        batches = len(train[-1]) / batchSize + 1
        
        learningRate /= 1.05
        decay -= 0.02
        #learningRate /= epoch / 5  + 1
        #decay -= (epoch / 5 + 1) * decay_delt

        for i in range(batches):
            # _x1, _x2, _maskX1, _maskX2, _y, size = getMinBatchOrder(train[0], train[1], train[2], train[3], train[4], 
            #                                                         batchIndex=i, batchSize=batchSize)
            _x1, _x2, _maskX1, _maskX2, _y = getMinBatchRandom(train[0], train[1], train[2], train[3], train[4], batchSize=batchSize)
            
            loss1 = sum(loss(model, _x1, _x2, _maskX1, _maskX2, _y))
            model.sgd_step(_x1, _x2, _maskX1, _maskX2, _y, learningRate, decay)
            loss2 = sum(loss(model, _x1, _x2, _maskX1, _maskX2, _y))
                
            info = 'Loss(' + str(i) + '/' + str(batches) +'):' + str(loss1) + ' - ' + str(loss2) + ' = ' + str(loss1 - loss2)
            DEBUG__(info, LINE=sys._getframe().f_lineno)
            log.write(info + '\n')
            
            num_batches_seen += 1
           
            if (i % (batches / 5) == 0 and i != 0) or i == batches - 1:
                DEBUG__('Batches(' + str(num_batches_seen) +'): ', LINE=sys._getframe().f_lineno)
                DEBUG__('Epochs(' + str(epoch) +'): ', LINE=sys._getframe().f_lineno)
                result_valid = '\nvalid: ' + testModel(model, validate, batchSize, senLen)
                result_test  = '\ntest: '  + testModel(model, test, batchSize, senLen)
    
                #print result_train
                print result_valid
                print result_test
        
                #log.write(result_train + '\n\n')
                log.write('Batches(' + str(num_batches_seen) +'): \n')
                log.write('Epochs(' + str(epoch) +'): \n')
                log.write(result_valid + '\n\n')
                log.write(result_test + '\n\n')
                # log.write(model.W.get_value())
                log.flush()
        model.save_model(MODEL_FILE + '.' + str(1000 + epoch)[1:])
    log.close()
    
    return model

train()
  
   
