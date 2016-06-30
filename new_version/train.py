# /usr/bin/env python
import sys
import os
import time
import numpy as np
from datetime import datetime
from utils import *
from models import * 
from setting import * 



"""
    Train
"""
def train_model():
    global g_model
    global g_train
    global g_valid
    global g_test
    global g_batch_size
    global g_learning_rate
    global g_decay
    global g_n_epoch
    global G_MODEL_FILE
    global G_LOG_FILE

    log = open(G_LOG_FILE + '.log', 'w')
    num_batches_seen = 0
    num_batches = len(g_train[4]) / g_batch_size + 1
    for epoch in range(g_n_epoch):
        # For each training example...
        g_learning_rate /= 1.05
        g_decay -= 0.02

        for batch_index in range(num_batches):
            _x1, _x2, _mask_x1, _mask_x2, _y, _yc, _ = getMinBatch(g_train[0], g_train[1], 
                                                                      g_train[2], g_train[3], 
                                                                      g_train[4], g_train[5],
                                                                      batchIndex=batch_index,
                                                                      batch_size=g_batch_size)
            
            loss1 = sum(loss(g_model, _x1, _x2, _mask_x1, _mask_x2, _y, _yc))
            g_model.sgd_step(_x1, _x2, _mask_x1, _mask_x2, _y, _yc, g_learning_rate, g_decay)
            loss2 = sum(loss(g_model, _x1, _x2, _mask_x1, _mask_x2, _y, _yc))
                
            info = 'Batch(%d/%d), Loss: %f - %f = %f' % (batch_index, num_batches - 1, loss1, loss2, loss1 - loss2)
            DEBUG__(info, LINE=sys._getframe().f_lineno)
            log.write(info + '\n')
            
            num_batches_seen += 1
           
            if (batch_index % (num_batches / 5) == 0 and batch_index != 0) or batch_index == num_batches - 1:
                DEBUG__('num_batches(' + str(num_batches_seen) +'): ', LINE=sys._getframe().f_lineno)
                DEBUG__('Epochs(' + str(epoch) +'): ', LINE=sys._getframe().f_lineno)
                
                result_valid = '\nvalid: ' + testModel(g_model, g_valid, g_batch_size)
                result_test  = '\ntest: '  + testModel(g_model, g_test,  g_batch_size)
    
                print result_valid
                print result_test
        
                log.write('num_batches(' + str(num_batches_seen) +'): \n')
                log.write('Epochs(' + str(epoch) +'): \n')
                log.write(result_valid + '\n\n')
                log.write(result_test + '\n\n')
                log.flush()
        g_model.save_model(G_MODEL_FILE + '.' + str(1000 + epoch)[1:])
    log.close()
    
    return g_model

if __name__ == "__main__":
    train_model()
  
   
