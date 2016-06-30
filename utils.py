#! /usr/bin/env python

import time,codecs,random,sys,math
import numpy as np
import theano
from datetime import datetime
#from pympler import tracker

CHARSET = 'gb18030'
DEBUG   = True
MACOS   = False

def DEBUG__(INFO, LINE=sys._getframe().f_lineno):
    if DEBUG:
    	print 'DEBUG__(%s): %s'% (str(LINE),str(INFO).replace('\n', ''))

def FORMAT__(seq, mask, DIC_FILE, charset=CHARSET):
    i2w = [w.strip().encode(charset) for w in codecs.open(DIC_FILE, 'r', charset)]
    txt = [i2w[seq[i]] for i, value in enumerate(mask) if value!=0.]
    return ''.join(txt)
    
def loadData(dataFile, dicFile, senLen, shuffle=True, charset=CHARSET):
    #indices to words
    i2w = [w.strip() for w in codecs.open(dicFile, 'r', charset) if w.strip()!='']
    #words to indices
    w2i = dict([(w,i) for i,w in enumerate(i2w) if w.strip()!='']) #for python 2.6
    #datas
    dataList = [line.strip() for line in codecs.open(dataFile, 'r', charset)]
    #shuffle randomly
    if shuffle:random.shuffle(dataList)
    
    DEBUG__(i2w[0].encode(charset), LINE=sys._getframe().f_lineno)
    DEBUG__(w2i.keys()[0].encode(charset), LINE=sys._getframe().f_lineno)
    
    #words to ids
    rx1, rx2, ry = [],[],[]
    for data in dataList:
        x1, x2, y = '', '', ''
        if len(data.strip().split('\t')) == 3:
            [x1, x2, y] = data.strip().split('\t')
        elif len(data.strip().split('\t')) == 2:
            [x1, x2] = data.strip().split('\t')
            y = 0
        else:
            continue
        #print data
        rx1.append([w2i[x] for x in x1 if x in w2i])
        rx2.append([w2i[x] for x in x2 if x in w2i])
        ry.append(int(y))
   
    DEBUG__(rx1[0], LINE=sys._getframe().f_lineno)
    DEBUG__(rx2[0], LINE=sys._getframe().f_lineno)
    DEBUG__(ry[0],  LINE=sys._getframe().f_lineno)
    """
    Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or senLen.

    if senLen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    nSamples = len(ry)
    x1 = np.zeros((senLen, nSamples)).astype('int32')
    x2 = np.zeros((senLen, nSamples)).astype('int32')
    maskX1 = np.zeros((senLen, nSamples)).astype(theano.config.floatX)
    maskX2 = np.zeros((senLen, nSamples)).astype(theano.config.floatX)
    lenX1 = [senLen if len(x) > senLen else len(x) for x in rx1]
    lenX2 = [senLen if len(x) > senLen else len(x) for x in rx2]
    y = np.zeros(nSamples).astype('int32')
    
    for idx in range(nSamples):
        x1[:lenX1[idx], idx] = rx1[idx][0:lenX1[idx]]
        x2[:lenX2[idx], idx] = rx2[idx][0:lenX2[idx]]
        maskX1[:lenX1[idx], idx] = 1.
        maskX2[:lenX2[idx], idx] = 1.
    y = ry 
    DEBUG__(FORMAT__(x1[:,0], maskX1[:, 0], dicFile), LINE=sys._getframe().f_lineno)
    return [x1, x2, maskX1, maskX2, y]




def buildModel(MODEL, DATA_TRAIN, DATA_VALIDATE, DATA_TEST, DATA_DIC, 
            learningRate, decay, batchSize, senLen,
            hiddenDim, charset=CHARSET):
    '''
    Load data begin
    '''
    train = loadData(DATA_TRAIN,  DATA_DIC, senLen)
    valid = loadData(DATA_VALIDATE, DATA_DIC, senLen)
    test  = loadData(DATA_TEST, DATA_DIC, senLen)
    
    classDim  = len(set(train[-1]))
    wordDim   = len([x.strip('\n') for x in open(DATA_DIC)])
    
    DEBUG__(train[0][:,0], LINE=sys._getframe().f_lineno)    #sentence1
    DEBUG__(train[1][:,0], LINE=sys._getframe().f_lineno)    #sentence2
    DEBUG__(train[-1][0], LINE=sys._getframe().f_lineno)     #label
    DEBUG__(FORMAT__(train[0][:,0], train[2][:,0], DATA_DIC), LINE=sys._getframe().f_lineno)
    
    print 'Class:',classDim
    print 'Words:',wordDim
    print "Load Data Done! Train=%d, Validate=%d, Test=%d" % (len(train[-1]), len(valid[-1]), len(test[-1]))
    
    '''
        Build model
    '''
    print MODEL
    model = MODEL(classDim=classDim, 
                    wordDim=wordDim, 
                    hiddenDim=hiddenDim, 
                    senLen=senLen, 
                    batchSize=batchSize)
    print "Build Model Done!"
    
    '''
        Check Model
    '''
    x1 = train[0][:,0:batchSize]
    x2 = train[1][:,0:batchSize]
    maskX1 = train[2][:,0:batchSize]
    maskX2 = train[3][:,0:batchSize]
    y = train[4][0:batchSize]
    
    print x1.shape, x2.shape, 
    
    t1 = time.time()
    model.sgd_step(x1, x2, maskX1, maskX2, y, learningRate, decay)
    t2 = time.time()
    print "Batch SGD step time: %f milliseconds" % ((t2 - t1) * 1000.)
    print model.bptt(x1, x2, maskX1, maskX2, y)[1]
    
    return model, train, valid, test

def buildTestModel(MODEL, DATA_TEST, DATA_MODEL, DATA_DIC, classDim, batchSize, senLen, hiddenDim, charset = CHARSET):
    wordDim = len([x for x in open(DATA_DIC)])
    print "Word Dim =", wordDim
    
    print MODEL
    model =  MODEL(classDim=classDim,
                wordDim=wordDim,
                hiddenDim=hiddenDim,
                senLen=senLen,
                batchSize=batchSize)
    print "Build Model Done!"

    model.load_model(DATA_MODEL)
    print "Load Paramenters Done!"
    
    test = loadData(DATA_TEST, DATA_DIC, senLen, shuffle=False)
    print "Load Data Done! Test=%d" % (len(test[-1]))
    
    return model, test

def testModel(model, data, batchSize, senLen):
    [x1, x2, maskX1, maskX2, y] = data
    
    totalLoss = 0.
    totalCorr = 0.
    
    labels = sorted(list(set(y)))
    classes = len(labels)
    r, p, c = np.zeros(classes), np.zeros(classes), np.zeros(classes) #real predict correct
    confuse = np.zeros((classes, classes))

    nSamples = len(y)
    nBatch = nSamples / batchSize
    nBatch = nBatch if nBatch * batchSize == nSamples else nBatch+1
    
    for i in range(nBatch):  
        _x1, _x2, _maskX1, _maskX2, _y, bs = getMinBatchOrder(x1, x2, maskX1, maskX2, y, i, batchSize)
        _p = predict(model, _x1, _x2, _maskX1, _maskX2)
        _s = weight(model, _x1, _x2, _maskX1, _maskX2)
        _c = loss(model, _x1, _x2, _maskX1, _maskX2, _y)
        
        for idx in range(bs):
            if idx % 3000 == 0 and idx != 0:
                print _s[idx], '\t', _c[idx], '\t', _y[idx], '\t', -math.log(_s[idx][_y[idx]])
            if _y[idx] == _p[idx]:
                c[_y[idx]] += 1.
                totalCorr += 1.
            r[_y[idx]] += 1.
            p[_p[idx]] += 1.
            confuse[_p[idx],_y[idx]] += 1.    
            totalLoss += _c[idx]
        del _x1, _x2, _maskX1, _maskX2, _y
    #test information
    info = str(int(totalCorr)) + 'Correct, Ratio = ' + str(totalCorr/nSamples*100) + '%\n'
    info += datetime.now().strftime("%Y-%m-%d-%H-%M") + '\n'
    for label in labels:
        info += 'Label(' + str(label) + '): '
        _p = 0. if p[label] == 0. else c[label]/p[label]
        _r = 0. if r[label] == 0. else c[label]/r[label]
        _f = 0. if _p == 0. or _r == 0. else (_p * _r * 2)/(_p + _r)
        info += 'P = ' + str(_p) + '%, R = ' + str(_r) + '%, F = ' + str(_f) + '%\n'
    info += 'Confuse Matrix:\n'
    for label in labels:
        info += '\t'
        for value in confuse[label]:
            info += str(int(value)) + ' '
        info += '\n'
        #info += '\t' + str(confuse[label]) + '\n'
        
    info += 'Loss:' + str(totalLoss) + '\n\n'
    
    return info
    
def getMinBatchOrder(x1, x2, maskX1, maskX2, y, batchIndex, batchSize):
    senLen, nSamples = x1.shape[0], x1.shape[1]
    begin, end = batchIndex * batchSize, (batchIndex + 1) * batchSize
    end = nSamples if end > nSamples else end
        
    _x1 = np.zeros((senLen, batchSize)).astype('int32')
    _x2 = np.zeros((senLen, batchSize)).astype('int32')
    _maskX1 = np.zeros((senLen, batchSize)).astype(theano.config.floatX)
    _maskX2 = np.zeros((senLen, batchSize)).astype(theano.config.floatX)
    _y = np.zeros(batchSize).astype('int32')
    
    if begin >= nSamples:
        return _x1, _x2, _maskX1, _maskX2, _y, 0
    
    idxs = [x for x in range(begin, end)]  
    while len(idxs) < batchSize:
        idxs.append(int(random.random() * nSamples) % nSamples)  

    for i, idx in enumerate(idxs):
        _x1[:, i] = x1[:, idx]
        _x2[:, i] = x2[:, idx]
        _maskX1[:, i] = maskX1[:, idx]
        _maskX2[:, i] = maskX2[:, idx]
        _y[i] = y[idx]
        
    return _x1, _x2, _maskX1, _maskX2, _y, end-begin

def getMinBatchRandom(x1, x2, maskX1, maskX2, y, batchSize):
    senLen, nSamples = x1.shape[0], x1.shape[1]
    
    _x1 = np.zeros((senLen, batchSize)).astype('int32')
    _x2 = np.zeros((senLen, batchSize)).astype('int32')
    _maskX1 = np.zeros((senLen, batchSize)).astype(theano.config.floatX)
    _maskX2 = np.zeros((senLen, batchSize)).astype(theano.config.floatX)
    _y = np.zeros(batchSize).astype('int32')
    idxs = [int(random.random() * nSamples) % nSamples for i in range(batchSize)]
    for i, idx in enumerate(idxs):
        _x1[:, i] = x1[:, idx]
        _x2[:, i] = x2[:, idx]
        _maskX1[:, i] = maskX1[:, idx]
        _maskX2[:, i] = maskX2[:, idx]
        _y[i] = y[idx]
    #random select samples
            
    return _x1, _x2, _maskX1, _maskX2, _y
    
'''
    Calculate losses of batch of data
'''
def loss(model, x1, x2, maskX1, maskX2, y):
    return model.errors(x1, x2, maskX1, maskX2, y)

'''
    Gives predictions of batch of data
'''
def predict(model, x1, x2, maskX1, maskX2):
    return model.predictions(x1, x2, maskX1, maskX2)

'''
    Gives soft prediction weights of batch of data
'''
def weight(model, x1, x2, maskX1, maskX2):
    return model.weights(x1, x2, maskX1, maskX2)

'''
    Gives attentions of batch of data
    noted that, attentions are different along with models
'''
def attention(model, x1, x2, maskX1, maskX2):
    return model.attentions(x1, x2, maskX1, maskX2)
 
