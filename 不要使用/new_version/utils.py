#! /usr/bin/env python

import time,codecs,random,sys,math
import numpy as np
import theano
from datetime import datetime
#from pympler import tracker

CHARSET = 'gb18030'
DEBUG   = True

def DEBUG__(INFO, LINE=sys._getframe().f_lineno):
    if DEBUG:
        print 'DEBUG__(%s): %s'% (str(LINE),str(INFO).replace('\n', ''))

def FORMAT__(seq, mask, DIC_FILE, charset=CHARSET):
    i2w = [w.strip().encode(charset) for w in codecs.open(DIC_FILE, 'r', charset)]
    txt = [i2w[seq[i]] for i, value in enumerate(mask) if value!=0.]
    return ''.join(txt)
    
def loadData(corpus_file, dic_file, sen_len, shuffle=True, charset=CHARSET, weights={}):
    #indices to words
    i2w = [w.strip() for w in codecs.open(dic_file, 'r', charset) if w.strip()!='']
    #words to indices
    w2i = dict([(w,i) for i,w in enumerate(i2w) if w.strip()!='']) #for python 2.6
    #datas
    dataList = [line.strip() for line in codecs.open(corpus_file, 'r', charset)]
    #shuffle randomly
    if shuffle:random.shuffle(dataList)
    
    DEBUG__(i2w[0].encode(charset), LINE=sys._getframe().f_lineno)
    DEBUG__(w2i.keys()[0].encode(charset), LINE=sys._getframe().f_lineno)
    
    #words to ids
    rx1, rx2, ry, ryc = [],[],[], []
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
        if int(y) > 30:print data
        ry.append(int(y))
        if y.strip() in weights:
            ryc.append(weights[y.strip()])
        else:
            ryc.append(1);
            
    DEBUG__(rx1[0], LINE=sys._getframe().f_lineno)
    DEBUG__(rx2[0], LINE=sys._getframe().f_lineno)
    DEBUG__(ry[0],  LINE=sys._getframe().f_lineno)
    DEBUG__(ryc[0],  LINE=sys._getframe().f_lineno)
    """
    Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or sen_len.

    if sen_len is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    n_samples = len(ry)
    x1 = np.zeros((sen_len, n_samples)).astype('int32')
    x2 = np.zeros((sen_len, n_samples)).astype('int32')
    mask_x1 = np.zeros((sen_len, n_samples)).astype(theano.config.floatX)
    mask_x2 = np.zeros((sen_len, n_samples)).astype(theano.config.floatX)
    lenX1 = [sen_len if len(x) > sen_len else len(x) for x in rx1]
    lenX2 = [sen_len if len(x) > sen_len else len(x) for x in rx2]
    y = np.zeros(n_samples).astype('int32')
    yc= np.zeros(n_samples).astype('int32')
    
    for idx in range(n_samples):
        x1[:lenX1[idx], idx] = rx1[idx][0:lenX1[idx]]
        x2[:lenX2[idx], idx] = rx2[idx][0:lenX2[idx]]
        mask_x1[:lenX1[idx], idx] = 1.
        mask_x2[:lenX2[idx], idx] = 1.
    y = ry 
    yc= ryc
    DEBUG__(FORMAT__(x1[:,0], mask_x1[:, 0], dic_file), LINE=sys._getframe().f_lineno)
    return [x1, x2, mask_x1, mask_x2, y, yc]




def buildModel(MODEL, DATA_TRAIN, DATA_VALIDATE, DATA_TEST, DATA_DIC, 
            learning_rate, decay, batch_size, sen_len,
            hidden_dim, charset=CHARSET, weights={}):
    '''
    Load data begin
    '''
    train = loadData(DATA_TRAIN,  DATA_DIC, sen_len, weights=weights)
    valid = loadData(DATA_VALIDATE, DATA_DIC, sen_len, weights=weights)
    test  = loadData(DATA_TEST, DATA_DIC, sen_len, weights=weights)
    
    class_dim  = len(set(train[4]))
    word_dim   = len([x.strip('\n') for x in open(DATA_DIC)])
    
    DEBUG__(train[0][:,0], LINE=sys._getframe().f_lineno)    #sentence1
    DEBUG__(train[1][:,0], LINE=sys._getframe().f_lineno)    #sentence2
    DEBUG__(train[4][0], LINE=sys._getframe().f_lineno)     #label
    DEBUG__(FORMAT__(train[0][:,0], train[2][:,0], DATA_DIC), LINE=sys._getframe().f_lineno)
    
    print 'Class:',class_dim
    print 'Words:',word_dim
    print "Load Data Done! Train=%d, Validate=%d, Test=%d" % (len(train[-1]), len(valid[-1]), len(test[-1]))
    
    '''
        Build model
    '''
    print MODEL
    model = MODEL(class_dim=class_dim, 
                    word_dim=word_dim, 
                    hidden_dim=hidden_dim, 
                    sen_len=sen_len, 
                    batch_size=batch_size)
    print "Build Model Done!"
    
    '''
        Check Model
    '''
    x1 = train[0][:,0:batch_size]
    x2 = train[1][:,0:batch_size]
    mask_x1 = train[2][:,0:batch_size]
    mask_x2 = train[3][:,0:batch_size]
    y = train[4][0:batch_size]
    yc= train[5][0:batch_size]
    
    print x1.shape, x2.shape, 
    
    t1 = time.time()
    model.sgd_step(x1, x2, mask_x1, mask_x2, y, yc, learning_rate, decay)
    t2 = time.time()
    print "Batch SGD step time: %f milliseconds" % ((t2 - t1) * 1000.)
    print model.bptt(x1, x2, mask_x1, mask_x2, y, yc)[1]
    
    return model, train, valid, test

def buildTestModel(MODEL, DATA_TEST, DATA_MODEL, DATA_DIC, class_dim, batch_size, sen_len, hidden_dim, charset = CHARSET):
    word_dim = len([x for x in open(DATA_DIC)])
    print "Word Dim =", word_dim
    
    print MODEL
    model =  MODEL(class_dim=class_dim,
                word_dim=word_dim,
                hidden_dim=hidden_dim,
                sen_len=sen_len,
                batch_size=batch_size)
    print "Build Model Done!"

    model.load_model(DATA_MODEL)
    print "Load Paramenters Done!"
    
    test = loadData(DATA_TEST, DATA_DIC, sen_len, shuffle=False)
    print "Load Data Done! Test=%d" % (len(test[-1]))
    
    return model, test

def testModel(model, data, batch_size):
    [x1, x2, mask_x1, mask_x2, y, yc] = data
    
    totalLoss = 0.
    totalCorr = 0.
    
    labels = sorted(list(set(y)))
    classes = len(labels)
    r, p, c = np.zeros(classes), np.zeros(classes), np.zeros(classes) #real predict correct
    confuse = np.zeros((classes, classes))

    n_samples = len(y)
    nBatch = n_samples / batch_size
    nBatch = nBatch if nBatch * batch_size == n_samples else nBatch+1
    # nBatch = 30
    for i in range(nBatch):  
        _x1, _x2, _mask_x1, _mask_x2, _y, _yc, bs = getMinBatch(x1, x2, mask_x1, mask_x2, y, yc, i, batch_size)
        _p = predict(model, _x1, _x2, _mask_x1, _mask_x2)
        _s = weight(model, _x1, _x2, _mask_x1, _mask_x2)
        _c = loss(model, _x1, _x2, _mask_x1, _mask_x2, _y, _yc)
        
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
        del _x1, _x2, _mask_x1, _mask_x2, _y, _yc
    #test information
    info = str(int(totalCorr)) + 'Correct, Ratio = ' + str(totalCorr/n_samples*100) + '%\n'
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
    
def getMinBatch(x1, x2, mask_x1, mask_x2, y, yc, batchIndex, batch_size, random_data=False):
    sen_len, n_samples = x1.shape[0], x1.shape[1]
    begin, end = batchIndex * batch_size, (batchIndex + 1) * batch_size
    end = n_samples if end > n_samples else end
        
    _x1 = np.zeros((sen_len, batch_size)).astype('int32')
    _x2 = np.zeros((sen_len, batch_size)).astype('int32')
    _mask_x1 = np.zeros((sen_len, batch_size)).astype(theano.config.floatX)
    _mask_x2 = np.zeros((sen_len, batch_size)).astype(theano.config.floatX)
    _y  = np.zeros(batch_size).astype('int32')
    _yc = np.zeros(batch_size).astype('int32')
    
    if begin >= n_samples:
        return _x1, _x2, _mask_x1, _mask_x2, _y, _yc, 0
    
    idxs = [x for x in range(begin, end)]  
    if random_data == True:
        idxs=[]
    while len(idxs) < batch_size:
        idxs.append(int(random.random() * n_samples) % n_samples)  

    for i, idx in enumerate(idxs):
        _x1[:, i] = x1[:, idx]
        _x2[:, i] = x2[:, idx]
        _mask_x1[:, i] = mask_x1[:, idx]
        _mask_x2[:, i] = mask_x2[:, idx]
        _y[i] = y[idx]
        _yc[i]= yc[idx]
        
    return _x1, _x2, _mask_x1, _mask_x2, _y, _yc, end-begin
    
'''
    Calculate losses of batch of data
'''
def loss(model, x1, x2, mask_x1, mask_x2, y, yc):
    return model.errors(x1, x2, mask_x1, mask_x2, y, yc)

'''
    Gives predictions of batch of data
'''
def predict(model, x1, x2, mask_x1, mask_x2):
    return model.predictions(x1, x2, mask_x1, mask_x2)

'''
    Gives soft prediction weights of batch of data
'''
def weight(model, x1, x2, mask_x1, mask_x2):
    return model.weights(x1, x2, mask_x1, mask_x2)

'''
    Gives attentions of batch of data
    noted that, attentions are different along with models
'''
def attention(model, x1, x2, mask_x1, mask_x2):
    return model.attentions(x1, x2, mask_x1, mask_x2)
 
