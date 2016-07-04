import time,codecs,random,sys,math
import numpy as np
import theano
from datetime import datetime

_BEG, _PAD = [0, 1]


"""
    Format id sequences into numpy format with padding
"""
def formatBatchData(batch_sent1, batch_sent2, batch_size, sen_len):


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