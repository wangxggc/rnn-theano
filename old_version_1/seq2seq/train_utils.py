import codecs,random
import numpy as np
import theano
from setting import *

"""
    Format id sequences into numpy format with padding
"""
def format_batch_data(sents_one, sents_two, sen_len):
    assert(len(sents_one) == len(sents_two))
    batch_size = len(sents_one)
    
    np_sents_one = np.zeros((sen_len, batch_size)).astype("int32")
    np_sents_two = np.zeros((sen_len, batch_size)).astype("int32")
    np_target = np.zeros((sen_len, batch_size)).astype("int32")

    np_m_sents_one = np.zeros((sen_len, batch_size)).astype(theano.config.floatX)
    np_m_sents_two = np.zeros((sen_len, batch_size)).astype(theano.config.floatX)
    np_m_target = np.zeros((sen_len, batch_size)).astype(theano.config.floatX)
    
    idx = 0
    for _s1, _s2 in zip(sents_one, sents_two):
        s1 = _s1
        s2 = [BEG_ID] + _s2
        t = _s2 + [PAD_ID]
        
        lens1 = len(s1) if len(s1) < sen_len else sen_len
        lens2 = len(s2) if len(s2) < sen_len else sen_len
        lent = lens2

        np_m_sents_one[0:lens1, idx] = 1.
        np_m_sents_two[0:lens2, idx] = 1.
        np_m_target[0:lent, idx] = 1.

        s1 = s1 + [PAD_ID] * sen_len
        s2 = s2 + [PAD_ID] * sen_len
        t = t + [PAD_ID] * sen_len

        np_sents_one[0:sen_len, idx] = s1[0:sen_len]
        np_sents_two[0:sen_len, idx] = s2[0:sen_len]
        np_target[0:sen_len, idx] = t[0:sen_len]
        idx += 1 
    return [np_sents_one, np_sents_two, np_target], [np_m_sents_one, np_m_sents_two, np_m_target]
    
    
def load_data(corpus_file, dic_file, shuffle=False):
    # indices to words
    i2w = [w.strip() for w in codecs.open(dic_file, "r", g_charset) if w.strip()!=""]
    i2w = DIC_HEAD + i2w
    # words to indices
    w2i = dict([(w, i) for i, w in enumerate(i2w) if w.strip()!=""]) #for python 2.6
    # datas
    data_list = [[w2i[s] for s in line.strip().split(" ") if s in w2i]
                        for line in codecs.open(corpus_file, "r", g_charset)]
    # shuffle randomly
    if shuffle:random.shuffle(data_list)
        
    return data_list


def get_min_batch_idxs(data_size, batch_index, batch_size, random_data=False):
    # get begin and end indices
    begin, end = batch_index * batch_size, batch_index * batch_size + batch_size
    if end > data_size: end = data_size
    if end < begin: begin = end
    
    # get batch index orderly
    idxs = [_ for _ in range(begin, end)]
    # if random_data, get indices randomly
    if random_data: idxs = []
    while len(idxs) < batch_size:
        idxs.append(int(random.random() * data_size) % data_size)
    
    return idxs
