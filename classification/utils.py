#! /usr/bin/env python

import codecs, random, math
import numpy as np
import theano
from datetime import datetime
from setting import *


def format_seq(seq, mask, dic_file, charset=g_charset):
    """
    Format an id sequence into words
    :param seq, words' ids
    :param mask, mask of sentence length
    :param dic_file, dictionary, each for a word
    :param charset, how to encode the sequence
    :return format text
    """
    i2w = [w.strip().encode(charset) for w in codecs.open(dic_file, "r", charset)]
    txt = [i2w[seq[i]] for i, value in enumerate(mask) if value != 0.]
    return "".join(txt)


def load_data(corpus_file, dic_file, weights={}, shuffle=False, charset=g_charset):
    """
    load data form file, each line for a document,
    :param corpus_file,
        format:
        each line for a document or a sentence pair,
            "document or sentence pair: label\tsentence1\tsentence2",
        sentence,
            word1 word2 word3, seperated by space
        when train for classification sentence1 can be same with sentence2,
        when used for prediction, label could be any value,
        when used for language inference, sentence1 is post, sentence2 is response
    :param dic_file:
    :param shuffle:
    :param charset:
    :param weights: class weights, format: "label,weights,count,class_name"
    :return: x1, x2, y, yc(label weights)
    """
    # datas
    data_list = [line.strip() for line in codecs.open(corpus_file, "r", charset)]
    # indices to words
    i2w = [w.strip() for w in codecs.open(dic_file, "r", charset) if w.strip()!=""]
    # words to indices
    w2i = dict([(w, i) for i, w in enumerate(i2w) if w.strip() != ""])
    # shuffle randomly
    if shuffle:
        random.shuffle(data_list)
    
    # words to ids
    rx1, rx2, ry, ryc = [], [], [], []
    for data in data_list:
        y, x1, x2 = "0", "", ""
        if len(data.strip().split("\t")) >= 3:
            [y, x1, x2] = [_.strip() for _ in data.strip().split("\t")][0:3]
        
        rx1.append([w2i[x] for x in x1.split(" ") if x in w2i])
        rx2.append([w2i[x] for x in x2.split(" ") if x in w2i])
        ry.append(int(y))
        if y in weights:
            ryc.append(weights[y.strip()])
        else:
            ryc.append(1)

    return rx1, rx2, ry, ryc


def format_batch_data(rx1, rx2, ry, ryc, sen_len):
    """
    format word id into numpy matrix, each column for a document
    :param rx1:
    :param rx2:
    :param ry:
    :param ryc:
    :param sen_len:
    :return: [x1, x2, mask_x1, mask_x2, y, yc]
    """
    n_samples = len(ry)
    x1 = np.zeros((sen_len, n_samples)).astype("int32")
    x2 = np.zeros((sen_len, n_samples)).astype("int32")
    mask_x1 = np.zeros((sen_len, n_samples)).astype(theano.config.floatX)
    mask_x2 = np.zeros((sen_len, n_samples)).astype(theano.config.floatX)
    y = np.zeros(n_samples).astype("int32")
    yc = np.zeros(n_samples).astype("int32")
    
    for idx in range(n_samples):
        len_x1 = sen_len if len(rx1[idx]) > sen_len else len(rx1[idx])
        len_x2 = sen_len if len(rx2[idx]) > sen_len else len(rx2[idx])

        x1[:len_x1, idx] = rx1[idx][0:len_x1]
        x2[:len_x2, idx] = rx2[idx][0:len_x2]
        mask_x1[:len_x1, idx] = 1.
        mask_x2[:len_x2, idx] = 1.
    y = ry 
    yc = ryc

    return [x1, x2, mask_x1, mask_x2, y, yc]


def test_model(model, test_data):
    """
    Test model, return(print) precision, recall and f-measure, confuse matrix, loss
    :param model:
    :param test_data:
    :return:
    """
    total_loss = 0.
    total_corr = 0.
    
    labels = sorted(list(set(test_data[2])))
    classes = len(labels)
    r, p, c = np.zeros(classes), np.zeros(classes), np.zeros(classes) #real predict correct
    confuse = np.zeros((classes, classes))

    n_samples = len(test_data[2])
    num_batches = n_samples / g_batch_size
    num_batches = num_batches if num_batches * g_batch_size == n_samples else num_batches + 1
    for num_batch in range(num_batches):
        # Get a min_batch indices from test_data orderly
        batch_idxs, batch_size = get_min_batch_idxs(len(test_data[0]), num_batch, g_batch_size, False)
        batch_x1 = [test_data[0][idx] for idx in batch_idxs]
        batch_x2 = [test_data[1][idx] for idx in batch_idxs]
        batch_y = [test_data[2][idx] for idx in batch_idxs]
        batch_yc = [test_data[3][idx] for idx in batch_idxs]

        # Format data into numpy matrix
        _x1, _x2, _mask_x1, _mask_x2, _y, _yc = format_batch_data(batch_x1, batch_x2, batch_y, batch_yc, g_sen_len)

        _p = model.predictions(_x1, _x2, _mask_x1, _mask_x2)
        _s = model.weights(_x1, _x2, _mask_x1, _mask_x2)
        _c = model.loss(_x1, _x2, _mask_x1, _mask_x2, _y, _yc)
        
        for idx in range(batch_size):
            if idx % 3000 == 0 and idx != 0:
                print _s[idx], "\t", _c[idx], "\t", _y[idx], "\t", -math.log(_s[idx][_y[idx]])
            if _y[idx] == _p[idx]:
                c[_y[idx]] += 1.
                total_corr += 1.
            r[_y[idx]] += 1.
            p[_p[idx]] += 1.
            confuse[_p[idx],_y[idx]] += 1.
        total_loss += _c
        del _x1, _x2, _mask_x1, _mask_x2, _y, _yc

    # test information
    info = str(int(total_corr)) + "Correct, Ratio = " + str(total_corr/n_samples*100) + "%\n"
    info += datetime.now().strftime("%Y-%m-%d-%H-%M") + "\n"
    for label in labels:
        info += "Label(" + str(label) + "): "
        _p = 0. if p[label] == 0. else c[label]/p[label]
        _r = 0. if r[label] == 0. else c[label]/r[label]
        _f = 0. if _p == 0. or _r == 0. else (_p * _r * 2)/(_p + _r)
        info += "P = " + str(_p) + "%, R = " + str(_r) + "%, F = " + str(_f) + "%\n"
    info += "Confuse Matrix:\n"
    for label in labels:
        info += "\t"
        for value in confuse[label]:
            info += str(int(value)) + " "
        info += "\n"

    info += "Loss:" + str(total_loss) + "\n\n"
    
    return info


def get_min_batch_idxs(data_size, batch_index, batch_size, random_data=False):
    """
    Get batch_size indices from range(data_size)
    :param data_size.
    :param batch_index, which batch is selected
    :param batch_size.
    :param random_data, if True, example indices will be selected randomly,
           else from begin (batch_index * batch_size) to end (batch_index * batch_size + batch_size or data_size -1).
           if end - begin < batch_size, the rest will be selected randomly
    :return:
    """
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

    return idxs, end-begin
