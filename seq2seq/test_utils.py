import sys, os
from train_utils import *
from setting import *
import numpy as np
import theano
from train import *

def test_train_model(model):
    train_data_file1 = "datas/post.dat"
    train_data_file2 = "datas/answer.dat"
    train_dic_file = "datas/dic.dat"
    model_save_file = "datas/dialog.model"

    sents_one = load_data(train_data_file1, train_dic_file)
    sents_two = load_data(train_data_file2, train_dic_file)

    train_model(model, [ sents_one, sents_two ])
    model.save_model(model_save_file)

def test_generation(model):
    model_file = "datas/dialog.model.npz"
    dic_file = "datas/dic.dat"
    post_file = "datas/post.dat"
    reps_file = "datas/post_result.dat"

    # load paramenters
    model.load_model(model_file)
    posts = load_data(post_file, dic_file)
    dic = [ "<B>", "<E>" ] + [ x.strip() for x in open(dic_file) ]
    responses = [ ]
    for post in posts[ 0:100 ]:
        print len(post)
        if len(post) > 0:
            responses.append(generate_response(model, post, dic))
        else:
            responses.append([ [ ], "" ])
    sout = open(reps_file)
    for post, response in zip(post[0:100], responses):
        sout.write(post + "\t" + "".join(response[ 1 ]) + "\n")

def test_embdding(model):
    model_file = "datas/dialog.model.npz"
    sentences_file = "datas/post.dat"
    dic_file = "datas/dic.dat"
    embdding_save_file = "datas/embdding_result.dat"

    # load paramenters
    model.load_model(model_file)
    posts = load_data(sentences_file, dic_file)
    embddings = []
    for post in posts:
        nh = np.zeros(model.hidden_dim).astype(theano.config.floatX)
        if len(post) > 0:
            nh, _ = model.encode(post)
        embddings.append(nh)

    sout = open(embdding_save_file, "w")
    for embdding in embddings:
        for i in range(embdding.shape[ 0 ]):
            sout.write(str(embdding[ i ]) + " ")
        sout.write("\n")