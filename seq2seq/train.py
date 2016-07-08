import sys, os
from train_utils import *
from setting import *
import numpy as np
import theano


# train_data[0] sentence 1
# train_data[1] sentence 2
def train_model(model, argv):
    train_data_file1 = argv[2]
    train_data_file2 = argv[3]
    train_dic_file = argv[4]
    model_save_file = argv[5]
    
    if len(argv) > 6:
        pre_trained_model = sys.argv[6]
        # train from last break
        model.load_model(pre_trained_model)

    train_data_one = load_data(train_data_file1, train_dic_file)
    train_data_two = load_data(train_data_file2, train_dic_file)

    train_data = [train_data_one, train_data_two]

    num_batches = len(train_data[0]) / g_batch_size + 1
    
    for epoch in range(g_n_epoch):
        loss_begin, loss_end = [], []
        
        for num_batch in range(num_batches):
            # get batch indices
            batch_idxs = get_min_batch_idxs(len(train_data[0]), num_batch, g_batch_size, True)
            # format data into numpy matrix
            sents_one = [train_data[0][idx] for idx in batch_idxs]
            sents_two = [train_data[1][idx] for idx in batch_idxs]
            inputs, masks = format_batch_data(sents_one, sents_two, g_sen_len)
            
            # sgd step on batch datas
            loss1 = model.loss(inputs, masks)
            model.sgd_step(inputs, masks, g_learning_rate, g_decay)
            loss2 = model.loss(inputs, masks)

            loss_begin.append(loss1)
            loss_end.append(loss2)

            loss1 = sum(loss_begin) / len(loss_begin)
            loss2 = sum(loss_end) / len(loss_end)

            print "Epoch(%d/%d), Batch(%d/%d), %f - %f = %f" % (epoch, g_n_epoch, num_batch, num_batches, loss1, loss2, loss1 - loss2)
        if epoch % g_save_epoch ==  g_save_epoch - 1:
            model.save_model(model_save_file + "." + str(1001 + epoch)[1:] + ".npz")
        print  
    
    model.save_model(model_save_file + ".final.npz")
    return model


def generate_response(model, input, dic):
    nh, nc = model.encode(input)
    # nh = model.encode(input) # for RNN
    idx = 0
    response = [BEG_ID]
    text = []
    while idx < g_max_length:
        [nh, nc, nx, prob] = model.generate(nh, nc, response[-1])
        # [nh, nx, prob] = model.generate(nh, response[-1]) # for RNN
        if nx == PAD_ID:
            break
        else:
            response.append(nx)
            text += dic[nx]
        idx += 1
    post = [dic[x] for x in input]

    input_text = ""
    for i in input:
        input_text += str(dic[i]) + " "
    response_text = ""
    for r in response[1:]:
        response_text += str(dic[r]) + " "

    return input_text, response_text


def generate_response_all(model, argv):
    model_file = argv[2]
    dic_file = argv[3]
    post_data_file = argv[4]
    response_save_file = argv[5]

    # load paramenters
    model.load_model(model_file)
    posts = load_data(post_data_file, dic_file)
    dic = DIC_HEAD + [x.strip() for x in open(dic_file)]

    responses = []
    for post in posts:
        responses.append(generate_response(model, post, dic))

    sout = open(response_save_file, "w")
    sout.write("post\tresponse\n")
    for response in responses:
        sout.write(response[0] + "\n" + response[1] + "\n\n")
    sout.close()


def embdding_sentence(model, argv):
    model_file = argv[2]
    dic_file = argv[3]
    sentences_file = argv[4]
    embdding_save_file = argv[5]

    # load paramenters
    model.load_model(model_file)
    posts = load_data(sentences_file, dic_file)
    embddings = []
    for post in posts:
        nh = np.zeros(model.hidden_dim).astype(theano.config.floatX)
        if len(post) > 0:
            nh, _ = model.encode(post)
            # nh = model.encode(post)
        embddings.append(nh)

    sout = open(embdding_save_file, "w")
    for embdding in embddings:
        for i in range(embdding.shape[0]):
            sout.write(str(embdding[i]) + " ")
        sout.write("\n")


def print_usage():
    print "Usage:"
    print "For training:"
    print "\tpython train.py -t train_data_file1 train_data_file2 dic_file model_save_file pre_trained_file(optional)"
    print "\ttrain_data_file1, posts"
    print "\ttrain_data_file2, responses"
    print
    print "For Generation:"
    print "\tpython train.py -g model_file dic_file post_data_file response_save_file"
    print
    print "For Embdding Sentence:"
    print "\tpython train.py -e model_file dic_file sentences_file embdding_save_file"
    exit(0)

if __name__ == "__main__":
    if len(sys.argv) <= 1 or (sys.argv[1] != "-t" and sys.argv[1] != "-g" and sys.argv[1] != "-e"):
        print_usage()
        exit(0)

    model = G_MODEL(word_dim=g_word_dim, hidden_dim=g_hidden_dim, sen_len=g_sen_len, batch_size=g_batch_size)

    if DEBUG:
        import test_utils

        test_utils.test_train_model(model)
        test_utils.test_generation(model)
        test_utils.test_embdding(model)
        exit(0)

    if sys.argv[1] == "-t":
        train_model(model, sys.argv)

    elif sys.argv[1] == "-g":
        generate_response_all(model, sys.argv)

    elif sys.argv[1] == "-e":
        embdding_sentence(model, sys.argv)
