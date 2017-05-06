# /usr/bin/env python
import os, sys, time
from utils import *
from setting import *


def train_model(model, argv):
    data_train_file = argv[2]
    data_test_file = argv[3]
    data_dic_file = argv[4]
    data_class_weights = argv[5]
    model_save_file = argv[6]

    # Load class weights
    weights = {}
    if os.path.exists(data_class_weights):
        weights = dict([(x.split(",")[0], int(float(x.split(",")[1].strip())))
                         for x in open(data_class_weights)])

    # Load pre-trained model
    if len(argv) > 7:
        pre_trained_model = argv[7]
        model.load_model(pre_trained_model)
        # print "Load Paraments Done!"
        # print "Test Model's Paramenters"
        # print testModel(model, validate, batch_size, sen_len)

    # Load train and test data
    train_data = load_data(data_train_file,  data_dic_file, weights=weights, shuffle=True)
    test_data = load_data(data_test_file, data_dic_file, weights=weights)


    num_batches_seen = 0
    num_batches = len(train_data[2]) / g_batch_size + 1

    for epoch in range(g_n_epoch):
        loss_begin, loss_end = [], []
        # For each training example...
        for num_batch in range(num_batches):
            batch_idxs, _ = get_min_batch_idxs(len(train_data[0]), num_batch, g_batch_size, False)
            batch_x1 = [train_data[0][idx] for idx in batch_idxs]
            batch_x2 = [train_data[1][idx] for idx in batch_idxs]
            batch_y = [train_data[2][idx] for idx in batch_idxs]
            batch_yc = [train_data[3][idx] for idx in batch_idxs]

            # Format data into numpy matrix
            _x1, _x2, _mask_x1, _mask_x2, _y, _yc = format_batch_data(batch_x1, batch_x2, batch_y, batch_yc, g_sen_len)
            
            loss1 = model.loss( _x1, _x2, _mask_x1, _mask_x2, _y, _yc)
            model.sgd_step(_x1, _x2, _mask_x1, _mask_x2, _y, _yc, g_learning_rate, g_decay)
            loss2 = model.loss( _x1, _x2, _mask_x1, _mask_x2, _y, _yc)
            
            loss_begin.append(loss1)
            loss_end.append(loss2)

            loss1 = sum(loss_begin) / len(loss_begin)
            loss2 = sum(loss_end) / len(loss_end)

            info = "Epoch(%d/%d), Batch(%d/%d), Loss: %f - %f = %f" % (epoch, g_n_epoch, num_batch, num_batches - 1, loss1, loss2, loss1 - loss2)
            print info

            num_batches_seen += 1
        
        # debug info
        print "num_batches(" + str(num_batches_seen) + "): "
        print "Epochs(" + str(epoch) + "): "
        print "\nresult: " + test_model(model, test_data)

        if epoch % g_save_epoch == g_save_epoch - 1:
            model.save_model(model_save_file + "." + str(1001 + epoch)[1:] + ".npz")

    model.save_model(model_save_file + ".final.npz")

    return model


def predict(model, argv):
    model_file = argv[2]
    data_predict_file = argv[3]
    data_dic_file = argv[4]
    result_save_file = argv[5]

    # Load trained model
    model.load_model(model_file)
    # Load predict data
    predict_data = load_data(data_predict_file, data_dic_file)
    # result save
    sout = open(result_save_file, "w")

    for num_batch in range(len(predict_data[0]) / g_batch_size + 1):
        batch_idxs, batch_size = get_min_batch_idxs(len(predict_data[0]), num_batch, g_batch_size, False)
        batch_x1 = [predict_data[0][idx] for idx in batch_idxs]
        batch_x2 = [predict_data[1][idx] for idx in batch_idxs]
        batch_y = [predict_data[2][idx] for idx in batch_idxs]
        batch_yc = [predict_data[3][idx] for idx in batch_idxs]

        # Format data into numpy matrix
        _x1, _x2, _mask_x1, _mask_x2, _y, _yc = format_batch_data(batch_x1, batch_x2, batch_y, batch_yc, g_sen_len)

        # predictions
        _p = model.predictions(_x1, _x2, _mask_x1, _mask_x2)
        _s = model.weights(_x1, _x2, _mask_x1, _mask_x2)

        print "\rpredicted, ", num_batch * g_batch_size + batch_size,

        for i in range(0, batch_size):
            info = ''
            for _v in _s[i]:
                info += str(_v) + '\t'
            info += str(_p[i])
            sout.write(info + '\n')

        if batch_size < g_batch_size: break

    sout.flush()
    sout.close()



def print_usage():
    print "Usage:"
    print "For training:"
    print "\tpython train.py -t data_train_file data_test_file data_dic_file data_class_weights model_save_file pre_trained_file(optional)"
    print "\tdata_train_file, train data"
    print "\tdata_test_file, test data"
    print "\tdata_class_weights, class weights, format: \"label,weights,count,class_name\""
    print
    print "For prediction:"
    print "\tpython train.py -p model_file predict_data_file dic_file result_save_file"
    print
    exit(0)

if __name__ == "__main__":
    if len(sys.argv) <= 1 or (sys.argv[1] != "-t" and sys.argv[1] != "-p"):
        print_usage()
        exit(0)

    model = G_MODEL(class_dim=g_class_dim,  word_dim=g_word_dim, hidden_dim=g_hidden_dim, sen_len=g_sen_len, batch_size=g_batch_size)
  
    if sys.argv[1] == "-t":
        train_model(model, sys.argv)
    elif sys.argv[1] == "-p":
        predict(model, sys.argv)

