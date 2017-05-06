from basic import data_utils, model_utils
import sys


def train_model(model, params, argv):
    for i in range(len(argv)):
        print(argv[i])
    data_train_file = argv[2]
    data_test_file = argv[3]
    data_dic_file = argv[4]
    model_save_file = argv[5]

    # Load pre-trained model
    if len(argv) > 6:
        pre_trained_model = argv[6]
        model.load_model(pre_trained_model)
        # print "Load Paraments Done!"
        # print "Test Model's Paramenters"
        # print testModel(model, validate, batch_size, sen_len)

    # Load train and test data
    train_data = data_utils.load_data(data_train_file, data_dic_file, weights={}, shuffle=True)
    test_data = data_utils.load_data(data_test_file, data_dic_file, weights={})

    num_batches_seen = 0
    num_batches = len(train_data[1]) / params["batch_size"] + 1

    scale = 1.0
    for epoch in range(params["epoches"]):
        loss_begin, loss_end = [], []
        # For each training example...
        for num_batch in range(num_batches):
            scale *= 0.9995
            batch_idxs, _ = data_utils.get_min_batch_idxs(len(train_data[0]), num_batch, params["batch_size"], True)
            batch_x = [train_data[0][idx] for idx in batch_idxs]
            batch_y = [train_data[1][idx] for idx in batch_idxs]
            # Format data into numpy matrix
            _x, _mask_x, _y = data_utils.format_batch_data(batch_x, batch_y, params["sen_len"])

            lr = max(params["learning_rate"] * scale, params["min_learning_rate"])
            loss1 = model.loss(_x, _mask_x, _y)
            model.sgd_step(_x, _mask_x, _y, lr, params["decay"])
            loss2 = model.loss(_x, _mask_x, _y)

            loss_begin.append(loss1)
            loss_end.append(loss2)

            loss1 = sum(loss_begin) / len(loss_begin)
            loss2 = sum(loss_end) / len(loss_end)

            info = "Epoch(%d/%d), Batch(%d/%d), Loss: %f - %f = %f, lr=%f" % (epoch, params["epoches"], num_batch, num_batches - 1, loss1, loss2, loss1 - loss2, lr)
            print "\r", info,
            sys.stdout.flush()
            num_batches_seen += 1
        print
        print
        # debug info
        print "num_batches(" + str(num_batches_seen) + "): "
        print "Epochs(" + str(epoch) + "): "
        print "\nresult: " + data_utils.test_model(model, test_data, batch_size=params["batch_size"], sen_len=params["sen_len"])

        if epoch % params["save_iter"] == params["save_iter"] - 1:
            model_utils.save_model(model_save_file + "." + str(1001 + epoch)[1:] + ".npz", model)

    model_utils.save_model( model_save_file + ".final.npz", model)

    return model


def predict(model, params, argv):
    model_file = argv[2]
    data_dic_file = argv[3]
    data_predict_file = argv[4]
    result_save_file = argv[5]

    # Load trained model
    model.load_model(model_file)
    # Load predict data
    predict_data = data_utils.load_data(data_predict_file, data_dic_file)
    # result save
    sout = open(result_save_file, "w")

    for num_batch in range(len(predict_data[0]) / params["batch_size"] + 1):
        batch_idxs, batch_size = data_utils.get_min_batch_idxs(len(predict_data[0]), num_batch, params["batch_size"], False)
        batch_x = [predict_data[0][idx] for idx in batch_idxs]
        batch_y = [predict_data[2][idx] for idx in batch_idxs]

        # Format data into numpy matrix
        _x, _mask_x, _y = data_utils.format_batch_data(batch_x, batch_y, params["sen_len"])

        # predictions
        _p = model.predictions(_x, _mask_x)
        _s = model.weights(_x, _mask_x)

        print "\rpredicted, ", num_batch * params["batch_size"] + batch_size,

        for i in range(0, batch_size):
            info = ''
            for _v in _s[i]:
                info += str(_v) + '\t'
            info += str(_p[i])
            sout.write(info + '\n')

        if batch_size < params["batch_size"]: break

    sout.flush()
    sout.close()


def print_usage():
    print "Usage:"
    print "For training:"
    print "\tpython train.py -t data_train_file data_test_file data_dic_file model_save_file pre_trained_file(optional)"
    print "\tdata_train_file, train data"
    print "\tdata_test_file, test data"
    print
    print "For prediction:"
    print "\tpython train.py -p model_file dic_file predict_data_file result_save_file"
    print
    exit(0)
