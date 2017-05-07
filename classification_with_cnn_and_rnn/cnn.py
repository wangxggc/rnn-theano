import numpy as np
import theano.tensor as T
import theano
from basic import cells, model_utils, layers, data_utils
import sys

class CNN:
    def __init__(self, settings, id=""):
        """
        Layers:
            1, Embedding Layer
            2, {ConvLayer + PoolLayer} * K
            3, MLPLayers
            4, PredictLayer
        """
        self.id = id
        self.batch_size = settings["batch_size"]
        self.sen_len = settings["sen_len"]
        self.feature_dim = settings["feature_dim"]
        self.embed_dim = settings["embed_dim"]
        # [(10, 1, 3, 3), (), ()]
        self.filters_shapes = settings["filters_shapes"]
        self.pooling = settings["pooling"]
        # input size = \sigma_i^k{kernel.shape[0]}
        self.hidden_dim = -1
        self.hidden_dims = [sum([shape[0] for shape in self.filters_shapes])] + settings["hidden_dims"]
        self.output_dim = settings["output_dim"]
        self.drop = settings["drop"]

        self.word_embedding_layer = layers.EmbddingLayer(
            feature_dim=self.feature_dim,
            embed_dim=self.embed_dim,
            id=self.id + "word_embed_"
        )

        if len(self.hidden_dims) >= 2:
            self.mlp_layers = layers.MLPLayers(
                hidden_dims=self.hidden_dims,
                nonlinear=T.nnet.sigmoid,
                id=self.id + "mlp_"
            )
        else:
            self.mlp_layers = None

        self.conv_layers = []
        for idx, conv_shape in enumerate(self.filters_shapes):
            conv_layer = layers.ConvPoolLayer(batch_size=self.batch_size,
                                              sen_len=self.sen_len, 
                                              embed_dim=self.embed_dim,
                                              filter_size=conv_shape[0],
                                              channels=conv_shape[1],
                                              filter_shape=conv_shape,
                                              pooling_mode = self.pooling,
                                              id = self.id + "conv_" + str(idx) + "_")
            self.conv_layers.append(conv_layer)

        self.predict_layer = layers.PredictLayer(
            hidden_dim=self.hidden_dims[-1],
            output_dim=self.output_dim,
            id=self.id + "predict_"
        )

        import json

        # collects all parameters
        self.params = dict()
        self.param_names = {"orign": [], "cache": []}
        self.params.update(self.word_embedding_layer.params)
        if self.mlp_layers:
            self.params.update(self.mlp_layers.params)
        for conv_layer in self.conv_layers:
            self.params.update(conv_layer.params)
        self.params.update(self.predict_layer.params)

        for name in ["orign", "cache"]:
            self.param_names[name] += self.word_embedding_layer.param_names[name]
            if self.mlp_layers:
                self.param_names[name] += self.mlp_layers.param_names[name]
            for conv_layer in self.conv_layers:
                self.param_names[name] += conv_layer.param_names[name]
            self.param_names[name] += self.predict_layer.param_names[name]

        param_key = self.params.keys()
        param_key.sort()
        putty = json.dumps(param_key, indent=4)
        print "param_names = ", putty

        self.__build__model__()

    def __build__model__(self):
        x = T.imatrix('x')  # first sentence
        x_mask = T.fmatrix("x_mask")
        y = T.ivector('y')  # label
        
        # for compatibility, input's shape is (sen_len, batch_size)
        # for cnn convolution, the input shape should be (batch_size, 1, sen_len, embed_dim)
        # embedding encoder and decoder inputs with two embedding layers
        embedding = self.word_embedding_layer.embedding(x.flatten())
        embedding = embedding.reshape([x.shape[0], 1, x.shape[1], self.embed_dim])
        embedding = embedding.dimshuffle(2, 1, 0, 3)
        # embedding = embedding.reshape([input.shape[0], input.shape[1], self.embed_dim])

        conv_outs = [conv_layer.get_output(embedding) for conv_layer in self.conv_layers]
        conv_hidden = T.concatenate(conv_outs, axis=1)
        # conv_hidden = theano.printing.Print("Conv Hidden")(conv_hidden)

        final_hidden_states = self.mlp_layers.get_output(conv_hidden)

        train_weights, _ = self.predict_layer.predict(final_hidden_states, drop=self.drop, train=True)
        # calculate cross entropy
        cost = T.sum(T.nnet.categorical_crossentropy(train_weights, y.flatten()))

        cost += self.predict_layer.get_l2_cost() * 0.01

        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        grads, updates = model_utils.rms_prop(cost, self.param_names, self.params, learning_rate, decay)

        # Assign functions
        self.loss = theano.function([x, x_mask, y], cost, on_unused_input='ignore')
        self.bptt = theano.function([x, x_mask, y], grads, on_unused_input='ignore')
        self.sgd_step = theano.function(
            [x, x_mask, y, learning_rate, decay],
            updates=updates, on_unused_input='ignore')

        # for test
        weights, predictions = self.predict_layer.predict(final_hidden_states, drop=self.drop, train=False)
        self.weights = theano.function([x, x_mask], weights, on_unused_input='ignore')
        self.predictions = theano.function([x, x_mask], predictions, on_unused_input='ignore')


if __name__ == "__main__":
    from train_utils import *

    if len(sys.argv) <= 1 or (sys.argv[1] != "-t" and sys.argv[1] != "-p"):
        print_usage()
        exit(0)

    params = {
        "output_dim":2,
        "feature_dim":-1,
        "embed_dim":64,
        "filters_shapes":[(10, 1, 2, 64), (10, 1, 3, 64), (10, 1, 5, 64), (10, 1, 4, 64)],
        "pooling":"max",
        "hidden_dims":[256, 256],
        "sen_len":30,
        "batch_size":1024,
        "learning_rate":0.001,
        "min_learning_rate":1e-6,
        "decay":0.95,
        "epoches":200,
        "save_iter":2,
        "drop":True
    }

    dic_file = sys.argv[4]
    if sys.argv[1] == "-p":dic_file=sys.argv[3]
    feature_dim = len([_ for _ in open(dic_file)])
    params["feature_dim"] = feature_dim

    import json
    print "Settings =", json.dumps(params, indent=2)

    model = CNN(params, id="CNN_")

    if sys.argv[1] == "-t":
        train_model(model, params, sys.argv)
    elif sys.argv[1] == "-p":
        predict(model, params, sys.argv)
