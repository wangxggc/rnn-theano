import numpy as np
import theano.tensor as T
import theano
from basic import cells, model_utils, layers, data_utils
import sys


class LSTM:
    def __init__(self, feature_dim, output_dim, embed_dim, hidden_dim, batch_size, max_len, id="", drop=True, bidirectional=True):
        """
        inputs and output are different, inputs are topics, outputs are words
        Four Layers:
            Embdding Layer,
                embedding word id to vectors
            Bidirectional Encoder Layer,
                embedding a sentence into a hidden matrix
            Attention Decoder Layer,
                decoding a encoding matrix into a decoding hidden matrix
                with attention weights for each word
            Predict Layer
                predict words by hidden matrix calculated by Decoder
        Args:
            output_dim: word size in seq2seq
            embed_dim: embedding size
            hidden_dim:
            batch_size:
            max_len: maximum length while generate
            id:
        """
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_len = max_len
        self.id = id
        self.drop = drop
        self.bidirectional = bidirectional

        # define e
        # usually the generate model can shared the same embedding with the inference model

        self.word_embedding_layer = layers.EmbddingLayer(
            feature_dim=feature_dim,
            embed_dim=embed_dim,
            id=self.id + "word_embed_"
        )

        if bidirectional:
            self.encoder = layers.BidirectionalEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                id=self.id+"encoder_",
                shared=False
            )
            self.predict_layer = layers.PredictLayer(
                hidden_dim=hidden_dim * 2,
                output_dim=output_dim,
                id=self.id+"predict_"
            )
        else:
            self.encoder = layers.BasicEncoder(
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                batch_size=batch_size,
                id=self.id + "encoder_"
            )
            self.predict_layer = layers.PredictLayer(
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                id=self.id + "predict_"
            )
        import json
        # collects all parameters
        self.params = dict()
        self.param_names = {"orign":[], "cache":[]}
        self.params.update(self.word_embedding_layer.params)
        self.params.update(self.encoder.params)
        self.params.update(self.predict_layer.params)

        for name in ["orign", "cache"]:
            self.param_names[name] += self.word_embedding_layer.param_names[name]
            self.param_names[name] += self.encoder.param_names[name]
            self.param_names[name] += self.predict_layer.param_names[name]
        param_key = self.params.keys()
        param_key.sort()
        putty = json.dumps(param_key, indent=4)

        print "param_names = ", putty

        self.__build__model__()

        print "Build LSTM Done! Params = %s" % (json.dumps({"id":id, "output_dim":output_dim, "embed_dim":embed_dim, "hidden_dim":hidden_dim, "batch_size":batch_size}, indent=4))

    def getParams(self):
        return self.params.values()

    def __build__model__(self):
        input = T.imatrix('input')  # first sentence
        input_mask = T.fmatrix('input_mask')  # mask
        target = T.ivector('target')  # label

        # embedding encoder and decoder inputs with two embedding layers
        encoding = self.word_embedding_layer.embedding(input.flatten())
        encoding = encoding.reshape([input.shape[0], input.shape[1], self.embed_dim])
        # encodes and decodes
        encoding, _ = self.encoder.encode(encoding, input_mask)

        final_hidden_states = encoding[-1, :, :] # shape(batch_size, embedding)
        train_weights, _ = self.predict_layer.predict(final_hidden_states, drop=self.drop, train=True)

        # calculate cross entropy
        cost = T.sum(T.nnet.categorical_crossentropy(train_weights, target.flatten()))

        cost += self.predict_layer.get_l2_cost() * 0.01

        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        grads, updates = model_utils.rms_prop(cost, self.param_names, self.params, learning_rate, decay)

        # Assign functions
        self.loss = theano.function([input, input_mask, target], cost, on_unused_input='ignore')
        self.bptt = theano.function([input, input_mask, target], grads, on_unused_input='ignore')
        self.sgd_step = theano.function(
            [input, input_mask, target, learning_rate, decay],
            updates=updates, on_unused_input='ignore')

        # for test
        weights, predictions = self.predict_layer.predict(final_hidden_states, drop=False, train=False)
        self.weights = theano.function([input, input_mask], weights, on_unused_input='ignore')
        self.predictions = theano.function([input, input_mask], predictions, on_unused_input='ignore')




if __name__ == "__main__":
    from train_utils import *

    if len(sys.argv) <= 1 or (sys.argv[1] != "-t" and sys.argv[1] != "-p"):
        print_usage()
        exit(0)

    params = {
        "output_dim":2,
        "word_dim":-1,
        "embed_dim":64,
        "hidden_dim":128,
        "sen_len":30,
        "batch_size":1024,
        "learning_rate":0.001,
        "min_learning_rate":1e-6,
        "decay":0.95,
        "epoches":200,
        "save_iter":2,
        "bidirectional":False,
        "drop":True
    }

    dic_file = sys.argv[4]
    if sys.argv[1] == "-p":dic_file=sys.argv[3]
    feature_dim = len([_ for _ in open(dic_file)])
    params["word_dim"] = feature_dim

    import json
    print "Settings =", json.dumps(params, indent=2)

    model = LSTM(feature_dim=params["word_dim"],
                 output_dim=params["output_dim"],
                 embed_dim=params["embed_dim"],
                 hidden_dim=params["hidden_dim"],
                 batch_size=params["batch_size"],
                 max_len=params["sen_len"],
                 id="LSTM_",
                 drop=params["drop"],
                 bidirectional=params["bidirectional"])

    if sys.argv[1] == "-t":
        train_model(model, params, sys.argv)
    elif sys.argv[1] == "-p":
        predict(model, params, sys.argv)