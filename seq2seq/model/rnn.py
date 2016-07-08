import numpy as np
import theano
from model_utils import *


class LSTM:
    
    def __init__(self, word_dim, hidden_dim, sen_len, batch_size, truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sen_len = sen_len
        self.batch_size = batch_size
        self.truncate = truncate

        params = {}
        # Initialize the network parameters
        params["E"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))        # Ebdding Matirx
        params["W"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim)) 	# RNN paramenters w(i,f,o,c)
        params["B"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim))             	# RNN paramenters bias
        params["DecodeW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, word_dim))  # Decoder
        params["DecodeB"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim))
        
        # Assign paramenters" names 
        self.param_names = {"orign": ["E", "W", "B", "DecodeW", "DecodeB"],
                            "cache": ["mE", "mW", "mB", "mDecodeW", "mDecodeB"]}
        # Theano: Created shared variables
        self.params = {}
        # Model"s shared variables
        for _n in self.param_names["orign"]:
            self.params[_n] = theano.shared(value=params[_n].astype(theano.config.floatX), name=_n)
        # Shared variables for RMSProp
        for _n in self.param_names["cache"]:
            self.params[_n] = theano.shared(value=np.zeros(params[_n[1:]].shape).astype(theano.config.floatX), name=_n)
        # Build model graph
        self.__theano_build_train__()
        self.__theano_build_generation__()

    def __theano_build_train__(self):
        params = self.params
        params_names = self.param_names
        hidden_dim = self.hidden_dim
        batch_size = self.batch_size

        # inputs[0], first sentence.
        # inputs[1], second sentence.
        # inputs[2], encoding target
        inputs = T.itensor3("inputs")
        masks = T.ftensor3("masks")

        def rnn_cell(x, mx, ph, Wh):
            h = T.tanh(ph.dot(Wh) + x)
            h = mx[:, None] * h + (1-mx[:, None]) * ph
            return [h]  # size = sample * hidden : 3 * 4

        # encoding first sentence
        _state = params["E"][inputs[0].flatten(), :].reshape([inputs[0].shape[0], inputs[0].shape[1], hidden_dim])
        _state = _state.dot(params["W"][0]) + params["B"][0]
        [h1], updates = theano.scan(
            fn=rnn_cell,
            sequences=[_state, masks[0]],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([batch_size, hidden_dim]))],
            non_sequences=[params["W"][1]])

        # decoding second sentence
        _state = params["E"][inputs[1].flatten(), :].reshape([inputs[1].shape[0], inputs[1].shape[1], hidden_dim])
        _state = _state.dot(params["W"][2]) + params["B"][1]
        [h2], updates = theano.scan(
            fn=rnn_cell,
            sequences=[_state, masks[1]],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=h1[-1])],
            non_sequences=[params["W"][3]])

        # Loss
        _s = h2.dot(params["DecodeW"]) + params["DecodeB"]
        _s = _s.reshape([_s.shape[0] * _s.shape[1], _s.shape[2]])
        _s = T.nnet.softmax(_s)
        _cost = T.nnet.categorical_crossentropy(_s, inputs[2].flatten())
        _cost = T.sum(_cost * masks[2].flatten())

        # SGD parameters
        learning_rate = T.scalar("learning_rate")
        decay = T.scalar("decay")

        _grads, _updates = rms_prop(_cost, params_names, params, learning_rate, decay)
        
        # Assign functions
        self.bptt = theano.function([inputs, masks], _grads)
        self.loss = theano.function([inputs, masks], _cost)
        self.weights = theano.function([inputs, masks], _s)
        self.sgd_step = theano.function(
            [inputs, masks, learning_rate, decay], #theano.In(decay, value=0.9)],
            updates=_updates)

    def __theano_build_generation__(self):
        params = self.params
        hidden_dim = self.hidden_dim

        # generation process
        g_input = T.ivector("g_input")

        def rnn_generation_cell(x, ph, Ex, Wx, Wh, bx):
            h = Ex[x, :].dot(Wx) + ph.dot(Wh) + bx
            return [h]

        [e_h], updates = theano.scan(
            fn=rnn_generation_cell,
            sequences=[g_input],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros(self.hidden_dim))],
            non_sequences=[params["E"], params["W"][2], params["W"][3], params["B"][1]])

        ph = T.fvector("ph")
        px = T.iscalar("px")
        [nh] = rnn_generation_cell(px, ph, params["E"], params["W"][2], params["W"][3], params["B"][1])
        _prob = T.nnet.softmax(nh.dot(params["DecodeW"]) + params["DecodeB"])
        _pred = T.argmax(_prob)

        # Encode first sentence into hidden states
        self.encode = theano.function([g_input], [e_h[-1]])
        # Generate a word based on the previous state
        self.generate = theano.function([ph, px], [nh, _pred, _prob], allow_input_downcast=True)

    #  save paramenters
    def save_model(self, outfile):
        np.savez(outfile,
            hidden_dim = self.hidden_dim,
            word_dim  = self.word_dim,
            batch_size = self.batch_size,
            E = self.params["E"].get_value(),
            W = self.params["W"].get_value(),
            B = self.params["B"].get_value(),
            DecodeW = self.params["DecodeW"].get_value(),
            DecodeB = self.params["DecodeB"].get_value())
        print "Saved LSTM's parameters to %s." % outfile

    def load_model(self, path):
        npzfile = np.load(path)
        print("Loading model from %s.\nParamenters: \n\t|hidden_dim=%d \n\t|word_dim=%d" 
                        % (path, npzfile["hidden_dim"], npzfile["word_dim"]))
        self.params["E"].set_value(npzfile["E"])
        self.params["W"].set_value(npzfile["W"])
        self.params["B"].set_value(npzfile["B"])
        self.params["DecodeW"].set_value(npzfile["DecodeW"])
        self.params["DecodeB"].set_value(npzfile["DecodeB"])
