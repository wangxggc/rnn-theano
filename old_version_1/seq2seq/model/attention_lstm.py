import numpy as np
import theano.tensor as T
import theano
import time, sys
from model_utils import *

# passage REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION, http://arxiv.org/pdf/1509.06664.pdf

class LSTMMLstm:
    
    def __init__(self, class_dim=0, word_dim=0, hidden_dim=0, sen_len=0, batch_size=0, truncate=-1):
        # Assign instance variables
        self.class_dim = class_dim
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sen_len = sen_len
        self.batch_size = batch_size
        self.truncate = truncate
        params = {}
        # Initialize the network parameters
        params["E"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))  # Ebdding Matirx
        params["W"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim * 4)) # W[0-1].dot(x), W[2-3].(i,f,o,c)
        params["B"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim * 4))  # B[0-1] for W.dot(x)
        # Attention paramenters
        params["AttenW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim, hidden_dim))  #attention U and w
        params["AttenV"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim))
        # Final full connection paramenters
        params["DecodeW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim, class_dim))
        params["DecodeB"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (class_dim))
        # Assign paramenters' names 
        self.param_names = {"orign":["E",  "W",  "B",  "AttenW",  "AttenV",  "DecodeW",  "DecodeB"],
                            "cache":["mE", "mW", "mB", "mAttenW", "mAttenV", "mDecodeW", "mDecodeB"]}
        # Theano: Created shared variables
        self.params = {}
        # Model's shared variables
        for _n in self.param_names["orign"]:
            self.params[_n] = theano.shared(value=params[_n].astype(theano.config.floatX), name=_n)
        # Shared variables for RMSProp
        for _n in self.param_names["cache"]:
            self.params[_n] = theano.shared(value=np.zeros(params[_n[1:]].shape).astype(theano.config.floatX), name=_n)
        # We store the Theano graph here
        self.__theano_build_train__()
        self.__theano_build_generation__()
    
    def __theano_build_train__(self):
        params = self.params
        param_names = self.param_names
        hidden_dim = self.hidden_dim

        # inputs[0], first sentence.
        # inputs[1], second sentence.
        # inputs[2], encoding target
        inputs = T.itensor3("inputs")
        masks = T.ftensor3("masks")

        #encoder
        def lstm_cell(x, mx, ph, pc, Wh):
            _x = ph.dot(Wh) + x
            i = T.nnet.hard_sigmoid(_x[:, hidden_dim * 0 : hidden_dim * 1])
            f = T.nnet.hard_sigmoid(_x[:, hidden_dim * 1 : hidden_dim * 2])
            o = T.nnet.hard_sigmoid(_x[:, hidden_dim * 2 : hidden_dim * 3])
            c = T.tanh(_x[:, hidden_dim * 3 : hidden_dim * 4])
            
            c = f * pc + i * c
            c = mx[:, None] * c + (1.- mx[:,None]) * pc
            
            h = o * T.tanh(c)
            h = mx[:, None] * h + (1.- mx[:,None]) * ph
            
            return [h, c]

        # decoder
        def attention_wbyw_cell(x, mx, ph, pc, pr, pa, Mt, Hd, Wh, Wa, Va):
            _x = ph.dot(Wh) + x
            i  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 0 : hidden_dim * 1])
            f  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 1 : hidden_dim * 2])
            o  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 2 : hidden_dim * 3])
            c  = T.tanh(_x[:, hidden_dim * 3 : hidden_dim * 4])

            c = f * pc + i * c
            c = mx[:, None] * c + (1.- mx[:,None]) * pc

            h = o * T.tanh(c)
            h = mx[:, None] * h + (1.- mx[:,None]) * ph

            # WbyW Attention
            a = T.tanh(Mt + h.dot(Wa))   # sen_len * batch * hidden
            a = T.nnet.softmax(a.dot(Va).T) # batch * sen_len
            a = mx[:, None] * a + (1.-mx[:, None]) * pa

            r = (Hd * a).sum(axis=2).T # (hidden, batch, sen_len) * (batch, sen_len) == > (batch, hidden)
            r = mx[:, None] * r + (1.-mx[:, None]) * pr

            return [h, c, r, a]

        # encoding first sentence
        _state = params["E"][inputs[0].flatten(), :].reshape( [inputs[0].shape[0], inputs[0].shape[1], hidden_dim])
        _state = _state.dot(params["W"][0]) + params["B"][0]
        [h1, c1], updates = theano.scan(
            fn=lstm_cell,
            sequences=[_state, masks[0]],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=[params["W"][2]])

        # attention pre-calculate for recurrent
        Mt = h1.dot(params["AttenW"][0]) # sen_len * batch * hidden
        Hd = h1.dimshuffle(2, 1, 0) # hidden * batch * sen_len
        # decoding second sentence
        _state = params["E"][inputs[1].flatten(), :].reshape([inputs[1].shape[0], inputs[1].shape[1], hidden_dim])
        _state = _state.dot(params["W"][2]) + params["B"][1]
        [h2, _, _, a], updates = theano.scan(
            fn=attention_wbyw_cell,
            sequences=[_state, masks[1]],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=h1[-1]),
                          dict(initial=c1[-1]),
                          # dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.sen_len]))],
            non_sequences=[Mt, Hd, params["W"][3], params["AttenW"][1], params["AttenV"]])

        _a = a
        # Loss
        _s = h2.dot(params["DecodeW"]) + params["DecodeB"]
        _s = _s.reshape([_s.shape[0] * _s.shape[1], _s.shape[2]])
        _s = T.nnet.softmax(_s)
        _cost = T.nnet.categorical_crossentropy(_s, inputs[2].flatten())
        _cost = T.sum(_cost * masks[2].flatten())

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')

        # Gradients and updates
        _grads, _updates = rms_prop(_cost, param_names, params, learning_rate, decay)

        # Assign functions
        self.bptt = theano.function([inputs, masks], _grads)
        self.loss = theano.function([inputs, masks], _cost)
        self.weights = theano.function([inputs, masks], _s)
        self.attentions = theano.function([inputs, masks], _a)
        self.sgd_step = theano.function(
            [inputs, masks, learning_rate, decay],  # theano.In(decay, value=0.9)],
            updates=_updates)


    def __theano_build_generation__(self):
        params = self.params
        hidden_dim = self.hidden_dim

        # generation process
        g_input = T.ivector("g_input")

        def lstm_generation_cell(x, ph, pc, Ex, Wx, Wh, bx):
            _x = Ex[ x, : ].dot(Wx) + ph.dot(Wh) + bx
            i = T.nnet.hard_sigmoid(_x[ hidden_dim * 0: hidden_dim * 1 ])
            f = T.nnet.hard_sigmoid(_x[ hidden_dim * 1: hidden_dim * 2 ])
            o = T.nnet.hard_sigmoid(_x[ hidden_dim * 2: hidden_dim * 3 ])
            c = T.tanh(_x[ hidden_dim * 3: hidden_dim * 4 ])

            c = f * pc + i * c
            h = o * T.tanh(c)

            return [ h, c ]

        # decoder
        def attention_wbyw_cell(x, ph, pc, pr, H, Ex, Wx, bx, Wh, AW, Aw):
            _x = Ex[ x, : ].dot(Wx) + ph.dot(Wh) + bx
            i = T.nnet.hard_sigmoid(_x[ :, hidden_dim * 0: hidden_dim * 1 ])
            f = T.nnet.hard_sigmoid(_x[ :, hidden_dim * 1: hidden_dim * 2 ])
            o = T.nnet.hard_sigmoid(_x[ :, hidden_dim * 2: hidden_dim * 3 ])
            c = T.tanh(_x[ :, hidden_dim * 3: hidden_dim * 4 ])

            c = f * pc + i * c
            h = o * T.tanh(c)

            # WbyW Attention
            mt = T.tanh(H.dot(AW[0]) + (h.dot(AW[1])))  # sen_len * hidden
            a = T.nnet.softmax(mt.dot(Aw).T)  # sen_len

            r = H.T.dot(a) # hidden

            return [ h, c, r, a ]

        [ e_h, e_c ], updates = theano.scan(
            fn=lstm_generation_cell,
            sequences=[ g_input ],
            truncate_gradient=self.truncate,
            outputs_info=[ dict(initial=T.zeros(self.hidden_dim)),
                           dict(initial=T.zeros(self.hidden_dim)) ],
            non_sequences=[ params[ "E" ], params[ "W" ][ 2 ], params[ "W" ][ 3 ], params[ "B" ][ 1 ] ])

        ph = T.fvector("ph")
        pc = T.fvector("pc")
        pr = T.fvector("pr")
        px = T.iscalar("px")
        eh = T.fmatrix("eh")

        [ nh, nc, nr, na] = attention_wbyw_cell(px, ph, pc, pr, eh,
                                          params[ "E" ], params[ "W" ][ 2 ], params[ "W" ][ 3 ], params[ "B" ][ 1 ],
                                          params["AttenW"], params["AttenV"])
        _prob = T.nnet.softmax(nh.dot(params[ "DecodeW" ]) + params[ "DecodeB" ])
        _pred = T.argmax(_prob)

        # Encode first sentence into hidden states
        self.encode = theano.function([ g_input ], [ e_h[ -1 ], e_c[ -1 ] ])
        # Generate a word based on the previous state
        self.generate = theano.function([ ph, pc, px ], [ nh, nc, nr, na, _pred, _prob ], allow_input_downcast=True)

    #save paramenters
    def save_model(self, outfile):
        np.savez(outfile,
            batch_size = self.batch_size,
            sen_len = self.sen_len,
            class_dim  = self.class_dim,
            hidden_dim = self.hidden_dim,
            E = self.params["E"].get_value(),
            W = self.params["W"].get_value(),
            B = self.params["B"].get_value(),
            AttenW = self.params["AttenW"].get_value(),
            AttenV = self.params["BAttenV"].get_value(),
            DecodeW = self.params["DecodeW"].get_value(),
            DecodeB = self.params["DecodeB"].get_value())
        print "Saved WbyW LSTM' parameters to %s." % outfile

    def load_model(self, path):
        npzfile = np.load(path)
        print "Loading model from %s.\nParamenters: \n\t|class_dim=%d \n\t|hidden_dim=%d \n\t|word_dim=%d" % (path, npzfile["class_dim"], npzfile["hidden_dim"], npzfile["word_dim"])
        sys.stdout.flush()
        self.params["E"].set_value(npzfile["E"])
        self.params["W"].set_value(npzfile["W"])
        self.params["B"].set_value(npzfile["B"])
        self.params["AttenW"].set_value(npzfile["AttenW"])
        self.params["AttenV"].set_value(npzfile["AttenV"])
        self.params["DecodeW"].set_value(npzfile["DecodeW"])
        self.params["DecodeB"].set_value(npzfile["DecodeB"])


