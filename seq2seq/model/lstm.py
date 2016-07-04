import numpy as np
import theano as theano
import theano.tensor as T
from utils import *
from cells import *

#Paper: REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION
class LSTM:
    
    def __init__(self, class_dim=2, word_dim=100, hidden_dim=32, sen_len=5, batch_size=10, truncate=-1):
        # Assign instance variables
        print class_dim,word_dim,hidden_dim
        self.class_dim  = class_dim
        self.word_dim   = word_dim
        self.hidden_dim = hidden_dim
        self.sen_len    = sen_len
        self.batch_size = batch_size
        self.truncate   = truncate
        self.cell       = LSTMCell
        params = {}
        # Initialize the network parameters
        params["E"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))          #Ebdding Matirx
        params["W"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim * 4)) #W[0-1].dot(x), W[2-3].(i,f,o,c)
        params["B"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim * 4))             #B[0-1] for W[0-1]
        params["EncodeW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, word_dim))         #LR W and b
        params["EncodeB"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim))
        
        # Assign paramenters" names 
        self.param_names = {"orign":["E", "W",  "B",  "EncodeW",   "EncodeB"], 
                           "cache":["mE", "mW", "mB", "mEncoderW", "mEncoderB"]}
        # Theano: Created shared variables
        self.params = {}
        # Model"s shared variables
        for _n in self.param_names["orign"]:
            self.params[_n] = theano.shared(value=params[_n].astype(theano.config.floatX), name=_n)
        # Shared variables for RMSProp
        for _n in self.param_names["cache"]:
            self.params[_n] = theano.shared(value=np.zeros(params[_n[1:]].shape).astype(theano.config.floatX), name=_n)
        # Build model graph
        self.__theano_build__()

    def __theano_build__(self):
        params = self.params
        hidden_dim = self.hidden_dim
        cell = self.cell

        # sent1, first sentence.
        # sent2, second sentence.
        # target, encoding target
        sent1  = T.imatrix("sent1")
        sent2  = T.imatrix("sent2")
        target = T.imatrix("target")
        # Mask
        m_sent1  = T.fmatrix("m_sent1")    
        m_sent2  = T.fmatrix("m_sent2")           
        m_target = T.fmatrix("m_target")

        # Embdding words
        _E1 = params["E"].dot(params["W"][0]) + params["B"][0]
        _E2 = params["E"].dot(params["W"][1]) + params["B"][1]
        state1 = _E1[sent1.flatten(), :].reshape([sent1.shape[0], sent1.shape[1], hidden_dim * 4])
        state2 = _E2[sent2.flatten(), :].reshape([sent2.shape[0], sent2.shape[1], hidden_dim * 4])
        

        [h1, c1], updates = theano.scan(
            fn=cell,
            sequences=[statex1, x1_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=params["W"][2])
        
        [h2, c2], updates = theano.scan(
            fn=cell,
            sequences=[statex2, x2_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=h1[-1]),  #using h[end] to initialize the second LSTM
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=params["W"][3])
       
        # Loss
        _s = h2.dot(params["EncodeW"]) + params["EncodeB"]
        _s = _s.reshape([_s.shape[0] * _s.shape[1], _s.shape[2]])
        _s = T.nnet.softmax(_s)
        _cost = T.nnet.categorical_crossentropy(_s, target)
        _cost = _cost * m_target
        _cost = T.sum(_cost)
        
        # SGD parameters
        learning_rate  = T.scalar("learning_rate")
        decay         = T.scalar("decay")
        
        # Gradients:
        # e.g. dE = T.grad(cost, E)
        _grads  = [T.grad(_cost, params[_n]) 
                    for _i, _n in enumerate(self.param_names["orign"])]
        # RMSProp caches: 
        # e.g. mE = decay * self.mE + (1 - decay) * dE ** 2
        _caches = [decay * params[_n] + (1 - decay) * _grads[_i] ** 2 
                    for _i, _n in enumerate(self.param_names["cache"])]
        # Learning rate: 
        # e.g. (E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
        updateOrign = [(params[_n], params[_n] - learning_rate * _grads[_i] / T.sqrt(_caches[_i] + 1e-6))
                    for _i, _n in enumerate(self.param_names["orign"])]
        # Update cache
        # e.g. (self.mE, mE)
        updateCaches= [(params[_n], _caches[_i])
                    for _i, _n in enumerate(self.param_names["cache"])]
        # Merge all updates
        updates = updateOrign + updateCaches
        
        # Assign functions
        self.bptt        = theano.function([x1, x2, x1_mask, x2_mask, y, y_c], _grads)
        self.errors      = theano.function([x1, x2, x1_mask, x2_mask, y, y_c], _c)
        self.weights     = theano.function([x1, x2, x1_mask, x2_mask], _s)
        self.predictions = theano.function([x1, x2, x1_mask, x2_mask], _p)
        self.sgd_step = theano.function(
            [x1, x2, x1_mask, x2_mask, y, y_c, learning_rate, decay], #theano.In(decay, value=0.9)],
            updates=updates)

    #save paramenters
    def save_model(self, outfile):
        np.savez(outfile,
            hidden_dim = self.hidden_dim,
            word_dim  = self.word_dim,
            batch_size = self.batch_size,
            E = self.params["E"].get_value(),
            W = self.params["W"].get_value(),
            B = self.params["B"].get_value(),
            lrW = self.params["lrW"].get_value(),
            lrb = self.params["lrb"].get_value())
        print "Saved LSTM" parameters to %s." % outfile

    def load_model(self, path):
        npzfile = np.load(path)
        print("Loading model from %s.\nParamenters: \n\t|hidden_dim=%d \n\t|word_dim=%d" 
                        % (path, npzfile["hidden_dim"], npzfile["word_dim"]))
        self.params["E"].set_value(npzfile["E"])
        self.params["W"].set_value(npzfile["W"])
        self.params["B"].set_value(npzfile["B"])
        self.params["lrW"].set_value(npzfile["lrW"])
        self.params["lrb"].set_value(npzfile["lrb"])
