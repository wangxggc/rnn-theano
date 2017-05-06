import numpy as np
import theano as theano
import theano.tensor as T
from model_utils import *

class RNN:
    
    def __init__(self, class_dim, word_dim, hidden_dim, sen_len, batch_size, truncate=-1):
        # Assign instance variables
        self.class_dim = class_dim
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.sen_len = sen_len
        self.batch_size = batch_size
        self.truncate = truncate
        params = {}
        # Initialize the network parameters
        params["E"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))        #Ebdding Matirx
        params["W"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim))     #W[0-1].dot(x), W[2-3].(i,f,o,c)
        params["B"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim))               #B[0-1] for W[0-1]
        params["lrW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim, class_dim))  #LR W and b
        params["lrb"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (class_dim))
        
        # Assign paramenters' names 
        self.param_names = {"orign":["E", "W", "B", "lrW", "lrb"], 
                           "cache":["mE", "mW", "mB", "mlrW", "mlrb"]}
        # Theano: Created shared variables
        self.params = {}
        # Model's shared variables
        for _n in self.param_names["orign"]:
            self.params[_n] = theano.shared(value=params[_n].astype(theano.config.floatX), name=_n)
        # Shared variables for RMSProp
        for _n in self.param_names["cache"]:
            self.params[_n] = theano.shared(value=np.zeros(params[_n[1:]].shape).astype(theano.config.floatX), name=_n)
        # Build model graph
        self.__theano_build__()

    def __theano_build__(self):
        params = self.params
        param_names = self.param_names
        hidden_dim = self.hidden_dim

        x1  = T.imatrix('x1')    # first sentence
        x2  = T.imatrix('x2')    # second sentence
        x1_mask = T.fmatrix('x1_mask')    #mask
        x2_mask = T.fmatrix('x2_mask')
        y   = T.ivector('y')     # label
        y_c = T.ivector('y_c')   # class weights 
        
        # Embdding words
        _E1 = params["E"].dot(params["W"][0]) + params["B"][0]
        _E2 = params["E"].dot(params["W"][1]) + params["B"][1]
        statex1 = _E1[x1.flatten(), :].reshape([x1.shape[0], x1.shape[1], hidden_dim])
        statex2 = _E2[x2.flatten(), :].reshape([x2.shape[0], x2.shape[1], hidden_dim])
        
        def rnn_cell(x, mx, ph, Wh):
            h = T.tanh(ph.dot(Wh) + x)
            h = mx[:, None] * h + (1-mx[:, None]) * ph
            return [h] 
            
        [h1], updates = theano.scan(
            fn=rnn_cell,
            sequences=[statex1, x1_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=params["W"][2])
        
        [h2], updates = theano.scan(
            fn=rnn_cell,
            sequences=[statex2, x2_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=h1[-1])],
            non_sequences=params["W"][3])
       
        #predict
        _s = T.nnet.softmax(h1[-1].dot(params["lrW"][0]) + h2[-1].dot(params["lrW"][1]) + params["lrb"])
        _p = T.argmax(_s, axis=1)
        _c = T.nnet.categorical_crossentropy(_s, y)
        _c = T.sum(_c * y_c)
        _l = T.sum(params["lrW"]**2)
        _cost = _c + 0.01 * _l
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # Gradients and updates
        _grads, _updates = rms_prop(_cost, param_names, params, learning_rate, decay)
        
        # Assign functions
        self.bptt = theano.function([x1, x2, x1_mask, x2_mask, y, y_c], _grads)
        self.loss = theano.function([x1, x2, x1_mask, x2_mask, y, y_c], _c)
        self.weights = theano.function([x1, x2, x1_mask, x2_mask], _s)
        self.predictions = theano.function([x1, x2, x1_mask, x2_mask], _p)
        self.sgd_step = theano.function(
            [x1, x2, x1_mask, x2_mask, y, y_c, learning_rate, decay],
            updates=_updates)

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
        print "Saved LSTM' parameters to %s." % outfile

    def load_model(self, path):
        npzfile = np.load(path)
        print("Loading model from %s.\nParamenters: \n\t|hidden_dim=%d \n\t|word_dim=%d" 
                        % (path, npzfile["hidden_dim"], npzfile["word_dim"]))
        self.params["E"].set_value(npzfile["E"])
        self.params["W"].set_value(npzfile["W"])
        self.params["B"].set_value(npzfile["B"])
        self.params["lrW"].set_value(npzfile["lrW"])
        self.params["lrb"].set_value(npzfile["lrb"])
