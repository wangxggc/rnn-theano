import numpy as np
import theano as theano
import theano.tensor as T
from utils import *

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
        params = {}
        # Initialize the network parameters
        params["E"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))          #Ebdding Matirx
        params["W"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim * 4)) #W[0-1].dot(x), W[2-3].(i,f,o,c)
        params["B"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim * 4))             #B[0-1] for W[0-1]
        params["lrW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim, class_dim))         #LR W and b
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
        statex1 = _E1[x1.flatten(), :].reshape([x1.shape[0], x1.shape[1], hidden_dim * 4])
        statex2 = _E2[x2.flatten(), :].reshape([x2.shape[0], x2.shape[1], hidden_dim * 4])
        
        def LSTMLayer(x, mx, ph, pc, Wh):
            _x = ph.dot(Wh) + x
            i  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 0 : hidden_dim * 1])
            f  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 1 : hidden_dim * 2])
            o  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 2 : hidden_dim * 3])
            c  = T.tanh(_x[:, hidden_dim * 3 : hidden_dim * 4])
            
            c  = f * pc + i * c
            c  = mx[:, None] * c + (1.- mx[:,None]) * pc
            
            h  = o * T.tanh(c)
            h  = mx[:, None] * h + (1.- mx[:,None]) * ph
            
            return [h, c]  # size = sample * hidden : 3 * 4
            
        [h1, c1], updates = theano.scan(
            fn=LSTMLayer,
            sequences=[statex1, x1_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=params["W"][2])
        
        [h2, c2], updates = theano.scan(
            fn=LSTMLayer,
            sequences=[statex2, x2_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=h1[-1]),  #using h[end] to initialize the second LSTM
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=params["W"][3])
       
        #predict
        _s = T.nnet.softmax(h1[-1].dot(params["lrW"][0]) + h2[-1].dot(params["lrW"][1]) + params["lrb"])
        _p = T.argmax(_s, axis=1)
        _c = T.nnet.categorical_crossentropy(_s, y)
        _c = _c * y_c
        _l = T.sum(params["lrW"]**2)
        _cost = T.sum(_c) + 0.01 * _l
        
        # SGD parameters
        learning_rate  = T.scalar('learning_rate')
        decay         = T.scalar('decay')
        
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
