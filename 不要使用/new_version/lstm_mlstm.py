import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
from utils import *


class LSTMMLstm:
    
    def __init__(self, class_dim=0, word_dim=0, hidden_dim=0, sen_len=0, batch_size=0, truncate=-1):
        # Assign instance variables
        # print class_dim,word_dim,hidden_dim
        self.class_dim  = class_dim
        self.word_dim   = word_dim
        self.hidden_dim = hidden_dim
        self.sen_len    = sen_len
        self.batch_size = batch_size
        self.truncate   = truncate
        params = {}
        # Initialize the network parameters
        params["E"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))            #Ebdding Matirx
        params["W"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim * 4))   #W[0-1].dot(x), W[2-3].(i,f,o,c)
        params["B"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim * 4))              #B[0-1] for W.dot(x)
        # Attention paramenters
        params["AttenW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (4, hidden_dim, hidden_dim))       #attention U and w
        params["Attenw"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim))
        # MLSTM paramenters
        params["MLstmW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim, hidden_dim * 4))
        params["MLstmV"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (sen_len, hidden_dim * 4))
        params["MLstmb"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim * 4))
        # Final full connection paramenters
        params["lrW"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (2, hidden_dim, class_dim))           #full connection V and b
        params["lrb"] = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (class_dim))
        # Assign paramenters' names 
        self.param_names = {"orign":["E",  "W",  "B",  "AttenW",  "Attenw",  "MLstmW",  "MLstmV",  "MLstmb",  "lrW",  "lrb"], 
                            "cache":["mE", "mW", "mB", "mAttenW", "mAttenw", "mMLstmW", "mMLstmV", "mMLstmb", "mlrW", "mlrb"]}
        # Theano: Created shared variables
        self.params = {}
        # Model's shared variables
        for _n in self.param_names["orign"]:
            self.params[_n] = theano.shared(value=params[_n].astype(theano.config.floatX), name=_n)
        # Shared variables for RMSProp
        for _n in self.param_names["cache"]:
            self.params[_n] = theano.shared(value=np.zeros(params[_n[1:]].shape).astype(theano.config.floatX), name=_n)
        # We store the Theano graph here
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
        # (sten_len * samples * hidden)
        statex1 = _E1[x1.flatten(), :].reshape([x1.shape[0], x1.shape[1], hidden_dim * 4])
        statex2 = _E2[x2.flatten(), :].reshape([x2.shape[0], x2.shape[1], hidden_dim * 4])
        
        #encoder
        def LSTMLayer(x, mx, ph, pc, Wh):
            _x = ph.dot(Wh) + x
            i  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 0 : hidden_dim * 1])
            f  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 1 : hidden_dim * 2])
            o  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 2 : hidden_dim * 3])
            c  = T.tanh(_x[:, hidden_dim * 3 : hidden_dim * 4])
            
            c = f * pc + i * c
            c = mx[:, None] * c + (1.- mx[:,None]) * pc
            
            h = o * T.tanh(c)
            h = mx[:, None] * h + (1.- mx[:,None]) * ph
            
            return [h, c]

        #decoder
        def WByWLayer(x, mx, ph, pc, pr, pa, Mt, Hd, Wh, AW, Aw):
            _x = ph.dot(Wh) + x
            i  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 0 : hidden_dim * 1])
            f  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 1 : hidden_dim * 2])
            o  = T.nnet.hard_sigmoid(_x[:, hidden_dim * 2 : hidden_dim * 3])
            c  = T.tanh(_x[:, hidden_dim * 3 : hidden_dim * 4])
            
            c = f * pc + i * c
            c = mx[:, None] * c + (1.- mx[:,None]) * pc
            
            h = o * T.tanh(c)
            h = mx[:, None] * h + (1.- mx[:,None]) * ph
            
            #WbyW Attention
            a = h.dot(AW[1]) + pr.dot(AW[2])  # batch * hidden
            a = T.tanh(Mt + a)   # sen_len * batch * hidden
            a = T.nnet.softmax(a.dot(Aw).T) # batch * sen_len
            a = mx[:, None] * a + (1.-mx[:, None]) * pa
            
            r = T.tanh(pr.dot(AW[3])) # batch * hidden
            r = (Hd * a).sum(axis=2).T + r # (hidden, batch, sen_len) * (batch, sen_len) == > (batch, hidden) 
            r = mx[:, None] * r + (1.-mx[:, None]) * pr
            
            return [h, c, r, a]
            
        [h1, c1], updates = theano.scan(
            fn=LSTMLayer,
            sequences=[statex1, x1_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=[params["W"][2]])    
        
        _Mt = h1.dot(params["AttenW"][0])
        _Hd = h1.dimshuffle(2, 1, 0)
        
        [h2, c2, r, a], updates = theano.scan(
            fn=WByWLayer,
            sequences=[statex2, x2_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=h1[-1]),
                          # dict(initial=T.zeros([self.batch_size, self.hidden_dim])),  #using h[end] to initialize the second LSTM
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.sen_len]))],
            non_sequences=[_Mt, _Hd, params["W"][3], params["AttenW"], params["Attenw"]])
        
        _Mx = h2.dot(params["MLstmW"][0]) + a.dot(params["MLstmV"]) + params["MLstmb"] #
        #final result
        #use lstm layer for scan
        [h3, c3], updates = theano.scan(
            fn=LSTMLayer,
            sequences=[_Mx, x2_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=[params["MLstmW"][1]]) 
        
        _a = a
        _s = T.nnet.softmax(h2[-1].dot(params["lrW"][0]) + h3[-1].dot(params["lrW"][1]) + params["lrb"])
        _p = T.argmax(_s, axis=1)
        _c = T.nnet.categorical_crossentropy(_s, y)
        _c = _c * y_c
        _l = T.sum(params["lrW"]**2) + T.sum(params["lrb"] ** 2)
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
        self.attentions  = theano.function([x1, x2, x1_mask, x2_mask], _a)
        self.sgd_step = theano.function(
            [x1, x2, x1_mask, x2_mask, y, y_c, learning_rate, decay], #theano.In(decay, value=0.9)],
            updates=updates)
        

    #save paramenters
    def save_model(self, outfile):
        np.savez(outfile,
            batch_size = self.batch_size,
            sen_len    = self.sen_len,
            class_dim  = self.class_dim,
            hidden_dim = self.hidden_dim,
            E = self.params["E"].get_value(),
            W = self.params["W"].get_value(),
            B = self.params["B"].get_value(),
            AttenW = self.params["AttenW"].get_value(),
            Attenw = self.params["BAttenw"].get_value(),
            MLstmW = self.params["MLstmW"].get_value,
            MLstmV = self.params["MLstmV"].get_value,
            MLstmb = self.params["MLstmb"].get_value,
            lrW = self.params["V"].get_value(),
            lrb = self.params["b"].get_value())
        print "Saved WbyW LSTM' parameters to %s." % outfile

    def load_model(self, path):
        npzfile = np.load(path)
        print "Loading model from %s.\nParamenters: \n\t|class_dim=%d \n\t|hidden_dim=%d \n\t|word_dim=%d" % (path, npzfile["class_dim"], npzfile["hidden_dim"], npzfile["word_dim"])
        sys.stdout.flush()
        self.params["E"].set_value(npzfile["E"])
        self.params["W"].set_value(npzfile["W"])
        self.params["B"].set_value(npzfile["B"])
        self.params["AttenW"].set_value(npzfile["AttenW"])
        self.params["Attenw"].set_value(npzfile["Attenw"])
        self.params["MLstmW"].set_value(npzfile["MLstmW"])
        self.params["MLstmV"].set_value(npzfile["MLstmV"])
        self.params["MLstmb"].set_value(npzfile["MLstmb"])
        self.params["lrW"].set_value(npzfile["lrW"])
        self.params["lrb"].set_value(npzfile["lrb"])


