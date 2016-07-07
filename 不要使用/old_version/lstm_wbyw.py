import numpy as np
import theano as theano
import theano.tensor as T
from theano.gradient import grad_clip
import time
import operator
from utils import *


class LSTMWByW:
    
    def __init__(self, classDim=0, wordDim=0, hiddenDim=0, senLen=0, batchSize=0, truncate=-1):
        # Assign instance variables
        # print classDim,wordDim,hiddenDim
        self.classDim  = classDim
        self.wordDim   = wordDim
        self.hiddenDim = hiddenDim
        self.senLen = senLen
        self.batchSize = batchSize
        self.truncate = truncate
        # Initialize the network parameters, Paper 2, 3
        E = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (wordDim, hiddenDim))        #Ebdding Matirx
        W = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (4, hiddenDim, hiddenDim * 4))   #W[0-1].dot(x), W[2-3].(i,f,o,c)
        B = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (2, hiddenDim * 4))               #B[0-1] for W.dot(x)
        U = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (7, hiddenDim, hiddenDim))   #attention U and w
        w = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (hiddenDim))
        V = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (2, hiddenDim, classDim))       #full connection V and b
        b = np.random.uniform(-np.sqrt(1./hiddenDim), np.sqrt(1./hiddenDim), (classDim))
        # Theano: Created shared variables
        self.E = theano.shared(value=E.astype(theano.config.floatX), name='E')
        self.W = theano.shared(value=W.astype(theano.config.floatX), name='W')
        self.B = theano.shared(value=B.astype(theano.config.floatX), name='B')
        self.U = theano.shared(value=U.astype(theano.config.floatX), name='U')
        self.w = theano.shared(value=w.astype(theano.config.floatX), name='w')
        self.V = theano.shared(value=V.astype(theano.config.floatX), name='V')
        self.b = theano.shared(value=b.astype(theano.config.floatX), name='b')
        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(value=np.zeros(E.shape).astype(theano.config.floatX), name='mE')
        self.mW = theano.shared(value=np.zeros(W.shape).astype(theano.config.floatX), name='mW')
        self.mB = theano.shared(value=np.zeros(B.shape).astype(theano.config.floatX), name='mB')
        self.mU = theano.shared(value=np.zeros(U.shape).astype(theano.config.floatX), name='mU')
        self.mw = theano.shared(value=np.zeros(w.shape).astype(theano.config.floatX), name='mw')
        self.mV = theano.shared(value=np.zeros(V.shape).astype(theano.config.floatX), name='mV')
        self.mb = theano.shared(value=np.zeros(b.shape).astype(theano.config.floatX), name='mb')
        # We store the Theano graph here
        self.__theano_build__()
    
    def __theano_build__(self):
        E, W, U, V, b, B, w, hiddenDim = self.E, self.W, self.U, self.V, self.b, self.B, self.w, self.hiddenDim

        x1  = T.imatrix('x1')    #first sentence
        x2  = T.imatrix('x2')    #second sentence
        x1_mask = T.fmatrix('x1_mask')    #mask sten_len * samples, 5 * 3
        x2_mask = T.fmatrix('x2_mask')
        y   = T.ivector('y')     #label
                    
        #Embdding words, from (sten_len * samples) to (sten_len * samples * hidden)
        ex1 = E[x1.flatten(), :].reshape([x1.shape[0], x1.shape[1], self.hiddenDim]) #5 * 3 --> 5 * 3 * 4
        ex2 = E[x2.flatten(), :].reshape([x2.shape[0], x2.shape[1], self.hiddenDim])
        #pre-calculate Wx, sten_len * samples * hidden
        statex1 = ex1.dot(W[0]) + B[0] #5 * 3 * (4*4)
        statex2 = ex2.dot(W[1]) + B[1]
        
        #encoder
        def LSTMLayer(x, mx, ph, pc):
            _x = ph.dot(W[2]) + x
            i  = T.nnet.hard_sigmoid(_x[:, hiddenDim * 0 : hiddenDim * 1])
            f  = T.nnet.hard_sigmoid(_x[:, hiddenDim * 1 : hiddenDim * 2])
            o  = T.nnet.hard_sigmoid(_x[:, hiddenDim * 2 : hiddenDim * 3])
            c  = T.tanh(_x[:, hiddenDim * 3 : hiddenDim * 4])
            
            c = f * pc + i * c
            c = mx[:, None] * c + (1.- mx[:,None]) * pc
            
            h = o * T.tanh(c)
            h = mx[:, None] * h + (1.- mx[:,None]) * ph
            
            return [h, c]
           
        [h1, c1], updates = theano.scan(
            fn=LSTMLayer,
            sequences=[statex1, x1_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([self.batchSize, self.hiddenDim])),
                          dict(initial=T.zeros([self.batchSize, self.hiddenDim]))])    
        
        _Mt = h1.dot(U[0])
        _Hd = h1.dimshuffle(2, 1, 0)
        #decoder
        def WByWLayer(x, mx, ph, pc, pr, pa, Mt, Hd):
            _x = ph.dot(W[3]) + x
            i  = T.nnet.hard_sigmoid(_x[:, hiddenDim * 0 : hiddenDim * 1])
            f  = T.nnet.hard_sigmoid(_x[:, hiddenDim * 1 : hiddenDim * 2])
            o  = T.nnet.hard_sigmoid(_x[:, hiddenDim * 2 : hiddenDim * 3])
            c  = T.tanh(_x[:, hiddenDim * 3 : hiddenDim * 4])
            
            c = f * pc + i * c
            c = mx[:, None] * c + (1.- mx[:,None]) * pc
            
            h = o * T.tanh(c)
            h = mx[:, None] * h + (1.- mx[:,None]) * ph
            
            #WbyW Attention
            a = h.dot(U[1]) + pr.dot(U[2])  # 3 * 4
            a = T.tanh(Mt + a) # 5 * 3 * 4 
            a = T.nnet.softmax(a.dot(w).T) # 3 * 5
            a = mx[:, None] * a + (1.-mx[:, None]) * pa
            
            #2016.05.30, Error: pr.dot(U[3]) should be T.tanh(pr.dot(U[3]))
            #r = (Hd * a).sum(axis=2).T + pr.dot(U[3]) # 4 * 3 * 5 --> 3 * 4
            
            r = T.tanh(pr.dot(U[3])) # 
            r = (Hd * a).sum(axis=2).T + r # 4 * 3 * 5 --> 3 * 4
            r = mx[:, None] * r + (1.-mx[:, None]) * pr
            
            return [h, c, r, a]
        
        [h2, c2, r, a], updates = theano.scan(
            fn=WByWLayer,
            sequences=[statex2, x2_mask],
            truncate_gradient=self.truncate,
            outputs_info=[dict(initial=T.zeros([self.batchSize, self.hiddenDim])),  #using h[end] to initialize the second LSTM
                          dict(initial=T.zeros([self.batchSize, self.hiddenDim])),
                          dict(initial=T.zeros([self.batchSize, self.hiddenDim])),
                          dict(initial=T.zeros([self.batchSize, self.senLen]))],
            non_sequences=[_Mt, _Hd])
        
        #final result
        _h = T.tanh(r[-1].dot(U[4]) + h2[-1].dot(U[5]) + h1[-1].dot(U[6]))   #final hidden state of LSTM1 and LSTM2
        _a = a.dimshuffle(2, 0, 1)
        
        _s = T.nnet.softmax(h1[-1].dot(V[0]) + _h.dot(V[1]) + b)
        _p = T.argmax(_s, axis=1)
        _c = T.nnet.categorical_crossentropy(_s, y)
        _l = T.sum(E**2) + T.sum(W**2) + T.sum(B**2)
        cost = T.sum(_c) + 0.01 * _l
        
        # Gradients
        dE = T.grad(cost, E)
        dW = T.grad(cost, W)
        dB = T.grad(cost, B)
        dU = T.grad(cost, U)
        dw = T.grad(cost, w)
        dV = T.grad(cost, V)
        db = T.grad(cost, b)
        
        # Assign functions
        self.bptt         = theano.function([x1, x2, x1_mask, x2_mask, y], [dE, dW, dB, dU, dw, dV, db])
        self.errors       = theano.function([x1, x2, x1_mask, x2_mask, y], _c)
        self.weights      = theano.function([x1, x2, x1_mask, x2_mask], _s)
        self.predictions  = theano.function([x1, x2, x1_mask, x2_mask], _p)
        self.attentions   = theano.function([x1, x2, x1_mask, x2_mask], _a)

        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mB = decay * self.mB + (1 - decay) * dB ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mw = decay * self.mw + (1 - decay) * dw ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        
        self.sgd_step = theano.function(
            [x1, x2, x1_mask, x2_mask, y, learning_rate, decay],#theano.In(decay, value=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (B, B - learning_rate * dB / T.sqrt(mB + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (w, w - learning_rate * dw / T.sqrt(mw + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (self.mE, mE),
                     (self.mW, mW),
                     (self.mB, mB),
                     (self.mU, mU),
                     (self.mw, mw),
                     (self.mV, mV),
                     (self.mb, mb)
                    ])
        
    #for classfication task, take loss from final prediction only

    def format_attention(self, x1, x2, y, i2w):
        att = attentions(self, x1, x2, y, self.senLen, self.batchSize)
        info = '\t'
        for idx in range(len(y)):
            for x in x1[idx]:
                info += i2w[x].encode('utf-8') + '\t'
            info += '\n'
            
            max1 = len(x1[idx]) if len(x1[idx])<self.senLen else self.senLen
            max2 = len(x2[idx]) if len(x2[idx])<self.senLen else self.senLen

            for i in range(max2):
                info += i2w[x2[idx][i]].encode('utf-8') + '\t'
                for j in range(max1):
                    info += str(att[idx][i][j])+'\t'
                info += '\n'
        return info + '\n\n'

    #save paramenters
    def save_model(self, outfile):
        np.savez(outfile,
            E = self.E.get_value(),
            W = self.W.get_value(),
            B = self.B.get_value(),
            U = self.U.get_value(),
            w = self.w.get_value(),
            V = self.V.get_value(),
            b = self.b.get_value())
        print "Saved WbyW LSTM' parameters to %s." % outfile

    def load_model(self, path):
        npzfile = np.load(path)
        E, W, B, U, w, V, b = npzfile["E"], npzfile["W"], npzfile["B"], npzfile["U"], npzfile["w"], npzfile["V"], npzfile["b"]
        self.hiddenDim, self.wordDim, self.classDim = E.shape[1], E.shape[0], V.shape[2]
        print "Loading model from %s.\nParamenters: \t|classDim=%d \t|hiddenDim=%d \t|wordDim=%d" % (path, self.classDim, self.hiddenDim, self.wordDim)
        sys.stdout.flush()
        self.E.set_value(E)
        self.W.set_value(W)
        self.B.set_value(B)
        self.U.set_value(U)
        self.w.set_value(w)
        self.V.set_value(V)
        self.b.set_value(b)


