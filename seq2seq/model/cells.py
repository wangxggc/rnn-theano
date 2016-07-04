import numpy as np
import theano as theano
import theano.tensor as T
from utils import *

#Paper: REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION
class Cells:
    
    def LSTMCell(x, mx, ph, pc, W):
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
            