import numpy as np
from numpy.random import uniform
import theano.tensor as T
import theano
import model_utils


def BasicLSTMCell(x, mx, ph, pc, wx, wh, b, hidden_dim):
    """
    basic lstm cell for encoder and decoder
    Args:
        x: batched input, column stands for a batch
        mx: batched input max, defines sentence length
        ph: pre-calculated hidden states for this batch
        pc: pre-calculated cell states for this btach
        W: gates weights for x and ph
        B: gates bais for hidden state
        hidden_dim:

    Returns:
    """
    x = x.dot(wx) + ph.dot(wh) + b
    i = T.nnet.sigmoid(x[:, hidden_dim * 0:hidden_dim * 1])
    f = T.nnet.sigmoid(x[:, hidden_dim * 1:hidden_dim * 2])
    o = T.nnet.sigmoid(x[:, hidden_dim * 2:hidden_dim * 3])
    c = T.tanh(x[:, hidden_dim * 3:hidden_dim * 4])
   
    # i = theano.printing.Print('i')(i)
    # f = theano.printing.Print('f')(f)
    # o = theano.printing.Print('o')(o)
    # c = theano.printing.Print('c')(c)

    c = f * pc + i * c
    c = mx[:, None] * c + (1. - mx[:, None]) * pc
    
    h = o * T.tanh(c)
    h = mx[:, None] * h + (1. - mx[:, None]) * ph

    return [h, c]


def WByWLSTMCell(x, ph, pc, pr, cy, cwy, ux, uh, ub, wh, wr, wt, wa, hidden_dim):
    """
    Attention based Cell implemented based on Paper Word By Word Attention
    The Original Paper is REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION,
    URL, http://arxiv.org/pdf/1509.06664.pdf
    Noted that there is no mask on Attention LSTM Cell which is only used for
    Args:
        x: embedded input, (batch, input)
        ph: pre-hidden states, (batch, hidden)
        pc: pre-cell states, (batch, hidden)
        pr: pre-representation, (batch, context)
        cy: Context with shuffled dimension (2, 1, 0), (context, batch, max_len)
        cwy: Context By Y.dot(W^{y}), (max_len, batch_size, hidden)
        ux: LSTM Ux, (input, hidden * 4)
        uh: LSTM Uh, (hidden, hidden * 4)
        ub: LSTM b, (hidden * 4)
        wh: (hidden, hidden)
        wr: (context, hidden)
        wt: (context, context)
        wa: (1, hidden)
        hidden_dim:
    Returns:
    """
    x = x.dot(ux) + ph.dot(uh) + ub
    i = T.nnet.sigmoid(x[:, hidden_dim * 0:hidden_dim * 1])
    f = T.nnet.sigmoid(x[:, hidden_dim * 1:hidden_dim * 2])
    o = T.nnet.sigmoid(x[:, hidden_dim * 2:hidden_dim * 3])
    c = T.tanh(x[:, hidden_dim * 3:hidden_dim * 4])

    c = f * pc + i * c
    h = o * T.tanh(c)

    # Word by Word Attention
    # (max_len, batch, hidden)
    M = T.tanh(cwy + h.dot(wh) + pr.dot(wr))
    # (batch, max_len)
    a = T.nnet.softmax(M.dot(wa).T)
    # (batch, context), r is a weighted context representation
    r = (cy * a).sum(axis=2).T + T.tanh(pr.dot(wt))

    return [h, c, r, a]


def AttentionLSTMCell(x, ph, pc, pr, cy, cwy, ux, uh, uc, ub, wh, wa, hidden_dim):
    """
    Attention based Cell implemented based on Paper Word By Word Attention
    The Original Paper is NEURAL MACHINE TRANSLATION
                    BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
    Noted that there is no mask on Attention LSTM Cell which is only used for
    Args:
        x: embedded input, (batch, input)
        ph: pre-hidden states, (batch, hidden)
        pc: pre-cell states, (batch, hidden)
        pr: pre-weighted context
        cy: Context with shuffled dimension (2, 1, 0), (context, batch, max_len)
        cwy: Context By Y.dot(W^{y}), (max_len, batch_size, hidden)
        ux: LSTM Ux, (input, hidden * 4)
        uh: LSTM Uh, (hidden, hidden * 4)
        uc, LSTM Uc, for context, (context, hidden * 4)
        ub: LSTM b, (hidden * 4)
        wh: (hidden, hidden)
        wa: (1, hidden)
        hidden_dim:
    Returns:
    """
    x = x.dot(ux) + ph.dot(uh) + pr.dot(uc) + ub
    i = T.nnet.sigmoid(x[:, hidden_dim * 0:hidden_dim * 1])
    f = T.nnet.sigmoid(x[:, hidden_dim * 1:hidden_dim * 2])
    o = T.nnet.sigmoid(x[:, hidden_dim * 2:hidden_dim * 3])
    c = T.tanh(x[:, hidden_dim * 3:hidden_dim * 4])

    c = f * pc + i * c
    h = o * T.tanh(c)

    # Basic Attention
    # (max_len, batch, hidden)
    M = T.tanh(cwy + h.dot(wh))
    # (batch, max_len)
    a = T.nnet.softmax(M.dot(wa).T)
    # (batch, context), r is a weighted context representation
    r = (cy * a).sum(axis=2).T

    return [h, c, r, a]

