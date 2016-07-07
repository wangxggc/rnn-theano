import theano.tensor as T


def lstm_cell(x, mx, ph, pc, Wh, hidden_dim):
    _x = ph.dot(Wh) + x
    i = T.nnet.hard_sigmoid(_x[:, hidden_dim * 0 : hidden_dim * 1])
    f = T.nnet.hard_sigmoid(_x[:, hidden_dim * 1 : hidden_dim * 2])
    o = T.nnet.hard_sigmoid(_x[:, hidden_dim * 2 : hidden_dim * 3])
    c = T.tanh(_x[:, hidden_dim * 3 : hidden_dim * 4])

    c = f * pc + i * c
    c = mx[:, None] * c + (1.- mx[:,None]) * pc

    h = o * T.tanh(c)
    h = mx[:, None] * h + (1.- mx[:,None]) * ph

    return [h, c]  # size = sample * hidden : 3 * 4


def lstm_generation_cell(x, ph, pc, Ex, Wx, Wh, bx, hidden_dim):
    _x = Ex[x, :].dot(Wx) + ph.dot(Wh) + bx
    i = T.nnet.hard_sigmoid(_x[hidden_dim * 0: hidden_dim * 1])
    f = T.nnet.hard_sigmoid(_x[hidden_dim * 1: hidden_dim * 2])
    o = T.nnet.hard_sigmoid(_x[hidden_dim * 2: hidden_dim * 3])
    c = T.tanh(_x[hidden_dim * 3: hidden_dim * 4])

    c = f * pc + i * c
    h = o * T.tanh(c)

    return [h, c]  # size = sample * hidden : 3 * 4


