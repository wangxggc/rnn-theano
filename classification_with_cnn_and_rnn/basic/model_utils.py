import theano.tensor as T
import theano
import numpy as np
import os


def rms_prop(cost, params_names, params, learning_rate, decay):
    # Gradients:
    # e.g. dE = T.grad(cost, E)
    _grads  = [T.grad(cost, params[_n])
                for _i, _n in enumerate(params_names["orign"])]
    # RMSProp caches: 
    # e.g. mE = decay * mE + (1 - decay) * dE ** 2
    _caches = [decay * params[_n] + (1 - decay) * _grads[_i] ** 2 
                for _i, _n in enumerate(params_names["cache"])]
    # Learning rate: 
    # e.g. (E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
    _update_orign = [(params[_n], params[_n] - learning_rate * _grads[_i] / T.sqrt(_caches[_i] + 1e-6))
                for _i, _n in enumerate(params_names["orign"])]
    # Update cache
    # e.g. (mE, mE)
    _update_cache = [(params[_n], _caches[_i])
                for _i, _n in enumerate(params_names["cache"])]
    # Merge all updates
    _updates = _update_orign + _update_cache
    
    return _grads, _updates


def share_params(param_names, params):
    """
    make shared params by theano.shared

    Args:
        param_names:
                 contain 2 types param names, one is orign, the other is cache
                 orign is real parameters that used for this model
                 cache is temporal parameter for rmsprop algorithm
                 both two parameters shared the same shape with parameters in param with idx for id
        params:
                 real parameters that used for making shared parameters

    Returns: shared params

    """
    shared_params = {}
    for _n1, _n2 in zip(param_names["orign"], param_names["cache"]):
        # Theano shared Model's params
        shared_params[_n1] = theano.shared(value=params[_n1].astype(theano.config.floatX), name=_n1)
        # Theano shared Model's params for RMSProp
        shared_params[_n2] = theano.shared(value=np.zeros(params[_n1].shape).astype(theano.config.floatX), name=_n2)
        # print _n1, params[_n1].shape
    return shared_params


def init_weights(shape, low_bound=None, up_bound=None):
    """
    Initialize weights with shape
    Args:
        shape:
        low_bound:
        up_bound:
    Returns:
    """
    if not low_bound or not up_bound:
        low_bound = -np.sqrt(0.01)
        up_bound = -low_bound
    return np.random.uniform(low_bound, up_bound, shape)


def save_model(path, model):
    if os.path.exists(path) and not os.path.isdir(path):
        os.remove(path)
    if not os.path.exists(path):
        os.mkdir(path)

    sout = open(path+ "/setting.txt", "w")
    print >> sout, model.hidden_dim
    print >> sout, model.embed_dim
    print >> sout, model.batch_size
    print >> sout, model.output_dim
    print >> sout, model.id

    import json

    sout.close()

    for name in model.param_names["orign"]:
        np.save(path + "/" + name + ".npy", model.params[name].get_value())

    print "Saved " + str(model) + " parameters to %s." % path


def load_model(path, model):
    # sin = open(path + "/setting.txt")
    # model.hidden_dim = int(sin.readline().strip())
    # model.embed_dim = int(sin.readline().strip())
    # model.batch_size = int(sin.readline().strip())
    # model.output_dim = int(sin.readline().strip())

    for name in model.param_names["orign"]:
        model.params[name].set_value(np.load(path + "/" + name + ".npy"))
        print name, model.params[name].get_value().shape

    print "Loaded " + str(model) + " parameters from %s." % path
