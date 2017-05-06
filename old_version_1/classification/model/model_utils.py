import theano.tensor as T


def rms_prop(cost, params_names, params, learning_rate, decay):
    # Gradients:
    # e.g. dE = T.grad(cost, E)
    _grads  = [T.grad(cost, params[_n])
                for _i, _n in enumerate(params_names["orign"])]
    # RMSProp caches: 
    # e.g. mE = decay * self.mE + (1 - decay) * dE ** 2
    _caches = [decay * params[_n] + (1 - decay) * _grads[_i] ** 2 
                for _i, _n in enumerate(params_names["cache"])]
    # Learning rate: 
    # e.g. (E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
    _update_orign = [(params[_n], params[_n] - learning_rate * _grads[_i] / T.sqrt(_caches[_i] + 1e-6))
                for _i, _n in enumerate(params_names["orign"])]
    # Update cache
    # e.g. (self.mE, mE)
    _update_cache = [(params[_n], _caches[_i])
                for _i, _n in enumerate(params_names["cache"])]
    # Merge all updates
    _updates = _update_orign + _update_cache
    
    return _grads, _updates