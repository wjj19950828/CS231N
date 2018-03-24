import numpy as np

def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config

def sgd_momentum(w,dw,config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    config['velocity'] = v
    return next_w,config

def rmsprop(x,dx,config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    config['cache'] = config['decay_rate']*config['cache'] + (1 - config['decay_rate'])*dx*dx
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])
    return next_x,config

def adam(x,dx,config=None):
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 1)

    next_x = None
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1'])*dx
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * dx * dx
    next_x = x - config['learning_rate'] * config['m'] / (np.sqrt(config['v']) + config['epsilon'])

    return next_x , config
