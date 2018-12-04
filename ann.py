import pickle
from graphviz import Digraph
import numpy as np
import utils
from pprint import pformat
            
class Layer:
    def __init__(self, a_func='tanh', learning_rate=0.01):
        self.f_func = a_func
        self.a_func = utils.a_func_dict[a_func]['func']
        self.d_a_func = utils.a_func_dict[a_func]['d_func']
        self.learning_rate = learning_rate
        self.in_dim = self.out_dim = None

    def activate(self, a_in):
        print('\'activate\' is not implemented for', type(self))


    def backpropagate(self, d_a):
        return np.zeros(self.in_dim)

    def __repr__(self):
        return pformat(vars(self))

    def update(self):
        pass

class Dense(Layer):
    def __init__(self, out_dim, **kwargs):
        super().__init__(**kwargs)
        self.out_dim = np.array(out_dim)
        self.z_out = None
        self.weights = None
        self.a_in = None

    def init(self, in_dim):
        self.in_dim = in_dim.copy()
        self.weights = np.random.uniform(-1, 1, (self.out_dim, self.in_dim+1))
        self.d_w = np.zeros(self.weights.shape)
        return self.out_dim

    def activate(self, a_in):
        self.a_in = np.concatenate([[1], a_in]) #constant 1 for the bias
        self.z_out = self.weights @ self.a_in
        return self.a_func(self.z_out)

    def backpropagate(self, d_a):
        deltas = self.d_a_func(self.z_out) * d_a
        self.d_w += self.learning_rate * np.outer(deltas, self.a_in)
        d_a_prev = self.weights.T @ deltas
        return d_a_prev[1:]

    def update(self):
        try:
            self.weights += self.d_w
        except:
            print('dw:', self.dw)
        self.d_w_prev = self.d_w[:]
        self.d_w = np.zeros(self.weights.shape)
        
class Conv2D(Layer):
    def __init__(self, n_filters=5, filter_size=5, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.stride = stride
        self.filters = None

    def init(self, in_dim):
        assert len(in_dim) == 3, 'input is not 3D'
        self.in_dim = in_dim.copy()
        filter_out_dim = self.in_dim[1:] - self.filter_size + 1
        self.out_dim  = (self.n_filters, *filter_out_dim)
        filter_shape = (self.in_dim[0], self.filter_size, self.filter_size)
        self.filters = np.random.uniform(-1, 1, (self.n_filters, *filter_shape))
        self.d_w = np.zeros(self.filters.shape)
        return self.out_dim

    def activate(self, a_in):
        self.a_in = a_in
        self.z_out = [utils.conv3d(a_in, f) for f in self.filters]
        return self.a_func(self.z_out)

    def backpropagate(self, d_in):
        deltas = self.d_a_func(self.z_out) * d_in
        self.d_w += self.learning_rate * utils.conv3d(self.a_in, deltas)
        return np.array([np.sum([utils.conv2d(f[i], deltas[f_idx], True, True)
            for f_idx, f in enumerate(self.filters)], axis=0) for i in
            range(self.in_dim[0])])

    def update(self):
        self.filters += self.d_w
        self.d_w = np.zeros(self.d_w.shape)


class MaxPooling(Layer):
    def __init__(self, pool_size=5):
        super().__init__('identity', 0.)
        self.pool_size = pool_size

    def init(self, in_dim):
        self.in_dim = in_dim.copy()
        self.out_dim = in_dim.copy()
        self.out_dim[0] = 1
        self.out_dim[1] = int(np.ceil(self.out_dim[1] / float(self.pool_size)))
        self.out_dim[2] = int(np.ceil(self.out_dim[2] / float(self.pool_size)))
        return self.out_dim
    
    def activate2D(self, mat_in):
        x = [[np.max(mat_in[row:row+self.pool_size, col:col+self.pool_size])
            for row in range(0, mat_in.shape[0], self.pool_size)]
            for col in range(0, mat_in.shape[1], self.pool_size)
            ]
        return np.array(x)

    def activate(self, a_in):
        return np.array([np.sum([self.activate2D(channel) for channel in a_in],
            0)])

class Flatten(Layer):
    def init(self, in_dim):
        self.in_dim = in_dim.copy()
        self.out_dim = np.prod(self.in_dim)
        return self.out_dim

    def activate(self, a_in):
        return a_in.flatten()

    def backpropagate(self, d_in):
        return d_in.reshape(self.in_dim)

class Max(Layer):
    def init(self, in_dim):
        self.in_dim = in_dim.copy()
        self.out_dim = 1

    def activate(self, a_in):
        return np.argmax(a_in)

class Model:
    def __init__(self, in_dim=2, batch_size=10, **kwargs):
        self.layers = []
        self.batch_size = batch_size
        self.next_in_dim = np.array(in_dim)
        self.kwargs = kwargs
        self.l_types = {
                'Dense' : Dense,
                'Conv2D': Conv2D, 
                }

    def reset(self):
        for l in self.layers:
            l.init(l.in_dim)

    def set_learning_rate(self, lr):
        for l in self.layers:
            l.learning_rate = lr

    def add(self, layer):
        self.layers.append(layer)
        self.next_in_dim = np.array(layer.init(self.next_in_dim))

    def pred(self, vec):
        for layer in self.layers:
           vec = layer.activate(vec)
        return vec

    def pred_vec(self, vec, printouts = False):
        return np.array([self.pred(x) for x in vec])

    def fit(self, xs, ys, n_epochs=1, printouts = False):
        for i in range(n_epochs):
            if(printouts):
                print('epoch: \t', i, 'loss', self.loss(xs, ys))
            for idx, (x, y), in enumerate(zip(xs, ys)):
                d_a = y - np.clip(self.pred(x), 0, 1)
                for layer in reversed(self.layers):
                    d_a = layer.backpropagate(d_a)
                if idx % self.batch_size == 0 or idx == len(ys)-1:
                    for layer in self.layers:
                        layer.update()

    def loss(self, xs, ys):
        squared_errors =[(self.pred(x) - y)**2 for x, y in zip(xs, ys)]
        return sum(sum(squared_errors)) / len(squared_errors)
    
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)



    def show(self):
        g = Digraph()
        g.graph_attr['rankdir'] = 'LR'
        for l_idx, layer in enumerate(self.layers):
            g.node("{}_{}".format(l_idx, 0), "Bias")
            for j in range(len(layer.weights[0])):
                for i in range(len(layer.weights)):
                    g.edge("{}_{}".format(l_idx, j),
                           "{}_{}".format(l_idx+1, i+1),
                           label="{:.3f}".format(layer.weights[i][j]))
        g.render("graph", view=True)

