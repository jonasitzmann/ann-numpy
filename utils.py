import numpy as np
import ann
import pickle
from scipy.signal import convolve2d
import cv2
import os
import matplotlib.pyplot as plt
plt.gray()

def imshow(img):
    plt.imshow(img[0])
    plt.show()

def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

def get_mnist_samples(n, n_classes = 10):
    n = int(n/n_classes)
    img_paths = ['mnist/trainingSet/{}/'.format(i) for i in range(n_classes)]
    xs = np.concatenate([[
        [cv2.imread(class_path + filename)[:,:,0]] for filename in os.listdir(class_path)[:n]
        ] for class_path in img_paths])
    ys = np.concatenate([np.ones(n)*i for i in range(n_classes)])
    ys = np.array([[int(i==y) for i in range(n_classes)] for y in ys])
    #shuffle_in_unison(xs, ys)
    return xs, ys

def get_edge_filters():
    return np.array([
        [[[ 1, 1], [-1,-1]]],
        [[[ 1,-1], [ 1,-1]]],
        [[[-1, 1], [ 1,-1]]],
        ], dtype = np.float64)

def conv2d(mat, kernel, reverse=True, full=False):
    assert len(mat.shape) == len(kernel.shape) == 2, 'invalid dimensions in conv2d : {} and {}'.format(mat.shape, kernel.shape)
    kernelOrdered = kernel if not reverse else kernel[::-1, ::-1] 
    mode = 'full' if full else 'valid'
    return convolve2d(mat, kernelOrdered, mode)

def conv3d(mat, kernel):
    return np.sum([conv2d(mat[i], channel) for i, channel in enumerate(kernel)], 0)

bound = 1000
def bounded(func):
    def wrapper(*args):
        return np.clip(func(*args), -bound, bound)
    return wrapper


# activation functions and its derivatives
@bounded
def relu(x):
    return max(0, x)

@bounded
def d_relu(x):
    return 1 if x > 0 else 0

@bounded
def leaky_relu(x):
    return x if x > 0 else 0.01*x

@bounded
def d_leaky_relu(x):
    return 1. if x>0 else 0.01

@bounded
def leaky_bounded_relu(x):
    if x < 0:
        return 0.01*x
    elif x < 1:
        return x
    else:
        return 1+0.01*(x-1)

@bounded
def d_leaky_bounded_relu(x):
    return 1 if 0 < x < 1 else 0.01

@bounded
def sigmoid(x):
    return 1/(1+np.exp(-x))

@bounded
def d_sigmoid(x):
    exp_neg_x = np.exp(-x)
    return (exp_neg_x)/(exp_neg_x + 1)**2

@bounded
def tanh(x):
    return 2*sigmoid(2*x)-1

@bounded
def d_tanh(x):
    return 1 - tanh(x)**2

@bounded
def softmax_vec(xs):
    ys = np.array([np.exp(x) for x in xs])
    ys = ys / sum(ys)
    return ys


def d_softmax_vec(xs, s=None): 
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x. 
    # s.shape = (1, n) 
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else: 
                jacobian_m[i][j] = -s[i]*s[j]
    result = jacobian_m @ xs
    return result

a_func_dict = {
        'relu': {
            'func'  : np.vectorize(relu),
            'd_func': np.vectorize(d_relu),
            },
        'leaky_relu': {
            'func'  : np.vectorize(leaky_relu),
            'd_func': np.vectorize(d_leaky_relu),
            },
        'leaky_bounded_relu': {
            'func'  : np.vectorize(leaky_bounded_relu),
            'd_func': np.vectorize(d_leaky_bounded_relu),
            },
        'sigmoid' : {
            'func'  : np.vectorize(sigmoid),
            'd_func': np.vectorize(d_sigmoid),
            },
        'tanh' : {
            'func'  : np.vectorize(tanh),
            'd_func': np.vectorize(d_tanh),
            },
        'identity' : {
            'func'  : np.vectorize(lambda x: x),
            'd_func': np.vectorize(lambda x: 1)
            },
        'softmax' : {
            'func'  : softmax_vec,
            'd_func': d_softmax_vec,
            },
        }

# general utility functions
def load(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

def get_samples_from_func(func, n_samples, n_variables, val_range = (-2, 2)):
    v_func = np.vectorize(func)
    xs = np.random.rand(n_samples, n_variables) * val_range[1]-val_range[0]
    xs += val_range[0]
    ys = v_func(*zip(*xs))
    return xs, ys

def split_data(xs, ys, test_proportion):
    shuffle_in_unison(xs, ys)
    split_idx = int(len(xs) * test_proportion)
    x_test = xs[:split_idx]
    y_test = ys[:split_idx]
    x_train = xs[split_idx:]
    y_train = ys[split_idx:]
    return x_train, y_train, x_test, y_test

def calc_accuracy(pred, ground_truth):
    pred = pred.flatten()
    ground_truth = ground_truth.flatten()
    pred = pred  > 0.5
    results = [p == gt for p, gt in zip(pred, ground_truth)]
    return sum(results) / float(len(pred))

# specific examples
def load_churn_data_data(folder = "churn"):
    xs = pd.read_csv(folder+"/features.csv", index_col = 0).values
    ys = pd.read_csv(folder+"/labels.csv", index_col = 0).values
    return xs, ys

def get_titanic_example():
    x, y = load_data("titanic")
    m = ann.Model()
    m.add_layer(x.shape[1], 3, 'relu')
    m.add_layer(3, 1, 'relu')
    return m, x, y

def get_churn_example():
    m = ann.Model()
    m.add_layer(11, 15)
    m.add_layer(15, 5)
    m.add_layer(5, 1)
    x, y = load_churn_data()
    return m, x, y

def get_mnist_example():
    x, y = get_mnist_samples(1000, 2)
    x = x / 50.
    y = np.array([y_[1] for y_ in y])
    m = ann.Model(x[0].shape)
    filters = get_edge_filters()
    m.add(ann.Conv2D(n_filters=len(filters), filter_size=len(filters[0,0,0]),
        a_func='relu'))
    m.layers[-1].filters = filters
    m.add(ann.Flatten())
    m.add(ann.Dense(10, a_func='sigmoid'))
    m.add(ann.Dense(1, a_func='sigmoid'))
    return m, x, y

def get_minimal_conv_example():
    func = lambda x: x > 1
    xs, ys = get_samples_from_func(func, 100, 1, (0, 2))
    xs = np.array([[[x]] for x in xs])
    m = ann.Model(xs[0].shape)
    m.add(ann.Conv2D(n_filters=1, filter_size=1))
    m.add(ann.Flatten(a_func='identity'))
    return m, xs, ys

def calc_mnist_acc(preds, labels):
    correct = [np.argmax(preds[i]) == np.argmax(labels[i]) for i in
            range(len(labels))]
    return sum(correct) / (0.+len(correct))
