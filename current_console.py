from ann import *
x, y = utils.get_mnist_samples(100)
m = Model(x[0].shape)
m.add(Conv2D())
m.add(MaxPooling())
m.add(Flatten())
m.add(Dense(15))
m.add(Dense(10, a_func='sigmoid'))
