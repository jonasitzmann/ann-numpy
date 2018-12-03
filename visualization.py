import matplotlib.pyplot as plt
import matplotlib.animation as ani
from ann import *
plt.style.use('seaborn')

# Set up formatting for the movie files
Writer = ani.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
def animate(i, model, xs, ys, train=True):
    print('epoch:',i)
    if(train):
        model.fit(xs, ys, 1, True)
    contourf(model, xs, ys)
    scatter(xs, ys)
def animate_no_gt(i, model, xs, ys, train=True):
    if(train):
        model.fit(xs, ys, 1)
    y_pred = model.pred_vec(xs)
    scatter(xs, y_pred)


def contourf(model, xs, ys):
    step_size = 0.3
    x_min, x_max = xs[:, 0].min() - 1, xs[:, 0].max() + 1
    y_min, y_max = xs[:, 1].min() - 1, xs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size),
                     np.arange(y_min, y_max, step_size))
    Z = model.pred_vec(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='plasma')


def plot_predictions(model, xs, ys):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    animate(0, model, xs, ys, False)
    plt.show()

def plot_training_no_ground_truth(model, xs, ys):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    anim = ani.FuncAnimation(fig, animate_no_gt, fargs = (model, xs, ys,),
            repeat=False)
    plt.show()


def scatter(xs, ys):
    pos_xs = [xs[i] for i,y in enumerate(ys) if y>0]
    neg_xs = [xs[i] for i,y in enumerate(ys) if y<=0]
    if pos_xs:
        plt.scatter(*list(zip(*pos_xs)), c='y')
    if neg_xs:
        plt.scatter(*list(zip(*neg_xs)), c='b')

    

def plot_training(model, xs, ys, n_epochs = 100, save=False):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    anim = ani.FuncAnimation(fig, animate, frames=n_epochs, fargs = (model, xs,
        ys, ),repeat=False,)
    if save:
        anim.save('training_animation.mp4', writer = writer)
    else:
        plt.show()



