import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt



class ANN:

    def __init__(self, indim, outdim, hidden=[32, 32, 32], afun=tf.tanh):
        self.INPUT = tf.placeholder(shape=[None, indim], dtype=tf.float64)
        self.layers = [self.INPUT]
        for dim in hidden:
            X = self.layers[-1]
            Y = tf.layers.dense(X, units=dim, activation=afun)
            self.layers.append(Y)
        self.OUTPUT = tf.layers.dense(self.layers[-1], units=outdim)
        self.layers.append(self.OUTPUT)


class Trainer:

    def __init__(self, network, session):
        self.network = network
        self.session = session
        self.X = self.network.INPUT
        self.YHAT = self.network.OUTPUT
        self.Y = tf.placeholder(shape=self.YHAT.shape, dtype=self.YHAT.dtype)
        self.LOSS = tf.reduce_mean((self.Y - self.YHAT) ** 2)
        self.LRATE = tf.placeholder(shape=[], dtype=tf.float64)
        self.optimizer = tf.train.GradientDescentOptimizer(self.LRATE)
        self.WALK = self.optimizer.minimize(self.LOSS)

    def step(self, x, y, lrate=0.01):
        total = len(x)
        idx = np.random.choice(range(total), size=total // 3)
        feed = {self.X: x[idx,], self.Y: y[idx,], self.LRATE: lrate}
        self.session.run(self.WALK, feed)
    


if __name__ == "__main__":

    network = ANN(1, 1)
    session = tf.Session()
    trainer = Trainer(network, session)

    trainer.session.run(tf.global_variables_initializer())

    num_points = 1000
    x = np.linspace(-1, +1, num_points).reshape((num_points, 1))
    y = np.reshape(np.cumsum(np.random.normal(size=num_points)), [-1, 1])

    yhat = trainer.session.run(trainer.YHAT, {trainer.X: x})

    plt.ion()
    plt.plot(x.flatten(), y.flatten(), ".", alpha=0.5)
    hatplot, = plt.plot(x.flatten(), yhat.flatten(), "-", lw=3, alpha=0.5)
    plt.pause(.001)

    for iteration in range(1000):
        lrate = 0.01 / (1 + 0.001*iteration)
        trainer.step(x, y, lrate)
        yhat = trainer.session.run(trainer.YHAT, {trainer.X: x})
        hatplot.set_data(x.flatten(), yhat.flatten())
        plt.pause(.001)
