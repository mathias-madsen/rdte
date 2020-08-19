import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt



class MultiLayerPerceptron:
    """
    A feed-forward artificial neural network with hyperbolic-tangent
    activation functions and linear readout, implemented in Tensorflow.
    """

    def __init__(self, indim, outdim, hidden=[32, 32, 32]):
        self.INPUT = tf.placeholder(shape=[None, indim], dtype=tf.float64)
        self.layers = [self.INPUT]
        self.functions = []
        for depth, dim in enumerate(hidden):
            name = "dense_layer_%s" % (depth + 1)
            function = tf.layers.Dense(dim, tf.tanh, name=name)
            OUTPUT = function(self.layers[-1])
            self.layers.append(OUTPUT)
            self.functions.append(function)
        function = tf.layers.Dense(outdim, tf.tanh, name="linear_readout")
        self.OUTPUT = function(self.layers[-1])
        self.layers.append(self.OUTPUT)
    
    def evaluate_parameters_in(self, session):
        return session.run([f.get_weights() for f in self.functions])


class NumericMultiLayerPerceptron:
    """
    An MLP implemented in numpy, for fast forward passes,
    inspection, and easier parameter saving and loading.
    """

    def __init__(self, params):
        self.weights = [w for w, b in params]
        self.biases = [b for w, b in params]
        self.indim = self.weights[0].shape[0]
        self.outdim = self.weights[-1].shape[1]
    
    def forward(self, x):
        assert np.ndim(x) in [1, 2]
        if np.ndim(x) == 1:
            x = np.expand_dims(x, 0)
        assert x.shape[1] == self.indim
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            x = np.tanh(b + x @ W)
        return self.biases[-1] + x @ self.weights[-1]
    
    def save_as(self, filepath):
        kwargs = {}
        for i in range(len(self.weights)):
            kwargs["weights_%s" % i] = self.weights[i]
            kwargs["biases_%s" % i] = self.biases[i]
        np.savez(filepath, **kwargs)
    
    def load_from(self, filepath):
        with np.load(filepath) as source:
            wkeys = [k for k in source.keys() if k.startswith("weight")]
            bkeys = [k for k in source.keys() if k.startswith("bias")]
            self.weights = [source[k] for k in sorted(wkeys)]
            self.biases = [source[k] for k in sorted(bkeys)]


class MeanSquaredErrorMinimizer:
    """
    A training class for taking gradient steps that reduce the mean
    squared distance between function outputs and target values.
    """

    def __init__(self, network, session=None):
        self.network = network
        self.session = session if session is not None else tf.Session()
        self.X = self.network.INPUT
        self.YHAT = self.network.OUTPUT
        self.Y = tf.placeholder(shape=self.YHAT.shape, dtype=self.YHAT.dtype)
        self.LOSS = tf.reduce_mean((self.Y - self.YHAT) ** 2)
        self.LRATE = tf.placeholder(shape=[], dtype=tf.float64)
        self.optimizer = tf.train.GradientDescentOptimizer(self.LRATE)
        self.WALK = self.optimizer.minimize(self.LOSS)

    def step(self, x, y, lrate=0.01, bsize=None):
        total = len(x)
        bsize = bsize if bsize is not None else total // 3
        idx = np.random.choice(range(total), size=bsize, replace=False)
        feed = {self.X: x[idx,], self.Y: y[idx,], self.LRATE: lrate}
        self.session.run(self.WALK, feed)
    


if __name__ == "__main__":

    network = MultiLayerPerceptron(1, 1)
    trainer = MeanSquaredErrorMinimizer(network)

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
