import time
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt


RED = np.array([1., 0, 0])
GREEN = np.array([0, .5, 0])


def compute_logits(theta, x):
    """ Compute linear function, with theta as packed bias + slope. """
    return theta[..., 0] + np.sum(x * theta[..., 1:], axis=-1)


def differentiate_logits_wrt_theta(theta, x):
    """ Compute the gradient of the logits with respect to `theta`. """
    shape = list(np.shape(x))
    shape[-1] += 1
    grad = np.ones(shape, dtype=x.dtype)
    grad[..., 1:] = x
    return grad


def sigmoid(z, maxexp=200):
    """ Compute the inverse of `scipy.special.logit`. """
    halfz = np.clip(z / 2, -abs(maxexp), +abs(maxexp))
    return np.exp(halfz) / (np.exp(halfz) + np.exp(-halfz))


def compute_probs(theta, x):
    """ Compute sigmoid of log-odds estimates. """
    logits = compute_logits(theta, x)
    clipped = np.clip(logits, -300, +300)
    return 1 / (1 + np.exp(-clipped))


def compute_logistic_loss(theta, x, y):
    """ Compute mean of signed log-odds estimates. """
    logits = compute_logits(theta, x)
    logits[y] *= -1
    return np.mean(logits, axis=0)


def compute_accuracy(theta, x, y):
    """ Compute proportion of correctly categorized data points. """
    logits = compute_logits(theta, x)
    yhats = logits >= 0
    return np.mean(yhats == y)


# class PolynomialLogisticModel:

#     def __init__(self, bias=0.0, weights=0.0):
#         self.weights = weights
#         self.bias = bias
    
#     def logits(self, x):
#         """ Estimate the log-odds in favor of positive decisions. """
#         return self.bias + np.sum(x * self.weights, axis=1)
    
#     def probs(self, x):
#         """ Estimate the probability of a positive decision given x. """
#         return sigmoid(self.logits(x))
    
#     def gradient(self, x):
#         """ Compute the gradient of `logits` wrt the parameters. """
#         bias_grad = np.ones_like(x[:, 0])
#         weights_grad = x
#         return [bias_grad, weights_grad]
    
#     def predict(self, x):
#         """ Compute a boolean prediction for each input vector. """
#         return self.logits(x) >= 0

#     def accuracy(self, x, y):
#         """ Compute the number of correct boolean predictions. """
#         return np.mean(self.predict(x) == y)
    
#     def save_as(self, path):
#         """ Save the model parameters as an `.npz` file. """
#         np.savez(path, bias=self.bias, weights=self.weights)
    
#     def load_from(self, path):
#         """ Load some model parameters from an `.npz` file. """
#         with np.load(path) as archive:
#             self.weights = archive["weights"]
#             self.bias = archive["bias"]
    

class LogisticRegressionSolver:

    def __init__(self, x, y, regweight=0.0):
        self.x = x
        self.y = y
        numx, self.dim = self.x.shape
        self.num_points, = numy, = self.y.shape
        assert numx == numy
        assert y.dtype == bool
        self.regweight = regweight
        assert self.regweight >= 0.0
        self.num_params = 1 + self.dim

    def jac(self, theta):
        dlogodds = differentiate_logits_wrt_theta(theta, self.x)
        dlogodds[self.y] *= -1
        meangrad = np.mean(dlogodds, axis=0)
        reggrad = 2.0 * theta
        reggrad *= self.regweight / self.num_points
        assert meangrad.shape == reggrad.shape == (self.num_params,)
        return meangrad + reggrad
    
    def hess(self, theta):
        unweighted = 2.0 * np.eye(self.num_params)
        return self.regweight / self.num_points * unweighted

    def hessp(self, theta, vector):
        return 2.0 * self.regweight / self.num_points * vector

    def compute_loss(self, theta):
        logistic_loss = compute_logistic_loss(theta, self.x, self.y)
        regularizer = self.regweight / self.num_points * np.sum(theta ** 2)
        assert np.shape(logistic_loss) == np.shape(regularizer) == ()
        return logistic_loss + regularizer
    
    def callback(self, theta):
        print(self.compute_loss(theta))

    def solve(self, verbose=True):
        start = time.time()
        theta = 1e-5 * np.random.normal(size=self.num_params)
        # solution = minimize(self.compute_loss, theta)
        solution = minimize(self.compute_loss,
                            theta,
                            method="newton-cg",
                            jac=self.jac,
                            hess=self.hess,
                            hessp=self.hessp)
        dur = time.time() - start
        if verbose and solution.success:
            print("\n\n\tFound solution in %.3f seconds.\n\n" % dur)
        elif verbose and not solution.success:
            print("\n\n\tGiving up after %.3f seconds.\n" % dur)
        return solution


def _demo_logistic_regression_in_ill_posed_synthetic_case():

    x = np.random.uniform(-5, +5, size=(2000, 2))
    xbig = np.concatenate([x ** 0, x ** 1, x ** 2], axis=1)
    wbig = np.random.normal(size=xbig.shape[1])
    y = np.sum(wbig * xbig, axis=1) > 0

    solver = LogisticRegressionSolver(x, y, regweight=10.0)
    solution = solver.solve()

    print(solution)

    logits = compute_logits(solution.x, x)
    probs = compute_probs(solution.x, x)
    yhats = logits >= 0
    accuracy = np.mean(yhats == y)

    _, (left, right) = plt.subplots(figsize=(12, 6), ncols=2)
    true_colors = ["g" if yt else "r" for yt in y]
    left.scatter(*x.T, color=true_colors)
    left.set_title("num pos: %s; num neg: %s" % (sum(y), sum(~y)))
    fake_colors = probs[:, None]*GREEN + (1 - probs[:, None])*RED
    right.scatter(*x.T, color=fake_colors)
    right.set_title("Accuracy: %.1f pct" % (100. * accuracy))
    plt.show()


def _demo_logistic_regression_in_well_posed_synthetic_case():

    x = np.random.uniform(-5, +5, size=(2000, 2))
    xbig = x
    wbig = np.random.normal(size=xbig.shape[1])
    y = np.sum(wbig * xbig, axis=1) > 0

    solver = LogisticRegressionSolver(x, y, regweight=10.0)
    solution = solver.solve()

    print(solution)

    logits = compute_logits(solution.x, x)
    probs = compute_probs(solution.x, x)
    yhats = logits >= 0
    accuracy = np.mean(yhats == y)

    _, (left, right) = plt.subplots(figsize=(12, 6), ncols=2)
    true_colors = ["g" if yt else "r" for yt in y]
    left.scatter(*x.T, color=true_colors)
    left.set_title("num pos: %s; num neg: %s" % (sum(y), sum(~y)))
    fake_colors = probs[:, None]*GREEN + (1 - probs[:, None])*RED
    right.scatter(*x.T, color=fake_colors)
    right.set_title("Accuracy: %.1f pct" % (100. * accuracy))
    plt.show()


def _demo_logistic_sensitivity_in_synthetic_case():

    x = np.random.uniform(-5, +5, size=(1000, 1))
    xbig = np.concatenate([x ** 0, x ** 1, x ** 2], axis=1)
    wbig = np.random.normal(size=(x.shape[1] * 3))
    y = np.sum(wbig * xbig, axis=1) > 0

    solver = LogisticRegressionSolver(x, y, regweight=10.0)
    solution = solver.solve()

    print(solution)

    logits = compute_logits(solution.x, x)
    yhats = logits >= 0
    accuracy = np.mean(yhats == y)

    xspan = np.linspace(-6, +6, 1000)[:, None]
    pspan = compute_probs(solution.x, xspan)
    _, axes = plt.subplots(figsize=(12, 6))
    xflat = np.ravel(x)
    yflat = np.zeros_like(xflat) - 0.05
    redgreen = ["g" if yt else "r" for yt in y]
    axes.scatter(xflat, yflat, c=redgreen)
    axes.plot(xspan, pspan, lw=3)
    axes.set_title("Accuracy: %.1f pct" % (100. * accuracy))
    axes.set_ylim(-0.1, 1.1)

    thetas = solution.x + np.random.uniform(-100, +100, size=(10000, 2))
    losses = compute_logistic_loss(thetas, x[:, None], y)
    explosses = np.exp(-losses/np.std(losses))
    explosses /= explosses.max()
    colors = explosses[:, None]*GREEN + (1 - explosses[:, None])*RED
    _, axes = plt.subplots(figsize=(7, 6))
    axes.scatter(*thetas.T, c=colors)
    axes.set_xlabel("bias")
    axes.set_ylabel("weight")

    plt.show()


if __name__ == "__main__":

    _demo_logistic_regression_in_well_posed_synthetic_case()
    _demo_logistic_regression_in_ill_posed_synthetic_case()
    # _demo_logistic_sensitivity_in_synthetic_case()
