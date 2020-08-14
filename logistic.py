import time
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt


class PolynomialLogisticModel:

    def __init__(self, coefficients=[]):
        self.coefficients = coefficients
    
    def logits(self, x):
        value = 0.0
        if len(self.coefficients) > 0:
            value += self.coefficients[0]
        if len(self.coefficients) > 1:
            value += np.sum(self.coefficients[1] * x, axis=1)
        if len(self.coefficients) > 2:
            value += 0.5 * np.sum((x @ self.coefficients[2].T) * x, axis=1)
        if len(self.coefficients) > 3:
            raise NotImplementedError("Degree-%s polynomials not supported"
                                      % len(self.coefficients))
        return value
    
    def accuracy(self, x, y):
        yhat = self.logits(x) >= 0
        return np.mean(yhat == y)
    
    def save_as(self, path):
        state = dict(("a%s" % k, c) for k, c in enumerate(self.coefficients))
        np.savez(path, **state)
    
    def load_from(self, path):
        with np.load(path) as archive:
            self.coefficients = [archive[k] for k in sorted(archive.files)]
    

class LogisticRegressionSolver:

    def __init__(self, x, y, regweight=0.0, degree=2):
        self.x = x
        self.y = y
        numx, self.dim = self.x.shape
        self.num_points, = numy, = self.y.shape
        assert numx == numy
        assert y.dtype == bool
        self.degree = degree
        assert self.degree in [0, 1, 2]
        self.regweight = regweight
        assert self.regweight >= 0.0
        self.num_params = sum(self.dim**k for k in range(degree + 1))

    def unpack_parameter_vector(self, theta):
        assert theta.shape == (self.num_params,)
        coefficients = []
        for k in range(self.degree + 1):
            flat, theta = theta[:self.dim ** k], theta[self.dim ** k:]
            coeff = np.reshape(flat, k * [self.dim])
            coefficients.append(coeff)
        assert theta.shape == (0,)  # now depleted
        return coefficients
    
    def build_model(self, theta):
        coefficients = self.unpack_parameter_vector(theta)
        return PolynomialLogisticModel(coefficients)
    
    def compute_logits(self, coefficients, x=None):
        a0, a1, a2 = coefficients
        v0 = a0
        v1 = np.sum(a1 * self.x, axis=1)
        v2 = np.sum((self.x @ a2.T) * self.x, axis=1)
        return v0 + v1 + v2
    
    def compute_loss(self, theta):
        model = self.build_model(theta)
        logps = model.logits(self.x)
        overpositiveness = np.sum(logps[~self.y], axis=0)
        overnegativeness = np.sum(-logps[self.y], axis=0)
        loss = (overpositiveness + overnegativeness) / self.num_points
        reg = self.regweight * np.sum(theta ** 2)
        return loss + reg

    def solve(self, verbose=True):
        start = time.time()
        theta = np.random.normal(size=self.num_params)
        solution = minimize(self.compute_loss, theta, method="cg")
        dur = time.time() - start
        if verbose and solution.success:
            print("Found solution in %.3f seconds." % dur)
        elif verbose and not solution.success:
            print("Giving up after %.3f seconds." % dur)
        return solution


def _test_saving_and_loading_of_quadratic_logistic_models():

    coefficients = [
        np.random.normal(),
        np.random.normal(size=(5,)),
        np.random.normal(size=(5, 5)),
    ]

    model = PolynomialLogisticModel(coefficients)

    for a, b in zip(coefficients, model.coefficients):
        assert np.all(a == b)

    model.save_as("/tmp/params.npz")
    model.load_from("/tmp/params.npz")

    for a, b in zip(coefficients, model.coefficients):
        assert np.all(a == b)


def _demo_logistic_regression_in_synthetic_case():

    coefficients = [
        np.random.normal(),
        np.random.normal(size=(2,)),
        np.random.normal(size=(2, 2)),
    ]

    true_model = PolynomialLogisticModel(coefficients)

    x = np.random.uniform(-3, +3, size=(1000, 2))
    y = true_model.logits(x) >= 0

    solver = LogisticRegressionSolver(x, y, regweight=1e-3, degree=2)
    solution = solver.solve()

    estimated_model = solver.build_model(solution.x)
    yhat = estimated_model.logits(x) >= 0
    accuracy = estimated_model.accuracy(x, y)

    figure, (left, right) = plt.subplots(figsize=(12, 6), ncols=2)
    true_colors = ["g" if yt else "r" for yt in y]
    left.scatter(*x.T, color=true_colors)
    left.set_title("num pos: %s; num neg: %s" % (sum(y), sum(~y)))
    fake_colors = ["g" if yt else "r" for yt in yhat]
    right.scatter(*x.T, color=fake_colors)
    right.set_title("Accuracy: %.1f pct" % (100. * accuracy))
    plt.show()

    # print(estimated_model.a0, a0)
    # print(estimated_model.a1, a1)
    # print(estimated_model.a2, a2)


if __name__ == "__main__":

    _test_saving_and_loading_of_quadratic_logistic_models()
    _demo_logistic_regression_in_synthetic_case()
    
    # x = np.random.normal(size=(100, 2))  # vectorial input
    # y = 1.5 + x[:, 0] > x[:, 1]**2   # boolean output

    # solver = LogisticRegressionSolver(x, y, regweight=1e-2)
    # solution = solver.solve()

    # print(solution)
