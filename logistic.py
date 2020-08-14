import time
import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt


class PolynomialLogisticModel:

    def __init__(self, coefficients=[]):
        self.coefficients = coefficients
    
    def logits(self, x):
        """ Estimate the log-odds in favor of positive decisions. """
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
    
    # def gradient(self, x):
    #     """ Compute the gradient of `logits` wrt the parameters. """
    #     grads = []
    #     if len(self.coefficients) > 0:
    #         grads.append(np.ones_like(x[:, 0]))
    #     if len(self.coefficients) > 1:
    #         grads.append(x)
    #     if len(self.coefficients) > 2:
    #         grads.append(0.5 * x[:, :, None] * x[:, None, :])
    #     if len(self.coefficients) > 3:
    #         raise NotImplementedError("Degree-%s polynomials not supported"
    #                                   % len(self.coefficients))
    #     return grads
    
    def predict(self, x):
        """ Compute a boolean prediction for each input vector. """
        return self.logits(x) >= 0

    def accuracy(self, x, y):
        """ Compute the number of correct boolean predictions. """
        return np.mean(self.predict(x) == y)
    
    def save_as(self, path):
        """ Save the model parameters as an `.npz` file. """
        state = dict(("a%s" % k, c) for k, c in enumerate(self.coefficients))
        np.savez(path, **state)
    
    def load_from(self, path):
        """ Load some model parameters from an `.npz` file. """
        with np.load(path) as archive:
            self.coefficients = [archive[k] for k in sorted(archive.files)]
    
    def pack_parameter_vector(self):
        """ Flatten and concatenate all coefficients. """
        return np.concatenate([np.ravel(c) for c in self.coefficients])
    

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
    
    # def jac(self, theta):
    #     model = self.build_model(theta)
    #     grads = model.gradient(self.x)
    #     flatshape = [self.num_points, -1]
    #     flatgrads = [np.reshape(g, flatshape) for g in grads]
    #     gradarray = np.concatenate(flatgrads, axis=1)
    #     assert gradarray.shape[0] == self.num_points
    #     assert gradarray.shape[1] == self.num_params
    #     # True ==> bad if very negative
    #     # False ==> bad if very positive
    #     gradarray[self.y] *= -1
    #     meangrad = np.sum(gradarray, axis=0)
    #     reggrad = 2.0 * self.regweight * theta
    #     assert meangrad.shape == reggrad.shape == (self.num_params,)
    #     grad = meangrad * reggrad
    #     return grad

    def compute_logits(self, coefficients):
        a0, a1, a2 = coefficients
        v0 = a0
        v1 = np.sum(a1 * self.x, axis=1)
        v2 = np.sum((self.x @ a2.T) * self.x, axis=1)
        return v0 + v1 + v2
    
    def compute_loss(self, theta):
        model = self.build_model(theta)
        logodds = model.logits(self.x)
        logodds[self.y] *= -1  # positive examples are bad when small
        assert logodds.shape == (self.num_points,)
        loss = np.sum(logodds)
        reg = self.regweight * np.sum(theta ** 2)
        return loss + reg
    
    def callback(self, theta):
        print(self.compute_loss(theta))

    def solve(self, verbose=True):
        start = time.time()
        theta = np.random.normal(size=self.num_params)
        solution = minimize(self.compute_loss, theta)
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


def _test_polynomial_approximation_of_logistic_model():

    coefficients = [
        np.random.normal(),
        np.random.normal(size=(2,)),
        np.random.normal(size=(2, 2)),
    ]

    # symmetrize second derivative:
    coefficients[2] = 0.5 * (coefficients[2] + coefficients[2].T)
    model = PolynomialLogisticModel(coefficients)


def _demo_logistic_regression_in_synthetic_case():

    coefficients = [
        np.random.normal(),
        np.random.normal(size=(2,)),
        np.random.normal(size=(2, 2)),
    ]

    # symmetrize second derivative:
    coefficients[2] = 0.5 * (coefficients[2] + coefficients[2].T)

    true_model = PolynomialLogisticModel(coefficients)

    x = np.random.uniform(-5, +5, size=(2000, 2))
    y = true_model.logits(x) >= 0

    solver = LogisticRegressionSolver(x, y, regweight=1e-5, degree=2)
    solution = solver.solve()

    estimated_model = solver.build_model(solution.x)
    yhat = estimated_model.logits(x) >= 0
    accuracy = estimated_model.accuracy(x, y)

    _, (left, right) = plt.subplots(figsize=(12, 6), ncols=2)
    true_colors = ["g" if yt else "r" for yt in y]
    left.scatter(*x.T, color=true_colors)
    left.set_title("num pos: %s; num neg: %s" % (sum(y), sum(~y)))
    fake_colors = ["g" if yt else "r" for yt in yhat]
    right.scatter(*x.T, color=fake_colors)
    right.set_title("Accuracy: %.1f pct" % (100. * accuracy))
    plt.show()


if __name__ == "__main__":

    _test_saving_and_loading_of_quadratic_logistic_models()
    _demo_logistic_regression_in_synthetic_case()
    
    # gamma = -0.001
    # x = np.random.uniform(-3, +3, size=(1, 5))

    # a0 = np.random.normal()
    # a1 = np.random.normal(size=5)
    # a2 = np.random.normal(size=(5, 5))

    # model_1 = PolynomialLogisticModel([a0, a1, a2])

    # g0, g1, g2 = model_1.gradient(x)

    # b0 = a0 + gamma*g0[0]
    # b1 = a1 + gamma*g1[0]
    # b2 = a2 + gamma*g2[0]

    # model_2 = PolynomialLogisticModel([b0, b1, b2])

    # print(model_1.logits(x) + gamma)
    # print(model_2.logits(x))
    # print()
    # print(gamma * np.ones_like(model_2.logits(x)))
    # print(model_2.logits(x) - model_1.logits(x))
    # print()

    # coefficients = [
    #     np.random.normal(),
    #     np.random.normal(size=(2,)),
    #     np.random.normal(size=(2, 2)),
    # ]

    # # symmetrize second derivative:
    # coefficients[2] = 0.5 * (coefficients[2] + coefficients[2].T)

    # true_model = PolynomialLogisticModel(coefficients)

    # x = np.random.uniform(-3, +3, size=(1000, 2))
    # y = x[:, 0] > x[:, 1]

    # solver = LogisticRegressionSolver(x, y, regweight=1e-3, degree=2)
    # solution = solver.solve()

    # estimated_model = solver.build_model(solution.x)
    # yhat = estimated_model.logits(x) >= 0
    # accuracy = estimated_model.accuracy(x, y)

    # theta = np.random.normal(size=solver.num_params)
    # for iteration in range(100):
    #     loss = solver.compute_loss(theta)
    #     grad = solver.jac(theta)
    #     theta -= 0.001 * grad
    #     if (iteration + 1) % 1 == 0:
    #         print("Loss: %.5f" % loss)
