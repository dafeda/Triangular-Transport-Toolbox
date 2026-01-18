"""
Rectifier functions for transport map monotonicity.
"""

import numpy as np


class Rectifier:
    def __init__(self, mode="softplus", delta=1e-8):
        """
        This object specifies what function is used to rectify the monotone
        map component functions if monotonicity = 'integrated rectifier',
        before the rectifier's output is integrated to yield a monotone
        map component in x_k.

        Variables:

            mode - [default = 'softplus']
                [string] : keyword string defining which function is used
                to rectify the map component functions.

            delta - [default = 1E-8]
                [float] : a small offset value to prevent arithmetic under-
                flow in some of the rectifier functions.
        """

        self.mode = mode
        self.delta = delta

    def evaluate(self, X):
        """
        This function evaluates the specified rectifier.

        Variables:

            X
                [array] : an array of function evaluates to be rectified.
        """

        if self.mode == "squared":
            res = X**2

        elif self.mode == "exponential":
            res = np.exp(X)

        elif self.mode == "expneg":
            res = np.exp(-X)

        elif self.mode == "softplus":
            a = np.log(2)
            aX = a * X
            below = aX < 0
            aX[below] = 0
            res = np.log(1 + np.exp(-np.abs(a * X))) + aX

        elif self.mode == "explinearunit":
            res = np.zeros(X.shape)
            res[(X < 0)] = np.exp(X[(X < 0)])
            res[(X >= 0)] = X[(X >= 0)] + 1

        return res

    def inverse(self, X):
        """
        This function evaluates the inverse of the specified rectifier.

        Variables:

            X
                [array] : an array of function evaluates to be rectified.
        """

        if len(np.where(X < 0)[0] > 0):
            raise Exception("Input to inverse rectifier are negative.")

        if self.mode == "squared":
            raise Exception("Squared rectifier is not invertible.")

        elif self.mode == "exponential":
            res = np.log(X)

        elif self.mode == "expneg":
            res = -np.log(X)

        elif self.mode == "softplus":
            a = np.log(2)

            opt1 = np.log(np.exp(a * X) - 1)
            opt2 = X

            opt1idx = opt1 - opt2 >= 0
            opt2idx = opt1 - opt2 < 0

            res = np.zeros(X.shape)
            res[opt1idx] = opt1[opt1idx]
            res[opt2idx] = opt2[opt2idx]

        elif self.mode == "explinearunit":
            res = np.zeros(X.shape)

            below = X < 1
            above = X >= 1

            res[below] = np.log(X[below])
            res[above] = X - 1

        return res

    def evaluate_dx(self, X):
        """
        This function evaluates the derivative of the specified rectifier.

        Variables:

            X
                [array] : an array of function evaluates to be rectified.
        """

        if self.mode == "squared":
            res = 2 * X

        elif self.mode == "exponential":
            res = np.exp(X)

        elif self.mode == "expneg":
            res = -np.exp(-X)

        elif self.mode == "softplus":
            a = np.log(2)
            res = 1 / (1 + np.exp(-a * X))

        elif self.mode == "explinearunit":
            below = X < 0
            above = X >= 0

            res = np.zeros(X.shape)

            res[below] = np.exp(X[below])
            res[above] = 0

        return res

    def evaluate_dfdc(self, f, dfdc):
        """
        This function evaluates terms used in the optimization of the map
        components if monotonicity = 'separable monotonicity'.
        """

        if self.mode == "squared":
            raise Exception("Not implemented yet.")

        elif self.mode == "exponential":
            # https://www.wolframalpha.com/input/?i=derivative+of+exp%28f%28c%29%29+wrt+c

            res = np.exp(f)

            # Combine with dfdc
            res = np.einsum("i,ij->ij", res, dfdc)

        elif self.mode == "expneg":
            # https://www.wolframalpha.com/input/?i=derivative+of+exp%28-f%28c%29%29+wrt+c

            res = -np.exp(-f)

            # Combine with dfdc
            res = np.einsum("i,ij->ij", res, dfdc)

        elif self.mode == "softplus":
            # https://www.wolframalpha.com/input/?i=derivative+of+log%282%5Ef%28c%29%2B1%29%2Flog%282%29+wrt+c

            # Calculate the first part
            a = np.log(2)
            res = 1 / (1 + np.exp(-a * f))

            # Combine with dfdc
            res = np.einsum("i,ij->ij", res, dfdc)

        elif self.mode == "explinearunit":
            raise Exception("Not implemented yet.")

        return res

    def logevaluate(self, X):
        """
        This function evaluates the logarithm of the specified rectifier.

        Variables:

            X
                [array] : an array of function evaluates to be rectified.
        """

        if self.mode == "squared":
            res = np.log(X**2)

        elif self.mode == "exponential":
            # res             = np.log(np.abs(np.exp(X))) # -marked-
            # res             = X
            if self.delta == 0:
                res = X
            else:
                res = np.log(np.exp(X) + self.delta)

        elif self.mode == "expneg":
            res = -X

        elif self.mode == "softplus":
            a = np.log(2)
            aX = a * X
            below = aX < 0
            aX[below] = 0
            res = np.log(1 + np.exp(-np.abs(a * X))) + aX

            res = np.log(res + self.delta)

        elif self.mode == "explinearunit":
            res = np.zeros(X.shape)
            res[(X < 0)] = np.exp(X[(X < 0)])
            res[(X >= 0)] = X[(X >= 0)] + 1

            res = np.log(res)

        return res
