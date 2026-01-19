"""
Monotonicity strategy classes for transport maps.

This module provides two strategies for enforcing monotonicity in transport maps:
1. IntegratedRectifier: Uses numerical integration of a rectified function
2. SeparableMonotonicity: Uses a separable parameterization with derivative constraints
"""

import copy
from abc import ABC, abstractmethod

import numpy as np

from .rectifier import Rectifier


class MonotonicityStrategy(ABC):
    """
    Abstract base class for monotonicity enforcement strategies.

    Subclasses must implement methods for:
    - Evaluating the monotone part of map components
    - Optimizing map component functions
    - Precalculating required matrices
    - Building derivative functions (if needed)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this monotonicity strategy."""
        pass

    @abstractmethod
    def evaluate_monotone_part(self, tm, x, k, coeffs_mon):
        """
        Evaluate the monotone part of the k-th map component function.

        Parameters
        ----------
        tm : TransportMap
            The transport map instance.
        x : ndarray
            Input samples, shape (N, D).
        k : int
            Index of the map component function.
        coeffs_mon : ndarray
            Coefficients for the monotone basis functions.

        Returns
        -------
        ndarray
            The monotone part evaluation, shape (N,).
        """
        pass

    @abstractmethod
    def optimize_component(self, tm, k):
        """
        Optimize the k-th map component function.

        Parameters
        ----------
        tm : TransportMap
            The transport map instance.
        k : int
            Index of the map component function.

        Returns
        -------
        tuple
            (coeffs_nonmon, coeffs_mon) - the optimized coefficients.
        """
        pass

    @abstractmethod
    def precalculate(self, tm, k):
        """
        Perform any precalculations needed for the k-th component.

        Parameters
        ----------
        tm : TransportMap
            The transport map instance.
        k : int
            Index of the map component function.
        """
        pass

    @abstractmethod
    def reset_precalculations(self):
        """
        Clear any precalculated matrices.
        Called before the precalculate loop when samples change.
        """
        pass

    @abstractmethod
    def build_derivative_functions(self, tm):
        """
        Build derivative functions if needed by this strategy.

        Parameters
        ----------
        tm : TransportMap
            The transport map instance.
        """
        pass

    @abstractmethod
    def supports_cross_terms_adaptation(self) -> bool:
        """Return whether this strategy supports cross-terms map adaptation."""
        pass

    @abstractmethod
    def supports_alternate_root_finding(self) -> bool:
        """Return whether this strategy supports alternate root finding."""
        pass


class IntegratedRectifier(MonotonicityStrategy):
    """
    Monotonicity strategy using integrated rectifier functions.

    This approach ensures monotonicity by integrating a rectified (always positive)
    function. The rectifier transforms an unconstrained function into a positive
    function, and integration then yields a monotonically increasing result.

    Parameters
    ----------
    rectifier_type : str, default="exponential"
        Type of rectifier function to use. Options include:
        - "exponential": exp(x)
        - "squared": x^2 (not invertible)
        - "softplus": log(1 + exp(x))
        - "explinearunit": ELU-like function

    delta : float, default=1e-8
        Small value added to the rectifier output to prevent numerical underflow.

    quadrature_input : dict, optional
        Configuration for Gaussian quadrature integration. Keys include:
        - "order" (int): Number of quadrature points (default: 100)
        - "xis" (ndarray): Pre-computed integration points
        - "Ws" (ndarray): Pre-computed integration weights
        - "adaptive" (bool): Whether to use adaptive quadrature
        - "threshold" (float): Convergence threshold for adaptive quadrature
        - "increment" (int): Points to add in adaptive mode
        - "verbose" (bool): Print adaptive quadrature progress

    Attributes
    ----------
    rect : Rectifier
        The rectifier object used to ensure positivity.
    quadrature_input : dict
        Quadrature configuration with pre-computed weights and points.

    Examples
    --------
    >>> from triangular_transport_toolbox import TransportMap, IntegratedRectifier
    >>> strategy = IntegratedRectifier(
    ...     rectifier_type="softplus",
    ...     quadrature_input={"order": 50}
    ... )
    >>> tm = TransportMap(samples, strategy, monotone=monotone,
    ...                    nonmonotone=nonmonotone, monotonicity=strategy)
    """

    def __init__(
        self,
        rectifier_type: str = "exponential",
        delta: float = 1e-8,
        quadrature_input: dict = None,
    ):
        self.rectifier_type = rectifier_type
        self.delta = delta
        self.rect = Rectifier(mode=rectifier_type, delta=delta)

        # Set up quadrature input
        self.quadrature_input = (
            quadrature_input if quadrature_input is not None else {"order": 100}
        )

        # Pre-calculate integration points if not provided
        self._setup_quadrature()

    def _setup_quadrature(self):
        """Pre-calculate Gaussian quadrature weights and points."""
        if "xis" not in self.quadrature_input and "Ws" not in self.quadrature_input:
            if "order" not in self.quadrature_input:
                raise KeyError(
                    "'order' must be specified in quadrature_input when "
                    "'xis' and 'Ws' are not provided."
                )
            order = self.quadrature_input["order"]

            # Get coefficients for the order-th Legendre polynomial
            coefs = [0] * order + [1]
            coefs_der = np.polynomial.legendre.legder(coefs)

            # Define the derivative of the Legendre polynomial
            LegendreDer = np.polynomial.legendre.Legendre(coefs_der)

            # Get integration points (roots of Legendre polynomial)
            xis = np.polynomial.legendre.legroots(coefs)

            # Calculate weights
            Ws = 2.0 / ((1.0 - xis**2) * (LegendreDer(xis) ** 2))

            # Store in quadrature_input
            self.quadrature_input["xis"] = copy.copy(xis)
            self.quadrature_input["Ws"] = copy.copy(Ws)

    @property
    def name(self) -> str:
        return "integrated rectifier"

    def evaluate_monotone_part(self, tm, x, k, coeffs_mon):
        """Evaluate monotone part using Gaussian quadrature integration."""

        def integral_argument(xi, y, coeffs_mon, k):
            # Reconstruct the full X matrix with integration variable
            X_loc = copy.copy(y)
            X_loc[:, tm.skip_dimensions + k] = copy.copy(xi)

            # Evaluate the basis functions
            Psi_mon_loc = tm.fun_mon[k](X_loc, tm)

            # Calculate the rectifier argument
            rect_arg = np.dot(Psi_mon_loc, coeffs_mon[:, np.newaxis])[..., 0]

            # Apply the rectifier
            arg = self.rect.evaluate(rect_arg)

            # Add delta to prevent underflow
            arg += self.delta

            return arg

        # Evaluate the integral using Gaussian quadrature
        monotone_part = tm.GaussQuadrature(
            f=integral_argument,
            a=0,
            b=x[..., tm.skip_dimensions + k],
            args=(x, coeffs_mon, k),
            **self.quadrature_input,
        )

        return monotone_part

    def optimize_component(self, tm, k):
        """Optimize using general BFGS optimization."""
        from scipy.optimize import minimize

        # Assemble the coefficient vector
        coeffs = np.zeros(len(tm.coeffs_nonmon[k]) + len(tm.coeffs_mon[k]))
        div = len(tm.coeffs_nonmon[k])

        coeffs[:div] = copy.copy(tm.coeffs_nonmon[k])
        coeffs[div:] = copy.copy(tm.coeffs_mon[k])

        # Optimize
        opt = minimize(
            method="BFGS",
            fun=tm.objective_function,
            jac=tm.objective_function_jacobian,
            x0=coeffs,
            args=(k, div),
        )

        # Extract optimized coefficients
        coeffs_opt = copy.copy(opt.x)
        coeffs_nonmon = coeffs_opt[:div]
        coeffs_mon = coeffs_opt[div:]

        return (coeffs_nonmon, coeffs_mon)

    def precalculate(self, tm, k):
        """No additional precalculations needed for integrated rectifier."""
        pass

    def reset_precalculations(self):
        """No precalculations to reset for integrated rectifier."""
        pass

    def build_derivative_functions(self, tm):
        """No derivative functions needed for integrated rectifier."""
        pass

    def supports_cross_terms_adaptation(self) -> bool:
        return True

    def supports_alternate_root_finding(self) -> bool:
        return False


class SeparableMonotonicity(MonotonicityStrategy):
    """
    Monotonicity strategy using separable parameterization.

    This approach ensures monotonicity by constraining the coefficients of
    monotonically increasing basis functions to be non-negative. This results
    in faster optimization but requires special basis function structures.

    Parameters
    ----------
    delta : float, default=1e-8
        Small offset added to prevent numerical underflow in the objective.

    alternate_root_finding : bool, default=True
        Whether to use an accelerated Newton-based root finding algorithm
        for map inversion. If False, uses bisection (slower but more robust).

    Attributes
    ----------
    der_fun_mon : list
        Derivative functions for monotone basis functions.
    der_Psi_mon : list
        Pre-evaluated derivative matrices.
    optimization_constraints_lb : list
        Lower bounds for coefficient optimization (zeros for monotonicity).
    optimization_constraints_ub : list
        Upper bounds for coefficient optimization (infinity).

    Examples
    --------
    >>> from triangular_transport_toolbox import TransportMap, SeparableMonotonicity
    >>> strategy = SeparableMonotonicity(alternate_root_finding=True)
    >>> tm = TransportMap(samples, strategy, monotone=monotone,
    ...                    nonmonotone=nonmonotone, monotonicity=strategy)
    """

    def __init__(
        self,
        delta: float = 1e-8,
        alternate_root_finding: bool = True,
    ):
        self.delta = delta
        self.alternate_root_finding = alternate_root_finding

        # These will be populated during map construction
        self.der_fun_mon = []
        self.der_fun_mon_strings = []
        self.der_Psi_mon = []
        self.optimization_constraints_lb = []
        self.optimization_constraints_ub = []

    @property
    def name(self) -> str:
        return "separable monotonicity"

    def evaluate_monotone_part(self, tm, x, k, coeffs_mon):
        """Evaluate monotone part directly (no integration needed)."""
        monotone_part = np.dot(tm.fun_mon[k](x, tm), coeffs_mon[:, np.newaxis])[:, 0]
        return monotone_part

    def optimize_component(self, tm, k):
        """Optimize using constrained L-BFGS-B with analytical Hessian."""
        from scipy.optimize import minimize

        coeffs_nonmon = copy.copy(tm.coeffs_nonmon[k])
        coeffs_mon = copy.copy(tm.coeffs_mon[k])

        # Get ensemble size
        N = tm.X.shape[0]

        # Build the optimization objective
        if tm.regularization is None:
            # Standard formulation using QR decomposition
            Q, R = np.linalg.qr(tm.Psi_nonmon[k], mode="reduced")
            A_sqrt = tm.Psi_mon[k] - np.linalg.multi_dot((Q, Q.T, tm.Psi_mon[k]))
            A = np.dot(A_sqrt.T, A_sqrt) / N

            def fun_mon_objective(coeffs_mon, A, k, all_outputs=True):
                b = self.delta * np.sum(A, axis=-1)
                Ax = np.dot(A, coeffs_mon[:, np.newaxis])

                dS = (
                    np.dot(self.der_Psi_mon[k], coeffs_mon[:, np.newaxis])
                    + np.sum(self.der_Psi_mon[k], axis=-1)[:, np.newaxis] * self.delta
                )

                objective = (
                    np.dot(coeffs_mon[np.newaxis, :], Ax)[0, 0] / 2
                    - np.sum(np.log(dS)) / N
                    + np.inner(coeffs_mon, b)
                )

                if not all_outputs:
                    return objective

                dPsi_dS = self.der_Psi_mon[k] / dS
                grad = Ax[:, 0] - np.sum(dPsi_dS, axis=0) / N + b
                hess = A + np.dot(dPsi_dS.T, dPsi_dS) / N

                return objective, grad, hess

        elif tm.regularization.lower() == "l2":
            # L2 regularization
            A = np.linalg.multi_dot(
                (
                    np.linalg.inv(
                        np.dot(tm.Psi_nonmon[k].T, tm.Psi_nonmon[k])
                        + tm.regularization_lambda
                        * np.identity(tm.Psi_nonmon[k].shape[-1])
                    ),
                    tm.Psi_nonmon[k].T,
                    tm.Psi_mon[k],
                )
            )

            A = np.dot(
                (tm.Psi_mon[k] - np.dot(tm.Psi_nonmon[k], A)).T,
                tm.Psi_mon[k] - np.dot(tm.Psi_nonmon[k], A),
            ) / 2 + tm.regularization_lambda * (
                np.dot(A.T, A) + np.identity(A.shape[-1])
            )

            def fun_mon_objective(coeffs_mon, A, k, all_outputs=True):
                b = self.delta * np.sum(A, axis=-1)
                Ax = np.dot(A, coeffs_mon[:, np.newaxis])

                dS = (
                    np.dot(self.der_Psi_mon[k], coeffs_mon[:, np.newaxis])
                    + np.sum(self.der_Psi_mon[k], axis=-1)[:, np.newaxis] * self.delta
                )

                objective = (
                    np.dot(coeffs_mon[np.newaxis, :], Ax)[0, 0] / 2
                    - np.sum(np.log(dS)) / N
                    + np.inner(coeffs_mon, b)
                )

                if not all_outputs:
                    return objective

                dPsi_dS = self.der_Psi_mon[k] / dS
                grad = Ax[:, 0] - np.sum(dPsi_dS, axis=0) / N + b
                hess = A + np.dot(dPsi_dS.T, dPsi_dS) / N

                return objective, grad, hess

        # Set up bounds
        bounds = [
            [
                self.optimization_constraints_lb[k][idx],
                self.optimization_constraints_ub[k][idx],
            ]
            for idx in range(len(self.optimization_constraints_lb[k]))
        ]

        # Optimize
        opt = minimize(
            fun=fun_mon_objective,
            method="L-BFGS-B",
            x0=coeffs_mon,
            jac=True,
            bounds=bounds,
            args=(A, k),
        )

        coeffs_mon = opt.x

        # Calculate nonmonotone coefficients
        if tm.regularization is None:
            Q, R = np.linalg.qr(tm.Psi_nonmon[k], mode="reduced")
            coeffs_nonmon = -np.linalg.multi_dot(
                (np.linalg.inv(R), Q.T, tm.Psi_mon[k], coeffs_mon[:, np.newaxis])
            )[:, 0]
        elif tm.regularization.lower() == "l2":
            coeffs_nonmon = -np.linalg.multi_dot(
                (
                    np.linalg.inv(
                        np.dot(tm.Psi_nonmon[k].T, tm.Psi_nonmon[k])
                        + 2
                        * tm.regularization_lambda
                        * np.identity(tm.Psi_nonmon[k].shape[-1])
                    ),
                    np.dot(tm.Psi_nonmon[k].T, tm.Psi_mon[k]),
                    coeffs_mon[:, np.newaxis],
                )
            )[:, 0]

        return (coeffs_nonmon, coeffs_mon)

    def precalculate(self, tm, k):
        """Precalculate derivative matrices for this component."""
        self.der_Psi_mon.append(copy.copy(self.der_fun_mon[k](copy.copy(tm.X), tm)))

    def reset_precalculations(self):
        """Clear precalculated matrices (called before precalculate loop)."""
        self.der_Psi_mon = []

    def build_derivative_functions(self, tm):
        """Build derivative functions for all map components."""
        # Reset the lists
        self.der_fun_mon = []
        self.der_fun_mon_strings = []
        self.der_Psi_mon = []
        self.optimization_constraints_lb = []
        self.optimization_constraints_ub = []

        # Delegate to TransportMap which has all the complex logic
        tm.build_derivative_functions_for_separable()

    def supports_cross_terms_adaptation(self) -> bool:
        return False

    def supports_alternate_root_finding(self) -> bool:
        return True
