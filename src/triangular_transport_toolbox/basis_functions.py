"""
Basis function classes for triangular transport maps.

This module provides a hierarchy of callable basis function classes that replace
the string-based function generation approach. Each class implements both
evaluation and derivative computation.

Classes
-------
BasisFunction : ABC
    Abstract base class for all basis functions.
ConstantBasis : BasisFunction
    Constant term (always returns 1).
PolynomialBasis : BasisFunction
    Polynomial basis in a single variable.
HermiteFunctionBasis : BasisFunction
    Polynomial multiplied by Gaussian envelope.
LeftEdgeTerm : BasisFunction
    Smooth left edge term for bounded support.
RightEdgeTerm : BasisFunction
    Smooth right edge term for bounded support.
RadialBasisFunction : BasisFunction
    Gaussian radial basis function.
IntegratedRBF : BasisFunction
    Integrated (CDF) radial basis function.
ProductBasis : BasisFunction
    Product of multiple basis functions.
ComponentFunction : callable
    Collection of basis functions forming a map component.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np
import scipy.special

if TYPE_CHECKING:
    from .transport_map import transport_map


# =============================================================================
# Abstract Base Class
# =============================================================================


class BasisFunction(ABC):
    """
    Abstract base class for all basis functions.

    Subclasses must implement `evaluate` and `derivative` methods.
    The `__call__` method delegates to `evaluate`.
    """

    @abstractmethod
    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """
        Evaluate the basis function at points x.

        Parameters
        ----------
        x : ndarray
            Input samples with shape (N, D) or (..., D) for batch processing,
            where D is dimensionality.
        tm : transport_map
            The transport map instance (for accessing parameters like
            special term locations).

        Returns
        -------
        ndarray
            Basis function values with shape matching x.shape[:-1].
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """
        Evaluate the derivative of the basis function with respect to x_k.

        Parameters
        ----------
        x : ndarray
            Input samples, shape (N, D) or (..., D) for batch processing.
        tm : transport_map
            The transport map instance.
        k : int
            Index of the variable to differentiate with respect to.

        Returns
        -------
        ndarray
            Derivative values with shape matching x.shape[:-1].
        """
        pass

    def __call__(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate the basis function (delegates to `evaluate`)."""
        return self.evaluate(x, tm)


# =============================================================================
# Constant Basis
# =============================================================================


@dataclass
class ConstantBasis(BasisFunction):
    """
    Constant basis function that always returns 1.

    Used for the constant/intercept term in map components.
    """

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Return ones with shape (N,)."""
        return np.ones(x.shape[:-1])

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """Derivative of a constant is zero."""
        return np.zeros(x.shape[:-1])


# =============================================================================
# Polynomial Basis
# =============================================================================


@dataclass
class PolynomialBasis(BasisFunction):
    """
    Polynomial basis function in a single variable.

    Parameters
    ----------
    dimension : int
        The variable index (column of x) this polynomial operates on.
    order : int
        The polynomial order.
    polyfunc : callable
        The numpy polynomial class to use (e.g., np.polynomial.hermite_e.HermiteE).
    polyfunc_der : callable
        The derivative function for the polynomial coefficients.
    coefficients : ndarray, optional
        Pre-computed polynomial coefficients. If None, uses standard
        basis coefficients [0, 0, ..., 0, 1].
    """

    dimension: int
    order: int
    polyfunc: Callable
    polyfunc_der: Callable
    coefficients: np.ndarray = None

    def __post_init__(self):
        """Set up the polynomial coefficients and objects."""
        if self.coefficients is None:
            # Standard basis: coefficient 1 for the highest order term
            self.coefficients = np.array([0.0] * self.order + [1.0])

        # Pre-compute the polynomial and its derivative
        self._poly = self.polyfunc(self.coefficients)
        self._der_coeffs = self.polyfunc_der(self.coefficients)
        self._der_poly = self.polyfunc(self._der_coeffs)

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate the polynomial at x[:, dimension]."""
        return self._poly(x[..., self.dimension])

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """
        Derivative with respect to x_k.

        Returns derivative polynomial if k == dimension, else zeros.
        """
        if k != self.dimension:
            return np.zeros(x.shape[:-1])
        return self._der_poly(x[..., self.dimension])


# =============================================================================
# Hermite Function Basis
# =============================================================================


@dataclass
class HermiteFunctionBasis(BasisFunction):
    """
    Hermite function: polynomial multiplied by Gaussian envelope exp(-x^2/4).

    This creates a basis function that decays to zero far from the origin,
    which is useful for stable numerical behavior.

    Parameters
    ----------
    dimension : int
        The variable index this function operates on.
    order : int
        The polynomial order.
    polyfunc : callable
        The numpy polynomial class to use.
    polyfunc_der : callable
        The derivative function for polynomial coefficients.
    """

    dimension: int
    order: int
    polyfunc: Callable
    polyfunc_der: Callable

    def __post_init__(self):
        """Set up normalized coefficients."""
        # Standard basis coefficients
        base_coefficients = np.array([0.0] * self.order + [1.0])

        # Normalize to have maximum absolute value of 1
        # Evaluate over a wide range to find the maximum
        hf_x = np.linspace(-100, 100, 100001)
        hf_eval = self.polyfunc(base_coefficients)(hf_x) * np.exp(-(hf_x**2) / 4)
        normalization = 1.0 / np.max(np.abs(hf_eval))

        self.coefficients = base_coefficients.copy()
        self.coefficients[-1] = normalization

        # Pre-compute polynomials
        self._poly = self.polyfunc(self.coefficients)
        self._der_coeffs = self.polyfunc_der(self.coefficients)
        self._der_poly = self.polyfunc(self._der_coeffs)

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate polynomial(x) * exp(-x^2/4)."""
        xi = x[..., self.dimension]
        return self._poly(xi) * np.exp(-(xi**2) / 4)

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """
        Derivative: d/dx[f(x) * exp(-x^2/4)] = (f'(x) - x*f(x)/2) * exp(-x^2/4).
        """
        if k != self.dimension:
            return np.zeros(x.shape[:-1])

        xi = x[..., self.dimension]
        poly_val = self._poly(xi)
        der_poly_val = self._der_poly(xi)
        gaussian = np.exp(-(xi**2) / 4)

        # Product rule: d/dx[f*g] = f'*g + f*g'
        # where g = exp(-x^2/4), so g' = -x/2 * g
        return (der_poly_val - xi * poly_val / 2) * gaussian


# =============================================================================
# Special Terms (Edge Terms and RBFs)
# =============================================================================


@dataclass
class SpecialTermBasis(BasisFunction):
    """
    Base class for special terms (LET, RET, RBF, iRBF).

    Special terms have location (mu) and scale parameters that are
    determined dynamically from the transport map's special_terms dictionary.

    Parameters
    ----------
    dimension : int
        The variable index this term operates on.
    component_index : int
        The map component index (k) this term belongs to.
    term_index : int
        The index of this special term among terms of the same type
        in this component.
    is_cross_term : bool
        Whether this is a cross-term (dimension != component_index).
    """

    dimension: int
    component_index: int
    term_index: int
    is_cross_term: bool = False

    def _get_mu_and_scale(self, tm: transport_map) -> tuple[float, float]:
        """Retrieve the center (mu) and scale from transport map."""
        st_dict = tm.special_terms[self.component_index]

        if self.is_cross_term:
            st_dict = st_dict["cross-terms"]

        dim_dict = st_dict[self.dimension]
        mu = dim_dict["centers"][self.term_index]
        scale = dim_dict["scales"][self.term_index]

        return mu, scale


@dataclass
class LeftEdgeTerm(SpecialTermBasis):
    """
    Left edge term for smooth behavior at the left boundary.

    The function smoothly transitions from linear (slope 1) on the left
    to zero slope on the right of the center point.

    Formula: ((x - μ)(1 - erf((x - μ)/(√2 σ))) - σ√(2/π) exp(-((x-μ)/(√2 σ))²)) / 2
    """

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate the left edge term."""
        mu, scale = self._get_mu_and_scale(tm)
        xi = x[..., self.dimension]

        z = (xi - mu) / (np.sqrt(2) * scale)
        erf_z = scipy.special.erf(z)

        return (
            (xi - mu) * (1 - erf_z) - scale * np.sqrt(2 / np.pi) * np.exp(-(z**2))
        ) / 2

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """Derivative: (1 - erf((x - μ)/(√2 σ))) / 2."""
        if k != self.dimension:
            return np.zeros(x.shape[:-1])

        mu, scale = self._get_mu_and_scale(tm)
        xi = x[..., self.dimension]

        z = (xi - mu) / (np.sqrt(2) * scale)
        return (1 - scipy.special.erf(z)) / 2


@dataclass
class RightEdgeTerm(SpecialTermBasis):
    """
    Right edge term for smooth behavior at the right boundary.

    The function smoothly transitions from zero slope on the left
    to linear (slope 1) on the right of the center point.

    Formula: ((x - μ)(1 + erf((x - μ)/(√2 σ))) + σ√(2/π) exp(-((x-μ)/(√2 σ))²)) / 2
    """

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate the right edge term."""
        mu, scale = self._get_mu_and_scale(tm)
        xi = x[..., self.dimension]

        z = (xi - mu) / (np.sqrt(2) * scale)
        erf_z = scipy.special.erf(z)

        return (
            (xi - mu) * (1 + erf_z) + scale * np.sqrt(2 / np.pi) * np.exp(-(z**2))
        ) / 2

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """Derivative: (1 + erf((x - μ)/(√2 σ))) / 2."""
        if k != self.dimension:
            return np.zeros(x.shape[:-1])

        mu, scale = self._get_mu_and_scale(tm)
        xi = x[..., self.dimension]

        z = (xi - mu) / (np.sqrt(2) * scale)
        return (1 + scipy.special.erf(z)) / 2


@dataclass
class RadialBasisFunction(SpecialTermBasis):
    """
    Gaussian radial basis function.

    Formula: exp(-((x - μ)/σ)²/2) / (σ √(2π))
    """

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate the RBF."""
        mu, scale = self._get_mu_and_scale(tm)
        xi = x[..., self.dimension]

        z = (xi - mu) / scale
        return np.exp(-(z**2) / 2) / (scale * np.sqrt(2 * np.pi))

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """Derivative: -(x - μ) / (√(2π) σ³) exp(-((x-μ)/σ)²/2)."""
        if k != self.dimension:
            return np.zeros(x.shape[:-1])

        mu, scale = self._get_mu_and_scale(tm)
        xi = x[..., self.dimension]

        z = (xi - mu) / scale
        return -(xi - mu) / (np.sqrt(2 * np.pi) * scale**3) * np.exp(-(z**2) / 2)


@dataclass
class IntegratedRBF(SpecialTermBasis):
    """
    Integrated radial basis function (CDF of Gaussian).

    Formula: (1 + erf((x - μ)/(√2 σ))) / 2
    """

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate the integrated RBF."""
        mu, scale = self._get_mu_and_scale(tm)
        xi = x[..., self.dimension]

        z = (xi - mu) / (np.sqrt(2) * scale)
        return (1 + scipy.special.erf(z)) / 2

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """Derivative: 1/(√(2π) σ) exp(-(x - μ)²/(2σ²))."""
        if k != self.dimension:
            return np.zeros(x.shape[:-1])

        mu, scale = self._get_mu_and_scale(tm)
        xi = x[..., self.dimension]

        return (
            1
            / (np.sqrt(2 * np.pi) * scale)
            * np.exp(-((xi - mu) ** 2) / (2 * scale**2))
        )


# =============================================================================
# Product Basis (for multivariate terms)
# =============================================================================


@dataclass
class ProductBasis(BasisFunction):
    """
    Product of multiple basis functions.

    Used for multivariate polynomial terms like x_0 * x_1.

    Parameters
    ----------
    factors : list[BasisFunction]
        The basis functions to multiply together.
    """

    factors: list[BasisFunction] = field(default_factory=list)

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate the product of all factors."""
        result = np.ones(x.shape[:-1])
        for factor in self.factors:
            result = result * factor.evaluate(x, tm)
        return result

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """
        Derivative using the product rule.

        d/dx_k [f1 * f2 * ... * fn] = sum_i (f1 * ... * f'_i * ... * fn)
        """
        if len(self.factors) == 0:
            return np.zeros(x.shape[:-1])

        if len(self.factors) == 1:
            return self.factors[0].derivative(x, tm, k)

        # Evaluate all factors and their derivatives
        values = [f.evaluate(x, tm) for f in self.factors]
        derivatives = [f.derivative(x, tm, k) for f in self.factors]

        # Apply product rule: sum over i of (product of all except i) * derivative_i
        result = np.zeros(x.shape[:-1])
        for i in range(len(self.factors)):
            term = derivatives[i]
            for j in range(len(self.factors)):
                if j != i:
                    term = term * values[j]
            result = result + term

        return result


# =============================================================================
# Linearized Basis (wrapper for tail linearization)
# =============================================================================


@dataclass
class LinearizedBasis(BasisFunction):
    """
    Wrapper that applies linearization in the tails.

    For values outside the linearization bounds, the function is
    linearly extrapolated using the value and derivative at the boundary.

    Parameters
    ----------
    inner : BasisFunction
        The basis function to wrap.
    dimension : int
        The dimension to linearize.
    increment : float
        Small increment for finite difference at boundary.
    """

    inner: BasisFunction
    dimension: int
    increment: float = 1e-6

    def evaluate(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """Evaluate with linearization in tails."""
        if tm.linearization is None:
            return self.inner.evaluate(x, tm)

        x_work = copy.copy(x)
        thresholds = tm.linearization_threshold

        # Find points outside bounds
        below = x_work[..., self.dimension] < thresholds[self.dimension, 0]
        above = x_work[..., self.dimension] > thresholds[self.dimension, 1]

        # Compute displacement from boundary
        vec = np.zeros(x.shape[:-1])
        vec[below] = x_work[below, self.dimension] - thresholds[self.dimension, 0]
        vec[above] = x_work[above, self.dimension] - thresholds[self.dimension, 1]

        # Truncate x to boundaries
        x_trc = copy.copy(x_work)
        x_trc[below, self.dimension] = thresholds[self.dimension, 0]
        x_trc[above, self.dimension] = thresholds[self.dimension, 1]

        # Evaluate at truncated points
        val_trc = self.inner.evaluate(x_trc, tm)

        # Create extrapolated points for derivative estimation
        x_ext = copy.copy(x_trc)
        x_ext[below, self.dimension] += self.increment
        x_ext[above, self.dimension] -= self.increment

        val_ext = self.inner.evaluate(x_ext, tm)

        # Linear extrapolation: f(x) ≈ f(x_trc) + f'(x_trc) * (x - x_trc)
        # Using finite difference for f': (f(x_ext) - f(x_trc)) / increment
        slope = (val_ext - val_trc) / self.increment

        result = val_trc + slope * vec

        return result

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """Derivative with linearization."""
        if tm.linearization is None:
            return self.inner.derivative(x, tm, k)

        if k != self.dimension:
            # For other dimensions, just truncate x and evaluate
            x_work = copy.copy(x)
            thresholds = tm.linearization_threshold

            below = x_work[..., self.dimension] < thresholds[self.dimension, 0]
            above = x_work[..., self.dimension] > thresholds[self.dimension, 1]

            x_work[below, self.dimension] = thresholds[self.dimension, 0]
            x_work[above, self.dimension] = thresholds[self.dimension, 1]

            return self.inner.derivative(x_work, tm, k)

        # For the linearized dimension, derivative is constant in tails
        x_work = copy.copy(x)
        thresholds = tm.linearization_threshold

        below = x_work[..., self.dimension] < thresholds[self.dimension, 0]
        above = x_work[..., self.dimension] > thresholds[self.dimension, 1]

        x_trc = copy.copy(x_work)
        x_trc[below, self.dimension] = thresholds[self.dimension, 0]
        x_trc[above, self.dimension] = thresholds[self.dimension, 1]

        return self.inner.derivative(x_trc, tm, k)


# =============================================================================
# Component Function (collection of basis functions)
# =============================================================================


class ComponentFunction:
    """
    Collection of basis functions forming a map component.

    This is a callable that evaluates all basis functions and returns
    a matrix of shape (N, num_terms).

    Parameters
    ----------
    basis_functions : list[BasisFunction]
        The basis functions in this component.
    """

    def __init__(self, basis_functions: list[BasisFunction]):
        self.basis_functions = basis_functions

    def __call__(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """
        Evaluate all basis functions.

        Returns
        -------
        ndarray
            Matrix of shape (N, num_terms) with basis function evaluations.
        """
        if len(self.basis_functions) == 0:
            return None

        results = [bf.evaluate(x, tm) for bf in self.basis_functions]
        return np.stack(results, axis=-1)

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """
        Evaluate derivatives of all basis functions with respect to x_k.

        Returns
        -------
        ndarray
            Matrix of shape (N, num_terms) with derivative evaluations.
        """
        if len(self.basis_functions) == 0:
            return None

        results = [bf.derivative(x, tm, k) for bf in self.basis_functions]
        return np.stack(results, axis=-1)

    def __len__(self):
        return len(self.basis_functions)


class _NullComponentFunction:
    """
    A component function that always returns None.

    Used for empty nonmonotone components.
    """

    def __call__(self, x: np.ndarray, tm: transport_map) -> None:
        """Return None for empty components."""
        return None

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> None:
        """Return None for empty components."""
        return None

    def __len__(self):
        return 0


class _DerivativeComponentFunction:
    """
    Wrapper that computes derivatives of a component function.

    This is used for separable monotonicity strategies, where we need
    derivative functions for optimization constraints.

    Parameters
    ----------
    component_function : ComponentFunction
        The component function to take derivatives of.
    derivative_dimension : int
        The dimension (k) to take derivatives with respect to.
    """

    def __init__(
        self, component_function: ComponentFunction, derivative_dimension: int
    ):
        self.component_function = component_function
        self.derivative_dimension = derivative_dimension

    def __call__(self, x: np.ndarray, tm: transport_map) -> np.ndarray:
        """
        Evaluate the derivative of the component function.

        Returns
        -------
        ndarray
            Matrix of shape (N, num_terms) with derivative evaluations.
        """
        return self.component_function.derivative(x, tm, self.derivative_dimension)

    def derivative(self, x: np.ndarray, tm: transport_map, k: int) -> np.ndarray:
        """
        Second derivative (not currently implemented).

        This would require second derivative support in basis functions.
        """
        raise NotImplementedError(
            "Second derivatives are not currently supported. "
            "This would require implementing second-order derivative methods "
            "in BasisFunction classes."
        )

    def __len__(self):
        return len(self.component_function)


# =============================================================================
# Term Parser
# =============================================================================


def parse_term(
    term_spec,
    tm: transport_map,
    component_k: int,
    st_counter: dict[int, int],
    apply_linearization: bool = False,
) -> BasisFunction:
    """
    Parse a term specification into a BasisFunction.

    Parameters
    ----------
    term_spec : list or str
        The term specification:
        - [] : Constant term
        - [0] : First-order polynomial in x_0
        - [0, 0] : Second-order polynomial in x_0
        - [0, 1] : Product of first-order polynomials in x_0 and x_1
        - [0, "HF"] : Hermite function in x_0
        - [0, 0, "HF"] : Second-order Hermite function in x_0
        - [0, "LIN"] : Linearized polynomial in x_0
        - "LET 0" : Left edge term in x_0
        - "RET 0" : Right edge term in x_0
        - "RBF 0" : Radial basis function in x_0
        - "iRBF 0" : Integrated RBF in x_0
    tm : transport_map
        The transport map instance.
    component_k : int
        The component index (k value for this map component).
    st_counter : dict[int, int]
        Counter for special terms per dimension (modified in place).
    apply_linearization : bool
        Whether to wrap polynomial terms with linearization.

    Returns
    -------
    BasisFunction
        The parsed basis function.
    """
    # Constant term
    if term_spec == []:
        return ConstantBasis()

    # Special term (string format)
    if isinstance(term_spec, str):
        st_type, dim_str = term_spec.split(" ")
        dimension = int(dim_str)

        # Get the term index for this dimension
        if dimension not in st_counter:
            st_counter[dimension] = 0
        term_index = st_counter[dimension]
        st_counter[dimension] += 1

        # Determine if cross-term
        is_cross_term = dimension != component_k

        kwargs = {
            "dimension": dimension,
            "component_index": component_k,
            "term_index": term_index,
            "is_cross_term": is_cross_term,
        }

        st_type_lower = st_type.lower()
        if st_type_lower == "let":
            return LeftEdgeTerm(**kwargs)
        elif st_type_lower == "ret":
            return RightEdgeTerm(**kwargs)
        elif st_type_lower == "rbf":
            return RadialBasisFunction(**kwargs)
        elif st_type_lower == "irbf":
            return IntegratedRBF(**kwargs)
        else:
            raise ValueError(
                f"Special term '{st_type}' not understood. "
                "Currently, only LET, RET, iRBF, and RBF are implemented."
            )

    # Polynomial term (list format)
    if isinstance(term_spec, list):
        # Check for modifiers
        hermite_function = any(
            item == "HF" for item in term_spec if isinstance(item, str)
        )
        linearize = any(item == "LIN" for item in term_spec if isinstance(item, str))

        # Remove string modifiers
        dimensions = [item for item in term_spec if not isinstance(item, str)]

        if len(dimensions) == 0:
            # Empty after removing modifiers -> constant
            return ConstantBasis()

        # Count occurrences of each dimension to get orders
        unique_dims, counts = np.unique(dimensions, return_counts=True)

        # Build basis functions for each unique dimension
        factors = []
        for dim, order in zip(unique_dims, counts):
            if hermite_function:
                bf = HermiteFunctionBasis(
                    dimension=int(dim),
                    order=int(order),
                    polyfunc=tm.polyfunc,
                    polyfunc_der=tm.polyfunc_der,
                )
            else:
                bf = PolynomialBasis(
                    dimension=int(dim),
                    order=int(order),
                    polyfunc=tm.polyfunc,
                    polyfunc_der=tm.polyfunc_der,
                )

            # Apply linearization if requested
            if linearize and apply_linearization:
                bf = LinearizedBasis(
                    inner=bf,
                    dimension=int(dim),
                    increment=tm.linearization_increment,
                )

            factors.append(bf)

        # Return single factor or product
        if len(factors) == 1:
            return factors[0]
        else:
            return ProductBasis(factors=factors)

    raise ValueError(f"Term specification not understood: {term_spec}")


def build_component_function(
    terms: list,
    tm: transport_map,
    component_k: int,
    apply_linearization: bool = False,
) -> ComponentFunction:
    """
    Build a ComponentFunction from a list of term specifications.

    Parameters
    ----------
    terms : list
        List of term specifications (same format as monotone[k] or nonmonotone[k]).
    tm : transport_map
        The transport map instance.
    component_k : int
        The component index.
    apply_linearization : bool
        Whether to apply linearization to polynomial terms.

    Returns
    -------
    ComponentFunction
        The assembled component function.
    """
    # Counter for special terms per dimension
    st_counter: dict[int, int] = {}

    # Track which terms are special terms (for cross-term handling)
    basis_functions = []
    st_indices = []

    for i, term_spec in enumerate(terms):
        bf = parse_term(
            term_spec=term_spec,
            tm=tm,
            component_k=component_k,
            st_counter=st_counter,
            apply_linearization=apply_linearization,
        )
        basis_functions.append(bf)

        # Track special terms
        if isinstance(term_spec, str):
            st_indices.append(i)

    # Handle cross-term products for multiple special terms in different dimensions
    if "cross-terms" in tm.special_terms.get(component_k, {}):
        import itertools

        # Extract special term basis functions
        st_bases = [basis_functions[i] for i in st_indices]

        # Group by dimension
        dim_to_bases: dict[int, list[BasisFunction]] = {}
        for bf in st_bases:
            if isinstance(bf, SpecialTermBasis):
                dim = bf.dimension
                if dim not in dim_to_bases:
                    dim_to_bases[dim] = []
                dim_to_bases[dim].append(bf)

        if len(dim_to_bases) > 1:
            # Create all combinations
            dim_lists = list(dim_to_bases.values())
            combinations = list(itertools.product(*dim_lists))

            # Create product basis functions
            product_bases = [
                ProductBasis(factors=list(combo)) for combo in combinations
            ]

            # Remove original special terms and add products
            basis_functions = [
                bf for i, bf in enumerate(basis_functions) if i not in st_indices
            ]
            basis_functions.extend(product_bases)

    return ComponentFunction(basis_functions)
