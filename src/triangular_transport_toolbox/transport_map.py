"""
Triangular transport map toolbox v1.0.0
"""

import copy
from typing import Literal

import numpy as np

from .basis_functions import (
    ConstantBasis,
    _DerivativeComponentFunction,
    _NullComponentFunction,
    build_component_function,
)
from .monotonicity import (
    MonotonicityStrategy,
    SeparableMonotonicity,
)


class TransportMap:
    def __init__(
        self,
        X,
        monotonicity: MonotonicityStrategy,
        regularization: Literal["l1", "l2"],
        monotone=None,
        nonmonotone=None,
        polynomial_type="hermite function",
        coeffs_init=0.0,
        verbose=True,
        regularization_lambda=0.1,
        adaptation=False,
        adaptation_map_type="cross-terms",
        adaptation_max_order=10,
        adaptation_skip_dimensions=0,
        adaptation_max_iterations=25,
    ):
        """
        This toolbox contains functions required to construct, optimize, and
        evaluate transporth methods.

        Maximilian Ramgraber, July 2025

        Variables:

            ===================================================================
            General variables
            ===================================================================

            monotone - [default = None]
                [list] : list specifying the structure of the monotone part of
                the transport map component functions. Required if map
                adaptation is not used.

            nonmonotone  - [default = None]
                [list] : list specifying the structure of the nonmonotone part
                of the transport map component functions. Required if map
                adaptation is not used.

            X
                [array] : N-by-D array of the training samples used to optimize
                the transport map, where N is the number of samples and D is
                the number of dimensions

            polynomial_type - [default = 'hermite function']
                [string] : keyword which specifies what kinds of polynomials
                are used for the transport map component functions.

            monotonicity
                [MonotonicityStrategy] : strategy object which specifies how the
                transport map ensures monotonicity in the last dimensions.
                Must be an IntegratedRectifier or SeparableMonotonicity instance.

            coeffs_init - [default = 0.]
                [float] : value used to initialize the coefficients at the
                start of the map optimization.

            verbose - [default = True]
                [boolean] : a True/False flag which determines whether the map
                prints updates or not. Set to 'False' if running on a cluster
                to avoid recording excessive output logs.

            Regularization ----------------------------------------------------

            regularization - [required]
                [string] : keyword which specifies what kind of regularization
                to use. Must be either 'l1' or 'l2'.
                Regularization cannot be disabled.

            regularization_lambda - [default = 0.1]
                [float] : float which specifies the weight for the map coeff-
                icient regularization.

        """
        # ---------------------------------------------------------------------
        # Load in pre-defined variables
        # ---------------------------------------------------------------------

        # Basis function specification for the monotone and nonmonotone parts
        # of the map component functions.
        self.monotone = copy.deepcopy(monotone)
        self.nonmonotone = copy.deepcopy(nonmonotone)

        # Set up the monotonicity strategy
        if not isinstance(monotonicity, MonotonicityStrategy):
            raise TypeError(
                "monotonicity must be a MonotonicityStrategy instance "
                "(IntegratedRectifier or SeparableMonotonicity), "
                f"got {type(monotonicity).__name__}"
            )
        self.monotonicity = monotonicity

        # Initial value for the coefficients
        self.coeffs_init = coeffs_init

        # Should the toolbox print the outputs to the console?
        self.verbose = verbose

        self.regularization = regularization
        self.regularization_lambda = regularization_lambda

        # ---------------------------------------------------------------------
        # Read and assign the polynomial type
        # ---------------------------------------------------------------------

        # What type of polynomials are we using for the specification of the
        # basis functions in the map component functions?
        self.polynomial_type = polynomial_type

        # Determine the derivative and polynomial terms depending on the chosen type
        if (
            polynomial_type.lower() == "standard"
            or polynomial_type.lower() == "polynomial"
            or polynomial_type.lower() == "power series"
        ):
            self.polyfunc = np.polynomial.polynomial.Polynomial
            self.polyfunc_der = np.polynomial.polynomial.polyder
            self.polyfunc_str = "np.polynomial.Polynomial"
        elif (
            polynomial_type.lower() == "hermite"
            or polynomial_type.lower() == "physicist's hermite"
        ):
            self.polyfunc = np.polynomial.hermite.Hermite
            self.polyfunc_der = np.polynomial.hermite.hermder
            self.polyfunc_str = "np.polynomial.Hermite"
        elif (
            polynomial_type.lower() == "hermite_e"
            or polynomial_type.lower() == "probabilist's hermite"
        ):
            self.polyfunc = np.polynomial.hermite_e.HermiteE
            self.polyfunc_der = np.polynomial.hermite_e.hermeder
            self.polyfunc_str = "np.polynomial.HermiteE"
        elif polynomial_type.lower() == "chebyshev":
            self.polyfunc = np.polynomial.chebyshev.Chebyshev
            self.polyfunc_der = np.polynomial.chebyshev.chebder
            self.polyfunc_str = "np.polynomial.Chebyshev"
        elif polynomial_type.lower() == "laguerre":
            self.polyfunc = np.polynomial.laguerre.Laguerre
            self.polyfunc_der = np.polynomial.laguerre.lagder
            self.polyfunc_str = "np.polynomial.Laguerre"
        elif polynomial_type.lower() == "legendre":
            self.polyfunc = np.polynomial.legendre.Legendre
            self.polyfunc_der = np.polynomial.legendre.legder
            self.polyfunc_str = "np.polynomial.Legendre"
        elif polynomial_type.lower() == "hermite function":
            # Unify this polynomial string, so we can use it as a flag
            self.polynomial_type = "hermite function"
            self.polyfunc = np.polynomial.hermite_e.HermiteE
            self.polyfunc_der = np.polynomial.hermite_e.hermeder
            self.polyfunc_str = "np.polynomial.HermiteE"
        else:
            raise Exception(
                "Polynomial type not understood. The variable polynomial_type "
                "should be either 'standard', 'polynomial', 'power series', "
                "'hermite', 'physicist's hermite', 'hermite_e', "
                "'probabilist's hermite', 'chebyshev', 'laguerre', 'legendre', "
                "or 'hermite function'."
            )

        # ---------------------------------------------------------------------
        # Load and prepare the variables
        # ---------------------------------------------------------------------

        # Load and standardize the samples
        self.X = copy.copy(X)
        self.standardize()

        # Do we specify map adaptation parameters?
        self.adaptation = adaptation
        self.adaptation_map_type = adaptation_map_type.lower()
        self.adaptation_max_order = adaptation_max_order
        self.adaptation_skip_dimensions = adaptation_skip_dimensions
        self.adaptation_max_iterations = adaptation_max_iterations

        # If we are not adapting the map
        if not self.adaptation:
            # Map adaptation is not active
            self.D = len(monotone)
            self.skip_dimensions = X.shape[-1] - self.D

        # If we are adapting the map
        elif self.adaptation:
            # Map adaptation is active. Create a linear transport marginal map.

            # Define lower map component blocks
            self.D = X.shape[-1] - self.adaptation_skip_dimensions
            self.skip_dimensions = self.adaptation_skip_dimensions

            # Initiate dummy monotone and nonmonotone variables
            self.monotone = []
            self.nonmonotone = []
            for _k in range(self.D):
                self.monotone.append([[]])
                self.nonmonotone.append([[]])

        # ---------------------------------------------------------------------
        # Construct the monotone and non-monotone functions
        # ---------------------------------------------------------------------

        # The function_constructor yields six variables:
        #   - fun_mon               : list of monotone functions
        #   - fun_mon_strings       : list of monotone function strings
        #   - coeffs_mon            : list of coefficients for monotone function
        #   - fun_nonmon            : list of nonmonotone functions
        #   - fun_nonmon_strings    : list of nonmonotone function strings
        #   - coeffs_nonmon         : list of coefficients for nonmonotone function

        self.function_constructor_alternative()

        # ---------------------------------------------------------------------
        # Precalculate the Psi matrices
        # ---------------------------------------------------------------------

        # The function_constructor yields two variables:
        #   - Psi_mon               : list of monotone basis evaluations
        #   - Psi_nonmon            : list of nonmonotone basis evaluations

        self.precalculate()

        # # Adapt map
        # self.adapt_map()

    def adapt_map(
        self,
        coeffs=None,
        maxorder_mon=10,
        maxorder_nonmon=10,
        threshold_sw=0.1,
        threshold_prec=0.1,
        sequential_updates=False,
        map_finished=None,
    ):
        """
        This function implements the adaptive transport map algorithm. It is
        currently only implemented for cross-term integrated maps, and for
        demonstration purposes, so it might be a bit unstable.

        =======================================================================
        Variables
        =======================================================================

        increment - [default = 1E-6]
            [float] : the increment for the finite difference approximation to
            the objective function's gradient.

        chronicle - [default = False]
            [boolean] : flag which stores the intermediate solutions of the
            adaptive map algorithm if True. Can be useful to visualize how the
            map is constructed. Creates a pickled dictionary file name
            dictionary_adaptation_chronicle.p in the working directory.
        """

        # =====================================================================
        # Separable map adaptation
        # =====================================================================

        if self.adaptation_map_type == "separable":
            import scipy.stats

            # Initiate monotone and nonmonotone terms
            nonmonotone = [[[]] for x in np.arange(self.D)]
            monotone = [[[x]] for x in np.arange(self.D)]

            # =================================================================
            # Start marginal adaptation
            # =================================================================

            # Array with flags for which marginals have been Gaussianized
            Gaussianized = np.zeros(self.D, dtype=bool)

            # Flag to decide when to stop iterating
            iterate = True
            iteration = 0

            # Create a matrix for the map order
            maporders = np.zeros((self.D, self.D), dtype=int)
            np.fill_diagonal(maporders, 1)

            # Create a matrix for the p values of the Shapiro-Wilk test
            pvals_mat = np.zeros((maxorder_mon, self.D))

            while iterate:
                # Increase the iteration counter
                iteration += 1

                # -------------------------------------------------------------
                # Reconstruct the new map type
                # -------------------------------------------------------------

                # Store the monotone and nonmonotone terms
                self.monotone = copy.deepcopy(monotone)
                self.nonmonotone = copy.deepcopy(nonmonotone)

                # Re-write the functions
                self.function_constructor_alternative()
                self.precalculate()

                # Optimize the map
                # print(np.arange(self.D)[~Gaussianized])
                self.optimize()

                # Apply the map
                Z = self.map()

                # Prepare an array for the pval normality test
                pval_normality_test = np.zeros(self.D)

                # Go through all terms
                for k in range(self.D):
                    # Throw in the p value
                    pval_normality_test[k] = scipy.stats.shapiro(Z[:, k]).pvalue

                # Copy that value into
                pvals_mat[iteration - 1, :] = copy.copy(pval_normality_test)

                # for idx in np.where(pval_normality_test < criterion)[0]:
                for idx in np.where(pval_normality_test >= threshold_sw)[0]:
                    index = np.arange(self.D)[idx]
                    Gaussianized[index] = True

                # Increase the map complexity of the non-Gaussian marginals
                for k in np.where(~Gaussianized)[0]:
                    if maporders[k, k + self.skip_dimensions] < maxorder_mon:
                        # Update map complexity storage
                        maporders[k, k + self.skip_dimensions] += 1

                        # Add an integrated iRBF term
                        monotone[k] += ["iRBF " + str(k)]

                if np.sum(Gaussianized) == self.D:
                    iterate = False

                if iteration >= maxorder_mon - 1:
                    iterate = False

            # =================================================================
            # Start off-diagonal adaptation
            # =================================================================

            # Get the standardized precision matrix
            # precmat = np.abs(np.linalg.inv(np.cov(Z.T)))
            covmat = np.abs(np.cov(Z.T))
            diagval = np.sqrt(np.diag(covmat))
            covmat /= diagval[np.newaxis, :]
            covmat /= diagval[:, np.newaxis]

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     glasso  = sklearn.covariance.GraphicalLassoCV(max_iter=50).fit(Z)
            # precmat = np.abs(glasso.get_precision())

            # Get the standardized precision matrix
            precmat = np.abs(np.linalg.inv(np.cov(Z.T)))
            diagval = np.sqrt(np.diag(precmat))
            precmat /= diagval[np.newaxis, :]
            precmat /= diagval[:, np.newaxis]

            # # with warnings.catch_warnings():
            # #     warnings.simplefilter("ignore")
            # #     glasso  = sklearn.covariance.GraphicalLassoCV(max_iter=50).fit(Z)
            # # precmat = np.abs(glasso.get_precision())
            # diagval = np.sqrt(np.diag(precmat))
            # precmat /= diagval[np.newaxis,:]
            # precmat /= diagval[:,np.newaxis]

            # Store precmat
            self.covmat = copy.copy(covmat)
            self.precmat = copy.copy(precmat)

            # Flag to decide when to stop iterating
            iterate = True
            iteration = 0

            # Create an array to decide when to stop
            if map_finished is None:
                map_finished = np.zeros((self.D, self.D), dtype=bool)

            # Store the precision matrix
            precmat_list = [copy.copy(precmat)]

            while iterate:
                # Increase the iteration counter
                iteration += 1

                # Store the monotone and nonmonotone terms
                self.monotone = copy.deepcopy(monotone)
                self.nonmonotone = copy.deepcopy(nonmonotone)

                # Re-write the functions
                self.function_constructor_alternative()
                self.precalculate()

                # Optimize the map
                self.optimize()

                # Apply the map
                Z = self.map()

                # Attempt to evaluate the precision matrix
                try:
                    if iteration == 1:
                        # Get the standardized precision matrix
                        precmat = np.abs(np.linalg.inv(np.cov(Z.T)))
                        # with warnings.catch_warnings():
                        #     warnings.simplefilter("ignore")
                        #     glasso = sklearn.covariance.GraphicalLassoCV(
                        #         max_iter=50
                        #     ).fit(Z)
                        # precmat = np.abs(glasso.get_precision())
                        diagval = np.sqrt(np.diag(precmat))
                        precmat /= diagval[np.newaxis, :]
                        precmat /= diagval[:, np.newaxis]

                    else:
                        # After we found the precision matrix, we proceed with
                        # the correlation matrix for simplicity.
                        precmat = np.corrcoef(Z.T)

                    # Go through all map components
                    for k in range(self.D):
                        # Go through all potential nonmonotone dependencies
                        for j in range(k):
                            # Is there significant correlation
                            # if precmat[k,j] > threshold_prec and
                            #    precmat[k,j] < precmat_prev[k,j] and
                            #    not map_finished[k,j]:
                            if (
                                precmat[k, j] > threshold_prec
                                and not map_finished[k, j]
                            ):
                                # Increase the map complexity by one order
                                maporders[k, j] += 1

                                # Add the corresponding map component
                                if maporders[k, j] == 1:
                                    nonmonotone[k].append([j] * maporders[k, j])
                                else:
                                    nonmonotone[k].append(
                                        [j] * maporders[k, j] + ["HF"]
                                    )

                            else:
                                # Mark this map component term as converged
                                map_finished[k, j] = True

                        # Sort the nonmonotone map components
                        nonmonotone[k].sort()

                    # And append it to the list
                    precmat_list.append(copy.copy(precmat))

                # If anything failed, stop iterating
                except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                    # Stop iterating
                    iterate = False

                # If all lower-triangular entries (excluding diagonal) are
                # marked as converged, stop iterating
                if np.sum(map_finished) >= self.D * (self.D - 1) / 2:
                    iterate = False

                # Raise a warning if the adaptation stopped at the maximum
                # number of iterations, then stop iterating
                if iteration >= maxorder_nonmon:
                    print(
                        "WARNING: Map adaptation stopped at maximum "
                        "number of iterations."
                    )
                    iterate = False

            # Store the monotone and nonmonotone terms
            self.monotone = copy.deepcopy(monotone)
            self.nonmonotone = copy.deepcopy(nonmonotone)

            # Re-write the functions
            self.function_constructor_alternative()
            self.precalculate()

            # Optimize the map
            self.optimize()

            # Store the maporders
            self.maporders = maporders

        # =====================================================================
        # Cross-term map adaptation
        # =====================================================================

        elif self.adaptation_map_type == "cross-terms":
            self.adaptation_cross_terms(*coeffs)

        else:
            raise Exception(
                "Currently, only adaptation_map_type = 'cross-terms' is implemented."
            )

    def check_inputs(self):
        """
        This function runs some preliminary checks on the input provided,
        alerting the user to any possible input errors.
        """

        if (self.monotone is None or self.nonmonotone is None) and not self.adaptation:
            raise ValueError(
                "Map is undefined. You must either specify 'monotone' and "
                + "'nonmonotone', or set 'adaptation' = True."
            )

        if self.adaptation_map_type not in ["cross-terms", "separable", "marginal"]:
            raise ValueError(
                "'adaptation_map_type' not understood. Must be either "
                + "'cross-terms', 'separable', or 'marginal'."
            )

        if (
            self.adaptation_map_type == "cross-terms"
            and not self.monotonicity.supports_cross_terms_adaptation()
        ):
            raise ValueError(
                "It is only possible to use adaptation_map_type = 'cross-terms' with "
                + "IntegratedRectifier monotonicity strategy."
            )

        if (
            self.hermite_function_threshold_mode != "composite"
            and self.hermite_function_threshold_mode != "individual"
        ):
            raise ValueError(
                "The flag hermite_function_threshold_mode must be "
                + "'composite' or 'individual'. Currently, it is defined as "
                + str(self.hermite_function_threshold_mode)
            )

        if not isinstance(self.regularization, str):
            raise TypeError(
                "'regularization' must be a string ('l1' or 'l2'). "
                + f"Currently, it is of type {type(self.regularization).__name__}"
            )

        if isinstance(self.monotonicity, SeparableMonotonicity):
            if self.regularization != "l2":
                raise ValueError(
                    "When using SeparableMonotonicity, "
                    + "'regularization' must be 'l2' (L2 regularization)."
                    + " Currently, it is defined as "
                    + str(self.regularization)
                )
        else:
            if self.regularization not in ["l1", "l2"]:
                raise ValueError(
                    "When using IntegratedRectifier, "
                    + "'regularization' must be 'l1' (L1 regularization) or "
                    + "'l2' (L2 regularization). Currently, it "
                    + "is defined as "
                    + str(self.regularization)
                )

    def reset(self, X):
        """
        This function is used if the transport map has been initiated with a
        different set of samples. It resets the standardization variables and
        the map's coefficients, requiring new optimization.

        Variables:

            X
                [array] : N-by-D array of the training samples used to optimize
                the transport map, where N is the number of samples and D is
                the number of dimensions
        """
        if len(X.shape) != 2:
            raise Exception(
                "X should be a two-dimensional array of shape (N,D), "
                "N = number of samples, D = number of dimensions. "
                f"Current shape of X is {X.shape}"
            )

        self.X = copy.copy(X)

        # Standardize the samples
        self.standardize()

        # Set all parameters to zero
        for k in range(self.D):
            # Reset coefficients to zero
            self.coeffs_mon[k] *= 0
            self.coeffs_nonmon[k] *= 0

            # Provide them with the desired initial values
            self.coeffs_mon[k] += self.coeffs_init
            self.coeffs_nonmon[k] += self.coeffs_init

        # Precalculate the Psi matrices
        self.precalculate()

    def standardize(self):
        """
        This function centers the samples around zero and re-scales them to
        have unit standard deviation. This is important for certain function
        types used in the map component parameterizations, for example Hermite
        functions, which revert to zero farther away from the origin.

        The standardization is applied before any other transport operations,
        and reverted before results are returned. It should only affect
        internal computations.
        """

        # Standardize samples via their mean and marginal standard deviations
        self.X_mean = np.mean(self.X, axis=0)
        self.X_std = np.std(self.X, axis=0)

        # Standardize the samples
        self.X -= self.X_mean
        self.X /= self.X_std

    def precalculate(self):
        """
        This function pre-calculates matrices of basis function evaluations for
        the samples provided. These matrices can be used to optimize the maps
        more quickly.
        """
        # Precalculate locations of any special terms
        self.determine_special_term_locations()

        # Prepare precalculation matrices
        self.Psi_mon = []
        self.Psi_nonmon = []

        # Reset any strategy-specific precalculations
        self.monotonicity.reset_precalculations()

        # Precalculate matrices
        for k in range(self.D):
            # Evaluate the basis functions
            self.Psi_mon.append(copy.copy(self.fun_mon[k](copy.copy(self.X), self)))
            self.Psi_nonmon.append(
                copy.copy(self.fun_nonmon[k](copy.copy(self.X), self))
            )

            # Perform any strategy-specific precalculations
            self.monotonicity.precalculate(self, k)

        return

    def function_constructor_alternative(self, k=None):
        """
        Build the monotone and nonmonotone component functions for the map.

        This method constructs ComponentFunction objects containing BasisFunction
        instances for each map component function.

        Parameters
        ----------
        k : int or None, default=None
            If None, construct functions for all dimensions.
            If an integer, only construct functions for that dimension.
        """
        if k is None:
            # Construct the functions for all dimensions
            partial_construction = False
            Ks = list(range(self.D))

            # Initialize lists for functions and coefficients
            self.fun_mon = []
            self.coeffs_mon = []
            self.fun_nonmon = []
            self.coeffs_nonmon = []

            # Check for any special terms
            self.check_for_special_terms()
            self.determine_special_term_locations()

        elif np.isscalar(k):
            # Construct the functions only for this dimension
            partial_construction = True
            Ks = [k]

        else:
            raise ValueError(
                "'k' for function_constructor_alternative must be either "
                "None or an integer."
            )

        # Go through all dimensions
        for k in Ks:
            # The component_k for special term lookup includes skip_dimensions
            component_k = k + self.skip_dimensions

            # =================================================================
            # Step 1: Build the monotone function
            # =================================================================

            fun_mon_k = build_component_function(
                terms=self.monotone[k],
                tm=self,
                component_k=component_k,
            )

            if not partial_construction:
                self.fun_mon.append(fun_mon_k)
                self.coeffs_mon.append(np.ones(len(fun_mon_k)) * self.coeffs_init)
            else:
                self.fun_mon[k] = fun_mon_k

            # =================================================================
            # Step 2: Build the nonmonotone function
            # =================================================================

            if len(self.nonmonotone[k]) > 0:
                fun_nonmon_k = build_component_function(
                    terms=self.nonmonotone[k],
                    tm=self,
                    component_k=component_k,
                )
            else:
                # Create a function that returns None
                fun_nonmon_k = _NullComponentFunction()

            if not partial_construction:
                self.fun_nonmon.append(fun_nonmon_k)
                self.coeffs_nonmon.append(
                    np.ones(len(self.fun_nonmon[k])) * self.coeffs_init
                )
            else:
                self.fun_nonmon[k] = fun_nonmon_k

        # =================================================================
        # Step 3: Finalize
        # =================================================================

        # Build derivative functions if required by the monotonicity strategy
        self.monotonicity.build_derivative_functions(self)

        return

    def build_derivative_functions_for_separable(self):
        """
        Build derivative functions for separable monotonicity.

        This is called by SeparableMonotonicity.build_derivative_functions().
        Creates derivative component functions and sets up optimization constraints.
        """
        # Verify we have a SeparableMonotonicity strategy
        assert isinstance(self.monotonicity, SeparableMonotonicity), (
            "build_derivative_functions_for_separable should only be called "
            "with SeparableMonotonicity strategy"
        )

        # Initialize lists in the monotonicity strategy
        self.monotonicity.der_fun_mon = []
        self.monotonicity.optimization_constraints_lb = []
        self.monotonicity.optimization_constraints_ub = []

        # Go through all map components
        for k in range(self.D):
            component_k = k + self.skip_dimensions

            # Set up optimization constraints
            # (non-negative coefficients for monotonicity)
            lb = np.zeros(len(self.fun_mon[k]))
            ub = np.ones(len(self.fun_mon[k])) * np.inf

            # Check for constant terms - they have unconstrained coefficients
            for j, basis_func in enumerate(self.fun_mon[k].basis_functions):
                if isinstance(basis_func, ConstantBasis):
                    # Constant term - unconstrained
                    lb[j] = -np.inf
                    ub[j] = +np.inf

            self.monotonicity.optimization_constraints_lb.append(lb)
            self.monotonicity.optimization_constraints_ub.append(ub)

            # Create derivative component function
            der_fun = _DerivativeComponentFunction(
                component_function=self.fun_mon[k],
                derivative_dimension=component_k,
            )
            self.monotonicity.der_fun_mon.append(der_fun)

        return

    def check_for_special_terms(self):
        """
        This function scans through the user-provided map specifications and
        seeks if there are any special terms ('RBF', 'iRBF', 'LET', 'RET')
        among the terms of the map components. If there are, it determines
        how many there are, and informs the rest of the function where these
        special terms should be located.
        """

        # Number of RBFs
        self.special_terms = {}

        # Go through all map components
        for k in range(self.D):
            # Add a key for this term
            self.special_terms[k + self.skip_dimensions] = {}

            # Check all nonmonotone terms of this map component
            for entry in self.nonmonotone[k]:
                # If this term is a string, it denotes a special term
                if isinstance(entry, str):
                    # Split the entry and extract its dimensional entry
                    index = int(entry.split(" ")[1])

                    # If this key does not yet exist, create it
                    if index not in list(
                        self.special_terms[k + self.skip_dimensions].keys()
                    ):
                        self.special_terms[k + self.skip_dimensions][index] = {
                            "counter": 0,
                            "centers": np.asarray([]),
                            "scales": np.asarray([]),
                        }

                    # Mark it in memory
                    self.special_terms[k + self.skip_dimensions][index]["counter"] += 1

            # Check all monotone terms of this map component
            for entry in self.monotone[k]:
                # If this term is a string, it denotes a special term
                if isinstance(entry, str):
                    # Split the entry and extract its dimensional entry
                    index = int(entry.split(" ")[1])

                    if index == k + self.skip_dimensions:
                        # If this key does not yet exist, create it
                        if index not in list(
                            self.special_terms[k + self.skip_dimensions].keys()
                        ):
                            self.special_terms[k + self.skip_dimensions][index] = {
                                "counter": 0,
                                "centers": np.asarray([]),
                                "scales": np.asarray([]),
                            }

                        # Mark it in memory
                        self.special_terms[k + self.skip_dimensions][index][
                            "counter"
                        ] += 1

                    # Does this monotone term have cross-terms?
                    elif (
                        index != k + self.skip_dimensions
                    ):  # The proposed monotone ST index is not the last argument
                        # Does the dictionary have a monotone key yet?
                        if "cross-terms" not in list(
                            self.special_terms[k + self.skip_dimensions].keys()
                        ):
                            # Create the key, if not
                            self.special_terms[k + self.skip_dimensions][
                                "cross-terms"
                            ] = {}

                        # Does this cross-term dependence have a key yet?
                        if index not in list(
                            self.special_terms[k + self.skip_dimensions][
                                "cross-terms"
                            ].keys()
                        ):
                            # Create the key, if not
                            self.special_terms[k + self.skip_dimensions]["cross-terms"][
                                index
                            ] = {
                                "counter": 0,
                                "centers": np.asarray([]),
                                "scales": np.asarray([]),
                            }

                        # Mark it in memory
                        self.special_terms[k + self.skip_dimensions]["cross-terms"][
                            index
                        ]["counter"] += 1

        return

    def determine_special_term_locations(self, k=None):
        """
        This function calculates the location and scale parameters for special
        terms in the transport map definition, specifically RBF (Radial Basis
        Functions), iRBF (Integrated Radial Basis Functions), and LET/RET (Edge
        Terms).

        Position and scale parameters are assigned in the order they have been
        defined, so make sure to define a left edge term first if you want it
        to be on the left side.

        Variables:

            k - [default = None]
                [integer or None] : an integer specifying what dimension of the
                samples the 'term' corresponds to. Used to clarify with respect
                to what dimension we build this basis function
        """

        def place_special_terms(self, dictionary):
            """
            A supporting function, which actually determines where the special
            terms are being placed.
            """

            # Find the key list
            keylist = list(dictionary.keys())

            # If there is a cross-term key, ignore it.
            if "cross-terms" in keylist:
                keylist.remove("cross-terms")

            # Go through all arguments with special terms
            for d in keylist:
                # -------------------------------------------------------------
                # One special term
                # -------------------------------------------------------------

                # We have only one ST
                if dictionary[d]["counter"] == 1:
                    # Determine the center
                    dictionary[d]["centers"] = np.asarray(
                        [np.quantile(self.X[:, d], q=0.5)]
                    )

                    # Determine the scales
                    dictionary[d]["scales"] = np.asarray([0.5])

                # ---------------------------------------------------------
                # Multiple special terms
                # ---------------------------------------------------------

                elif dictionary[d]["counter"] > 1:
                    # Decide where to place the special terms
                    quantiles = np.arange(1, dictionary[d]["counter"] + 1, 1) / (
                        dictionary[d]["counter"] + 1
                    )

                    # Append an empty array, then fill it
                    scales = np.zeros(dictionary[d]["counter"])

                    # Determine the centers
                    dictionary[d]["centers"] = copy.copy(
                        np.quantile(a=self.X[:, d], q=quantiles)
                    )

                    # Determine the scales based on relative differences
                    for i in range(dictionary[d]["counter"]):
                        # Left edge-case: base is half distance to next basis
                        if i == 0:
                            scales[i] = (
                                dictionary[d]["centers"][1]
                                - dictionary[d]["centers"][0]
                            )

                        # Right edge-case: base is half distance to previous basis
                        elif i == dictionary[d]["counter"] - 1:
                            scales[i] = (
                                dictionary[d]["centers"][i]
                                - dictionary[d]["centers"][i - 1]
                            )

                        # Otherwise: base is average distance to neighbours
                        else:
                            scales[i] = (
                                dictionary[d]["centers"][i + 1]
                                - dictionary[d]["centers"][i - 1]
                            ) / 2

                    # Copy the scales into the array
                    dictionary[d]["scales"] = copy.copy(scales)

            return dictionary

        # ---------------------------------------------------------------------
        # Find the special term locations
        # ---------------------------------------------------------------------

        # For what terms shall we apply this update?
        if k is None:
            # No k is supplied, go through all components
            K = np.arange(self.D) + self.skip_dimensions

        else:
            # A k is supplied, only apply the operation to this component
            K = [k + self.skip_dimensions]

        # Go through all terms
        for k in K:
            # If there are cross-terms, do the same thing
            if "cross-terms" in self.special_terms[k]:
                # Write in the special term locations
                self.special_terms[k]["cross-terms"] = place_special_terms(
                    self, dictionary=copy.deepcopy(self.special_terms[k]["cross-terms"])
                )

            # Write in the special term locations
            self.special_terms[k] = place_special_terms(
                self, dictionary=copy.deepcopy(self.special_terms[k])
            )

        return

    def map(self, X=None):
        """
        This function maps the samples X from the target distribution to the
        standard multivariate Gaussian reference distribution. If X has not
        been provided, the samples in storage will be used instead

        Variables:

            X - [default = None]
                [None or array] : N-by-D array of the training samples used to
                optimize the transport map, where N is the number of samples
                and D is the number of dimensions.
        """
        # If X is provided, standardize it; otherwise use stored samples
        if X is not None:
            # Create a local copy of X
            X = copy.copy(X)

            # Standardize the samples
            X -= self.X_mean
            X /= self.X_std

        else:
            # Retrieve X from memory, create a local copy
            X = copy.copy(self.X)

        # Initialize the output array
        Z = np.zeros((X.shape[0], self.D))

        # Evaluate each of the map component functions
        for k in range(self.D):
            # Apply the forward map
            Z[:, k] = copy.copy(
                self.s(
                    x=X,
                    k=k,
                    coeffs_nonmon=self.coeffs_nonmon[k],
                    coeffs_mon=self.coeffs_mon[k],
                )
            )

        return Z

    def s(self, x, k, coeffs_nonmon=None, coeffs_mon=None):
        """
        This function evaluates the k-th map component.

        Variables:

            x
                [array] : N-by-D array of the training samples used to optimize
                the transport map, where N is the number of samples and D is
                the number of dimensions. Can be None, at which point it is
                replaced with X from storage.

            k
                [integer] : an integer variable defining what map component
                is being evaluated. Corresponds to a dimension of sample space.

            coeffs_nonmon - [default = None]
                [vector] : a vector specifying the coefficients of the non-
                monotone part of the map component's terms, i.e., those entries
                which do not depend on x_k. This vector is replaced from
                storage if it is not overwritten.

            coeffs_nonmon - [default = None]
                [vector] : a vector specifying the coefficients of the monotone
                part of the map component's terms, i.e., those entries which do
                not depend on x_k. This vector is replaced from storage if it
                is not overwritten.
        """
        # Load in values if required
        if x is None:
            # If x has not been specified, load it from memory
            x = copy.copy(self.X)

            # Also load the matrix of nonmonotone basis function evaluations
            Psi_nonmon = copy.copy(self.Psi_nonmon[k])

        else:
            # Evaluate the matrix of nonmonotone basis functions
            Psi_nonmon = copy.copy(self.fun_nonmon[k](x, self))

        # If coefficients have not been specified, load them from storage
        if coeffs_mon is None:
            coeffs_mon = self.coeffs_mon[k]

        # If coefficients have not been specified, load them from storage
        if coeffs_nonmon is None:
            coeffs_nonmon = self.coeffs_nonmon[k]

        # ---------------------------------------------------------------------
        # Calculate the non-monotone part
        # ---------------------------------------------------------------------

        # If there are nonmonotone basis functions
        if Psi_nonmon is not None:
            # Multiply them with their corresponding coefficients
            nonmonotone_part = np.dot(Psi_nonmon, coeffs_nonmon[:, np.newaxis])[..., 0]

        # Else, the nonmonotone part is zero
        else:
            nonmonotone_part = 0

        # ---------------------------------------------------------------------
        # Calculate the monotone part
        # ---------------------------------------------------------------------

        # Delegate to the monotonicity strategy
        monotone_part = self.monotonicity.evaluate_monotone_part(self, x, k, coeffs_mon)

        # ---------------------------------------------------------------------
        # Combine both terms
        # ---------------------------------------------------------------------

        # Combine the terms
        result = copy.copy(nonmonotone_part + monotone_part)

        return result

    def evaluate_pushforward_density(self, Z, log_target_pdf, X_star=None):
        """
        This function evaluates the pushforward density, that is to say the
        map's approximation to the standard Gaussian reference. Currently only
        implemented for SeparableMonotonicity.

        Variables:

            Z
                [array] : N-by-D or N-by-(D-E) array of reference distribution
                samples to be mapped to the target distribution, where N is the
                number of samples, D is the number of target distribution
                dimensions, and E the number of pre-specified dimensions (if
                X_precalc is specified).

            log_target_pdf
                [function] : a function which takes as input an N-by-D array of
                samples X and returns the logarithm of the probability density
                of the target pdf for these samples. If the triangular map is
                only partially defined (i.e., self.skip_dimensions != 0), this
                function should return the conditional log density instead.

            X_star - [default = None]
                [None or array] : N-by-E array of samples in the space of the
                target distribution, used to condition the lower D-E dimensions
                during the inversion process.
        """
        # Make sure the user uses separable monotonicity
        assert isinstance(self.monotonicity, SeparableMonotonicity), (
            "evaluate_pushforward_density is currently only implemented "
            "for SeparableMonotonicity."
        )

        # Compute the pre-image points
        X = self.inverse_map(Z, X_star)

        # Apply the change-of-variables formula
        # See https://arxiv.org/abs/2503.21673, Eq. 4

        # First, evaluate the target densities
        log_target_densities = log_target_pdf(X)

        # If this is a conditional map, re-create the full target vector
        if X_star is not None:
            X = np.column_stack((X_star, X))

        # Now, let us find the determinant of the forward map's Jacobian
        log_determinant = 0
        for k in range(self.D):
            # Extract the monotone coefficients
            coeffs_mon = copy.copy(self.coeffs_mon[k])

            # Evaluate the derivative of the monotone part wrt x_k of the k-th
            # map component function.
            der_Psi_mon = copy.copy(
                self.monotonicity.der_fun_mon[k](copy.copy(X), self)
            )

            # Compute the derivatives of this map component function wrt the
            # last variable x_k
            dSkdxk = np.dot(der_Psi_mon, coeffs_mon[:, np.newaxis])[:, 0]

            # Account for the gradient from the standard scaler
            # If X_std is 2, the non-standardized function got compressed
            # We have to correct for that.
            dSkdxk /= self.X_std[k + self.skip_dimensions]

            # Add the logarithm of this derivative
            log_determinant += np.log(dSkdxk)

        # Return the result from the change-of-variables pdf
        return np.exp(log_target_densities - log_determinant)

    def evaluate_pullback_density(self, X, X_star=None):
        """
        This function evaluates the pullback density, that is to say the
        map's approximation to the target density. Currently only implemented
        for SeparableMonotonicity.

        Variables:

            X
                [None or array] : N-by-D array of positions in the target
                distribution space at which the pullback will be evaluated,
                where N is the number of samples and D is the number of
                dimensions.
        """
        import scipy.stats

        # Make sure the user uses separable monotonicity
        assert isinstance(self.monotonicity, SeparableMonotonicity), (
            "evaluate_pullback_density is currently only implemented "
            "for SeparableMonotonicity."
        )

        # If this is a conditional map, re-create the full target vector
        if X_star is not None:
            X = np.column_stack((X_star, X))

        # Compute the image points
        Z = self.map(X)

        # Apply the change-of-variables formula
        # See https://arxiv.org/abs/2503.21673, Eq. 4

        # First, evaluate the target densities
        log_reference_densities = scipy.stats.multivariate_normal.logpdf(
            Z, mean=np.zeros(self.D), cov=np.identity(self.D)
        )

        # Now, let us find the determinant of the forward map's Jacobian
        log_determinant = 0
        for k in range(self.D):
            # Extract the monotone coefficients
            coeffs_mon = copy.copy(self.coeffs_mon[k])

            # Evaluate the derivative of the monotone part wrt x_k of the k-th
            # map component function.
            der_Psi_mon = copy.copy(
                self.monotonicity.der_fun_mon[k](copy.copy(X), self)
            )

            # Compute the derivatives of this map component function wrt the
            # last variable x_k
            dSkdxk = np.dot(der_Psi_mon, coeffs_mon[:, np.newaxis])[:, 0]

            # Account for the gradient from the standard scaler
            # If X_std is 2, the non-standardized function got compressed
            # We have to correct for that.
            dSkdxk /= self.X_std[k]

            # Add the logarithm of this derivative
            log_determinant += np.log(dSkdxk)

        # Return the result from the change-of-variables pdf
        return np.exp(log_reference_densities + log_determinant)

    def optimize(self, K=None):
        """
        This function optimizes the map's component functions, seeking the
        coefficients which best map the samples to a standard multivariate
        Gaussian distribution.

        Variables:

            K - [default = None]
                [None or list] : a list of integers specifying which map
                component functions we are optimizing. If None, the function
                optimizes all map component functions.

        """
        # If we haven't specified which components should be optimized, then
        # we optimize all components
        if K is None:
            K = np.arange(self.D)

        # Delegate to the monotonicity strategy for optimization
        for k in K:
            # Optimize this map component using the strategy
            results = self.monotonicity.optimize_component(self, k)

            # Print optimization progress
            if self.verbose:
                string = "\r" + "Progress: |"
                string += (k + 1) * "█"
                string += (len(K) - k - 1) * " "
                string += "|"
                print(string, end="\r")

            # Extract and store the optimized coefficients
            self.coeffs_nonmon[k] = copy.deepcopy(results[0])
            self.coeffs_mon[k] = copy.deepcopy(results[1])

        return

    def worker_task(self, k):
        """
        This function provides the optimization task for the k-th map component
        function. This specific function only becomes active if
        monotonicity = 'integrated rectifier'.

        Variables:

            k
                [integer] : an integer variable defining what map component
                is being evaluated. Corresponds to a dimension of sample space.
        """
        from scipy.optimize import minimize

        # Assemble the theta vector we are optimizing
        coeffs = np.zeros(len(self.coeffs_nonmon[k]) + len(self.coeffs_mon[k]))
        div = len(self.coeffs_nonmon[k])  # Divisor for the vector

        # Write in the coefficients
        coeffs[:div] = copy.copy(self.coeffs_nonmon[k])
        coeffs[div:] = copy.copy(self.coeffs_mon[k])

        # ---------------------------------------------------------------------
        # Call the optimization routine
        # ---------------------------------------------------------------------

        # Minimize the objective function
        opt = minimize(
            method="BFGS",  #'L-BFGS-B',
            fun=self.objective_function,
            jac=self.objective_function_jacobian,
            x0=coeffs,
            args=(k, div),
        )

        # ---------------------------------------------------------------------
        # Post-process the optimization results
        # ---------------------------------------------------------------------

        # Retrieve the optimized coefficients
        coeffs_opt = copy.copy(opt.x)

        # Separate them into coefficients for monotone and nonmonotone parts
        coeffs_nonmon = coeffs_opt[:div]
        coeffs_mon = coeffs_opt[div:]

        # Return both optimized coefficients
        return (coeffs_nonmon, coeffs_mon)

    def objective_function(self, coeffs, k, div=0):
        """
        This function evaluates the objective function used in the optimization
        of the map's component functions.

        Variables:

            coeffs
                [vector] : a vector containing the coefficients for both the
                nonmonotone and monotone terms of the k-th map component
                function. Is replaced for storage is specified as None.

            k
                [integer] : an integer variable defining what map component
                is being evaluated. Corresponds to a dimension of sample space.

            div - [default = 0]
                [integer] : an integer specifying where the cutoff between the
                nonmonotone and monotone coefficients in 'coeffs' is.
        """
        # Partition the coefficient vector, if necessary
        if coeffs is not None:
            # Separate the vector into nonmonotone and monotone coefficients
            coeffs_nonmon = copy.copy(coeffs[:div])
            coeffs_mon = copy.copy(coeffs[div:])
        else:
            if self.verbose:
                print("loading")
            # Otherwise, load them from object
            coeffs_nonmon = copy.copy(self.coeffs_nonmon[k])
            coeffs_mon = copy.copy(self.coeffs_mon[k])

        # ---------------------------------------------------------------------
        # First part: How close is the ensemble mapped to zero?
        # ---------------------------------------------------------------------

        # Map the samples to the reference marginal
        map_result = self.s(
            x=None, k=k, coeffs_nonmon=coeffs_nonmon, coeffs_mon=coeffs_mon
        )

        # Check how close these samples are to the origin
        objective = 1 / 2 * map_result**2

        # print(objective)

        # ---------------------------------------------------------------------
        # Second part: How much is the ensemble inflated?
        # ---------------------------------------------------------------------

        Psi_mon = self.fun_mon[k](self.X, self)

        # Determine the gradients of the polynomial functions
        monotone_part_der = np.dot(Psi_mon, coeffs_mon[:, np.newaxis])[..., 0]

        # Evaluate the logarithm of the rectified monotone part
        obj = self.monotonicity.rect.logevaluate(monotone_part_der)

        # Subtract this from the objective
        objective -= obj

        # ---------------------------------------------------------------------
        # Average the objective function
        # ---------------------------------------------------------------------

        # Now summarize the contributions and take their average
        objective = np.mean(objective)

        # ---------------------------------------------------------------------
        # Add regularization, if desired
        # ---------------------------------------------------------------------

        if self.regularization is not None:
            # A scalar regularization was specified
            if isinstance(self.regularization, str):
                if self.regularization.lower() == "l1":
                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):
                        # Add l1 regularization for all coefficients
                        objective += self.regularization_lambda * np.sum(
                            np.abs(coeffs_mon)
                        )
                        objective += self.regularization_lambda * np.sum(
                            np.abs(coeffs_nonmon)
                        )

                    elif isinstance(self.regularization_lambda, list):
                        # Add l1 regularization for all coefficients
                        objective += np.sum(
                            self.regularization_lambda[k][div:] * np.abs(coeffs_mon)
                        )
                        objective += np.sum(
                            self.regularization_lambda[k][:div] * np.abs(coeffs_nonmon)
                        )

                    else:
                        raise ValueError(
                            "Data type of regularization_lambda not understood. "
                            "Must be either scalar or list."
                        )

                elif self.regularization == "l2":
                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):
                        # Add l2 regularization for all coefficients
                        objective += self.regularization_lambda * np.sum(coeffs_mon**2)
                        objective += self.regularization_lambda * np.sum(
                            coeffs_nonmon**2
                        )

                    elif isinstance(self.regularization_lambda, list):
                        # Add l1 regularization for all coefficients
                        objective += np.sum(
                            self.regularization_lambda[k][div:] * coeffs_mon**2
                        )
                        objective += np.sum(
                            self.regularization_lambda[k][:div] * coeffs_nonmon**2
                        )

                    else:
                        raise ValueError(
                            "Data type of regularization_lambda not understood. "
                            "Must be either scalar or list."
                        )

                else:
                    raise ValueError("regularization must be either 'l1' or 'l2'.")

            else:
                raise ValueError(
                    "The variable 'regularization' must be either None, 'l1', or 'l2'."
                )

        return objective

    def objective_function_jacobian(self, coeffs, k, div=0):
        """
        This function evaluates the derivative of the objective function used
        in the optimization of the map's component functions.

        Variables:

            coeffs
                [vector] : a vector containing the coefficients for both the
                nonmonotone and monotone terms of the k-th map component
                function. Is replaced for storage is specified as None.

            k
                [integer] : an integer variable defining what map component
                is being evaluated. Corresponds to a dimension of sample space.

            div - [default = 0]
                [integer] : an integer specifying where the cutoff between the
                nonmonotone and monotone coefficients in 'coeffs' is.
        """
        # Partition the coefficient vector, if necessary
        if coeffs is not None:
            # Separate the vector into nonmonotone and monotone coefficients
            coeffs_nonmon = copy.copy(coeffs[:div])
            coeffs_mon = copy.copy(coeffs[div:])
        else:
            # Otherwise, load them from object
            coeffs_nonmon = copy.copy(self.coeffs_nonmon[k])
            coeffs_mon = copy.copy(self.coeffs_mon[k])

        # =====================================================================
        # Prepare term 1
        # =====================================================================

        # First, handle the scalar
        term_1_scalar = self.s(
            x=None, k=k, coeffs_nonmon=coeffs_nonmon, coeffs_mon=coeffs_mon
        )

        # Define the integration argument
        def integral_argument_term1_jac(x, coeffs_mon, k):
            # First reconstruct the full X matrix
            X_loc = copy.copy(self.X)
            X_loc[:, self.skip_dimensions + k] = copy.copy(x)

            # Calculate the local basis function matrix
            Psi_mon_loc = self.fun_mon[k](X_loc, self)

            # Determine the gradients
            rec_arg = np.dot(Psi_mon_loc, coeffs_mon[:, np.newaxis])[..., 0]

            objective = self.monotonicity.rect.evaluate_dfdc(
                f=rec_arg, dfdc=Psi_mon_loc
            )

            return objective

        # Add the integration
        term_1_vector_monotone = self.GaussQuadrature(
            f=integral_argument_term1_jac,
            a=0,
            b=self.X[:, self.skip_dimensions + k],
            args=(coeffs_mon, k),
            **self.monotonicity.quadrature_input,
        )

        # If we have non-monotone terms, consider them
        if self.Psi_nonmon[k] is not None:
            # Evaluate the non-monotone vector term
            term_1_vector_nonmonotone = copy.copy(self.Psi_nonmon[k])

            # Stack the results together
            term_1_vector = np.column_stack(
                (term_1_vector_nonmonotone, term_1_vector_monotone)
            )

        else:
            # If we have no non-monotone terms, the vector is only composed of
            # monotone coefficients
            term_1_vector = term_1_vector_monotone

        # Combine to obtain the full term 1
        term_1 = np.einsum("i,ij->ij", term_1_scalar, term_1_vector)

        # =====================================================================
        # Prepare term 2
        # =====================================================================

        # Create term_2
        # https://www.wolframalpha.com/input/?i=derivative+of+log%28f%28c%29%29+wrt+c

        rec_arg = np.dot(self.Psi_mon[k], coeffs_mon[:, np.newaxis])[
            ..., 0
        ]  # This is dfdk

        numer = self.monotonicity.rect.evaluate_dfdc(f=rec_arg, dfdc=self.Psi_mon[k])

        denom = 1 / (self.monotonicity.rect.evaluate(rec_arg) + self.monotonicity.delta)

        term_2 = np.einsum("ij,i->ij", numer, denom)

        if div > 0:
            # If we have non-monotone terms, expand the term accordingly
            term_2 = np.column_stack((np.zeros((term_2.shape[0], div)), term_2))

        # =====================================================================
        # Combine both terms
        # =====================================================================

        objective = np.mean(term_1 - term_2, axis=0)

        # ---------------------------------------------------------------------
        # Add regularization, if desired
        # ---------------------------------------------------------------------

        if self.regularization is not None:
            # A scalar regularization was specified
            if isinstance(self.regularization, str):
                if self.regularization.lower() == "l1":
                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):
                        # Add l1 regularization for all coefficients
                        term = np.asarray(
                            list(self.regularization_lambda * np.sign(coeffs_nonmon))
                            + list(self.regularization_lambda * np.sign(coeffs_mon))
                        )

                    elif isinstance(self.regularization_lambda, list):
                        # Add l1 regularization for all coefficients
                        term = (
                            np.asarray(
                                list(np.sign(coeffs_nonmon)) + list(np.sign(coeffs_mon))
                            )
                            * self.regularization_lambda[k]
                        )

                    else:
                        raise ValueError(
                            "Data type of regularization_lambda not understood. "
                            "Must be either scalar or list."
                        )

                    objective += term

                elif self.regularization == "l2":
                    # Regularization_lambda is identical for all parameters
                    if np.isscalar(self.regularization_lambda):
                        # Add l2 regularization for all coefficients
                        term = np.asarray(
                            list(self.regularization_lambda * 2 * coeffs_nonmon)
                            + list(self.regularization_lambda * 2 * coeffs_mon)
                        )

                    elif isinstance(self.regularization_lambda, list):
                        # Add l2 regularization for all coefficients
                        term = (
                            np.asarray(list(2 * coeffs_nonmon) + list(2 * coeffs_mon))
                            * self.regularization_lambda[k]
                        )

                    else:
                        raise ValueError(
                            "Data type of regularization_lambda not understood. "
                            "Must be either scalar or list."
                        )

                    objective += term

                else:
                    raise ValueError("regularization must be either 'l1' or 'l2'.")

            else:
                raise ValueError(
                    "The variable 'regularization' must be either None, 'l1', or 'l2'."
                )

        return objective

    def inverse_map(self, Z, X_star=None):
        """
        This function evaluates the inverse transport map, mapping samples from
        a multivariate standard Gaussian back to the target distribution. If
        X_precalc is specified, the map instead evaluates a conditional of the
        target distribution given X_precalc. The function assumes any
        precalculated output are the FIRST dimensions of the total output. If
        X_precalc is specified, its dimensions and the input dimensions must
        sum to the full dimensionality of sample space.

        Variables:

            Z
                [array] : N-by-D or N-by-(D-E) array of reference distribution
                samples to be mapped to the target distribution, where N is the
                number of samples, D is the number of target distribution
                dimensions, and E the number of pre-specified dimenions (if
                X_precalc is specified).

            X_star - [default = None]
                [None or array] : N-by-E array of samples in the space of the
                target distribution, used to condition the lower D-E dimensions
                during the inversion process.
        """
        # Create a local copy of Z to prevent overwriting the input
        Z = copy.copy(Z)

        # Extract number of samples
        N = Z.shape[0]

        # =====================================================================
        # No X_star was provided
        # =====================================================================

        if X_star is None:  # Yes
            # Initialize the output ensemble
            X = np.zeros((N, self.skip_dimensions + self.D))

            # Go through all dimensions
            for k in np.arange(0, self.D, 1):
                if (
                    isinstance(self.monotonicity, SeparableMonotonicity)
                    and self.monotonicity.alternate_root_finding
                ):
                    X = self.vectorized_root_search_alternate(Zk=Z[:, k], X=X, k=k)

                else:
                    X = self.vectorized_root_search_bisection(Zk=Z[:, k], X=X, k=k)

            # Undo the standardization
            X *= self.X_std
            X += self.X_mean

        # =====================================================================
        # X_star was provided, and matches the reduced map definition
        # =====================================================================

        if X_star is not None:
            if X_star.shape[-1] == self.skip_dimensions:  # Yes
                # Initialize the output ensemble
                X = np.zeros((N, self.skip_dimensions + self.D))

                # Standardize the precalculated values first
                X[:, : self.skip_dimensions] = copy.copy(X_star)
                X[:, : self.skip_dimensions] -= self.X_mean[: self.skip_dimensions]
                X[:, : self.skip_dimensions] /= self.X_std[: self.skip_dimensions]

                # Go through all dimensions
                for k in np.arange(0, self.D, 1):
                    if (
                        isinstance(self.monotonicity, SeparableMonotonicity)
                        and self.monotonicity.alternate_root_finding
                    ):
                        X = self.vectorized_root_search_alternate(Zk=Z[:, k], X=X, k=k)

                    else:
                        X = self.vectorized_root_search_bisection(Zk=Z[:, k], X=X, k=k)

                # Undo the standardization
                X *= self.X_std
                X += self.X_mean

            # =================================================================
            # A full map was defined, but so were precalculated values
            # =================================================================

            elif self.skip_dimensions == 0 and X_star is not None:
                # Create a local copy of skip_dimensions
                skip_dimensions = X_star.shape[-1]
                D = skip_dimensions + Z.shape[-1]

                # Initialize the output ensemble
                X = np.zeros((N, D))

                # Standardize the precalculated values first
                X[:, :skip_dimensions] = copy.copy(X_star)
                X[:, :skip_dimensions] -= self.X_mean[:skip_dimensions]
                X[:, :skip_dimensions] /= self.X_std[:skip_dimensions]

                # Go through all dimensions
                for i, k in enumerate(np.arange(skip_dimensions, D, 1)):
                    if (
                        isinstance(self.monotonicity, SeparableMonotonicity)
                        and self.monotonicity.alternate_root_finding
                    ):
                        X = self.vectorized_root_search_alternate(Zk=Z[:, i], X=X, k=k)

                    else:
                        X = self.vectorized_root_search_bisection(Zk=Z[:, i], X=X, k=k)

                # Undo the standardization
                X *= self.X_std
                X += self.X_mean

        return X[:, self.skip_dimensions :]

    def vectorized_root_search_bisection(
        self, X, Zk, k, max_iterations=100, threshold=1e-9, start_distance=2
    ):
        """
        This function searches for the roots of the k-th map component through
        bisection. It is called in the inverse_map function.

        Variables:

            X
                [array] : N-by-k array of samples inverted so far, where the
                k-th column still contains the reference samples used as a
                residual in the root finding process

            Zk
                [vector] : a vector containing the target values in the k-th
                dimension, for which the root finding algorithm must solve.

            k
                [integer] : an integer variable defining what map component
                is being evaluated. Corresponds to a dimension of sample space.

            max_iterations - [default = 100]
                [integer] : number of function calls before the algorithm stops
                continuing the root search to avoid becoming stuck in an
                endless loop.

            threshold - [default = 1E-9]
                [float] : threshold value below which the algorithm assumes the
                root finding problem to be solves.

            start_distance - [default = 2]
                [integer] : starting distance from the origin for the interval
                edges used for bisection. This window can be moved by the
                algorithm should the root not lie within.

        """

        # Extract number of particles
        N = X.shape[0]

        # Check whether samples have been marked for removal
        indices = np.arange(N)  # Indices of all particles
        failure = np.isnan(
            X[:, self.skip_dimensions + k]
        )  # Particles marked for removal
        indices = indices[~failure]  # Kill the associated indices

        # Initialize the start bisection points
        bsct_pts = np.zeros((N, 2))
        bsct_pts[:, 0] = -np.ones(N) * start_distance
        bsct_pts[:, 1] = +np.ones(N) * start_distance

        bsct_out = np.zeros((N, 2))

        # Calculate the initial bracket
        X[indices, self.skip_dimensions + k] = bsct_pts[indices, 0]
        bsct_out[indices, 0] = self.s(x=X[indices, :], k=k) - Zk[indices]
        X[indices, self.skip_dimensions + k] = bsct_pts[indices, 1]
        bsct_out[indices, 1] = self.s(x=X[indices, :], k=k) - Zk[indices]

        # Sort the bsct_pts so that bsct_out is increasing
        for n in indices:
            if bsct_out[n, 0] > bsct_out[n, 1]:
                dummy = bsct_out[n, 0]
                bsct_out[n, 0] = bsct_out[n, 1]
                bsct_out[n, 1] = dummy

                dummy = bsct_pts[n, 0]
                bsct_pts[n, 0] = bsct_pts[n, 1]
                bsct_pts[n, 1] = dummy

        # =====================================================================
        # Shift windows
        # =====================================================================

        # An initial proposal for the windows has been made. If zero is not
        # between the two bsct_pts, we must shift the window

        # Create a copy of indices
        shiftindices = copy.copy(indices)

        # Where the product has different signs, zero is in-between
        failure = np.where(np.prod(bsct_out[shiftindices, :], axis=1) > 0)[0]
        shiftindices = shiftindices[failure]

        # While
        while len(shiftindices) > 0:
            # Re-sort the windows if necessary
            for n in shiftindices:
                if bsct_out[n, 0] > bsct_out[n, 1]:
                    dummy = bsct_out[n, 0]
                    bsct_out[n, 0] = bsct_out[n, 1]
                    bsct_out[n, 1] = dummy

                    dummy = bsct_pts[n, 0]
                    bsct_pts[n, 0] = bsct_pts[n, 1]
                    bsct_pts[n, 1] = dummy

            # Find out the sign of the points which were NOT successful
            # sign_failure    = np.sign(np.sum(bsct_out[shiftindices,:],axis=1))
            sign_failure = np.sign(bsct_out[shiftindices, 0])

            # This difference tells us how much we must shift X to move RIGHT
            difference = np.diff(bsct_pts[shiftindices, :], axis=1)[:, 0]

            # For positive signs, shift the window to the LEFT bound
            failure_pos = np.where(sign_failure > 0)[0]
            bsct_pts[shiftindices[failure_pos], 1] = copy.copy(
                bsct_pts[shiftindices[failure_pos], 0]
            )
            bsct_pts[shiftindices[failure_pos], 0] -= difference[failure_pos] * 2

            # Re-simulate that
            bsct_out[shiftindices[failure_pos], 1] = copy.copy(
                bsct_out[shiftindices[failure_pos], 0]
            )
            X[shiftindices[failure_pos], self.skip_dimensions + k] = copy.copy(
                bsct_pts[shiftindices[failure_pos], 0]
            )
            bsct_out[shiftindices[failure_pos], 0] = copy.copy(
                self.s(x=X[shiftindices[failure_pos], :], k=k)
                - Zk[shiftindices[failure_pos]]
            )

            # For negative signs, shift the window to the RIGHT bound
            failure_neg = np.where(sign_failure < 0)[0]
            bsct_pts[shiftindices[failure_neg], 0] = copy.copy(
                bsct_pts[shiftindices[failure_neg], 1]
            )
            bsct_pts[shiftindices[failure_neg], 1] += difference[failure_neg] * 2

            # Re-simulate that
            bsct_out[shiftindices[failure_neg], 0] = copy.copy(
                bsct_out[shiftindices[failure_neg], 1]
            )
            X[shiftindices[failure_neg], self.skip_dimensions + k] = copy.copy(
                bsct_pts[shiftindices[failure_neg], 1]
            )
            bsct_out[shiftindices[failure_neg], 1] = copy.copy(
                self.s(x=X[shiftindices[failure_neg], :], k=k)
                - Zk[shiftindices[failure_neg]]
            )

            # Where the product has different signs, zero is in-between
            failure = np.where(np.prod(bsct_out[shiftindices, :], axis=1) > 0)[0]
            shiftindices = shiftindices[failure]

        # =====================================================================
        # Start the actual root search
        # =====================================================================

        # Prepare iteration counter
        itr_counter = 0

        # Start optimization loop
        while np.sum(indices) > 0 and itr_counter < max_iterations:
            itr_counter += 1

            # Propose bisection
            mid_pt = np.mean(bsct_pts[indices, :], axis=1)

            # Calculate the biscetion point output
            X[indices, self.skip_dimensions + k] = mid_pt
            mid_out = self.s(x=X[indices, :], k=k) - Zk[indices]

            # Set the Lower or upper boundary depending on the sign of mid_out
            below = np.where(mid_out < 0)[0]
            above = np.where(mid_out > 0)[0]

            bsct_pts[indices[below], 0] = copy.copy(mid_pt[below])

            bsct_pts[indices[above], 1] = copy.copy(mid_pt[above])

            not_converged = np.where(np.abs(mid_out) > threshold)
            indices = indices[not_converged]

        if itr_counter == max_iterations and self.verbose:
            print(
                "WARNING: root search for particles "
                + str(indices)
                + " stopped"
                + " at maximum iterations."
            )

        return X

    def vectorized_root_search_alternate(
        self, X, Zk, k, start_distance=10, resolution=1001
    ):
        """
        This function is an alternative root search routine, not based on
        bisection but interpolation.

        Only used for "separable monotonicity"

        Variables:

            X
                [array] : N-by-k array of samples inverted so far, where the
                k-th column still contains the reference samples used as a
                residual in the root finding process

            Zk
                [vector] : a vector containing the target values in the k-th
                dimension, for which the root finding algorithm must solve.

            k
                [integer] : an integer variable defining what map component
                is being evaluated. Corresponds to a dimension of sample space.

            max_iterations - [default = 100]
                [integer] : number of function calls before the algorithm stops
                continuing the root search to avoid becoming stuck in an
                endless loop.

            threshold - [default = 1E-9]
                [float] : threshold value below which the algorithm assumes the
                root finding problem to be solves.

            start_distance - [default = 2]
                [integer] : starting distance from the origin for the interval
                edges used for bisection. This window can be moved by the
                algorithm should the root not lie within.

        """
        from scipy.interpolate import interp1d

        # Create a local copy
        X = copy.copy(X)

        # ---------------------------------------------------------------------
        # Step 1: For separable monotonicity, all non-monotone terms are just
        # constant offsets. So let's just calculate that once

        offset = np.dot(
            self.fun_nonmon[k](copy.copy(X), self), self.coeffs_nonmon[k][:, np.newaxis]
        )[:, 0]

        # ---------------------------------------------------------------------
        # Step 2: Evaluate the forward map
        pts = np.linspace(-start_distance, start_distance, resolution)

        # Create a fake X vector for the evaluation points
        fakeX = np.zeros((resolution, X.shape[-1]))
        fakeX[:, self.skip_dimensions + k] = copy.copy(pts)

        # Evaluate the monotone map part
        out = np.dot(self.fun_mon[k](fakeX, self), self.coeffs_mon[k][:, np.newaxis])[
            :, 0
        ]

        # ---------------------------------------------------------------------
        # Step 3: Create a 1D interpolator
        itp = interp1d(x=out, y=pts, fill_value="extrapolate")

        # ---------------------------------------------------------------------
        # Step 4: Evaluate the 1D interpolator

        # Find the target values
        target = -offset + Zk

        # Find the target root
        result = itp(target)

        # Save the result
        X[:, self.skip_dimensions + k] = copy.copy(result)

        return X

    def GaussQuadrature(
        self,
        f,
        a,
        b,
        order=100,
        args=None,
        Ws=None,
        xis=None,
        adaptive=False,
        threshold=1e-6,
        increment=1,
        verbose=False,
        full_output=False,
    ):
        """
        This function implements a Gaussian quadrature numerical integration
        scheme. It is used if the monotonicity = 'integrated rectifier', for
        which monotonicity is ensured by integrating a strictly positive
        function obtained from a rectifier.

        Variables:

            ===================================================================
            General variables
            ===================================================================

            f
                [function] : function to be integrated element-wise.

            a
                [float or vector] : lower bound for integration, defined as
                either a scalar or a vector.

            b
                [float or vector] : upper bound for integration, defined as
                either a scalar or a vector.

            order - [default = 100]
                [integer] : order of the Legendre polynomial used for the
                integration scheme..

            args - [default = None]
                [None or dictionary] : a dictionary with supporting keyword
                arguments to be passed to the function.

            Ws - [default = None]
                [vector] : weights of the integration points, can be calculated
                in advance to speed up the computation. Is calculated by the
                integration scheme, if not specified.

            xis - [default = None]
                [vector] : positions of the integration points, can be
                calculated in advance to speed up the computation. Is
                calculated by the integration scheme, if not specified.

            full_output - [default = False]
                [boolean] : Flag for whether the positions and weights of the
                integration points should returned along with the integration
                results. If True, returns a tuple with (results,order,xis,Ws).
                If False, only returns results.

            ===================================================================
            Adaptive integration variables
            ===================================================================

            adaptive - [default = False]
                [boolean] : flag which determines whether the numerical scheme
                should adjust the order of the Legendre polynomial adaptively
                (True) or use the integer provided by 'order' (False).

            threshold - [default = 1E-6]
                [float] : threshold for the difference in the adaptive
                integration, adaptation stops after difference in integration
                result falls below this value.

            increment - [default = 1]
                [integer] : increment by which the order is increased in each
                adaptation cycle. Higher values correspond to larger steps.

            verbose - [default = False]
                [boolean] : flag which determines whether information about the
                integration process should be printer to console (True) or not
                (False).

        """
        # =========================================================================
        # Here the actual magic starts
        # =========================================================================

        # If adaptation is desired, we must iterate; prepare a flag for this
        repeat = True
        iteration = 0

        # Iterate, if adaptation = True; Otherwise, iteration stops after one round
        while repeat:
            # Increment the iteration counter
            iteration += 1

            # If required, determine the weights and positions of the integration
            # points; always required if adaptation is active
            if Ws is None or xis is None or adaptive:
                # Weights and integration points are not specified; calculate them
                # To get the weights and positions of the integration points, we must
                # provide the *order*-th Legendre polynomial and its derivative
                # As a first step, get the coefficients of both functions
                coefs = np.zeros(order + 1)
                coefs[-1] = 1
                coefs_der = np.polynomial.legendre.legder(coefs)

                # With the coefficients defined, define the Legendre function
                LegendreDer = np.polynomial.legendre.Legendre(coefs_der)

                # Obtain the locations of the integration points
                xis = np.polynomial.legendre.legroots(coefs)

                # Calculate the weights of the integration points
                Ws = 2.0 / ((1.0 - xis**2) * (LegendreDer(xis) ** 2))

            # If any of the boundaries is a vector, vectorize the operation
            if not np.isscalar(a) or not np.isscalar(b):
                # If only one of the bounds is a scalar, vectorize it
                if np.isscalar(a) and not np.isscalar(b):
                    a = np.ones(b.shape) * a
                if np.isscalar(b) and not np.isscalar(a):
                    b = np.ones(a.shape) * b

                # Alternative approach, more amenable to dimension-sensitivity in
                # the function f. To speed up computation, pre-calculate the limit
                # differences and sum
                lim_dif = b - a
                lim_sum = b + a
                result = np.zeros(a.shape)

                # print('limdifshape:'+str(lim_dif.shape))
                # print('resultshape:'+str(result.shape))

                # =============================================================
                # To understand what's happening here, consider the following:
                #
                # lim_dif and lim_sum   - shape (N)
                # funcres               - shape (N) up to shape (N-by-C-by-C)

                # If no additional arguments were given, simply call the function
                if args is None:
                    result = (
                        lim_dif
                        * 0.5
                        * (Ws[0] * f(lim_dif * 0.5 * xis[0] + lim_sum * 0.5))
                    )

                    for i in np.arange(1, len(Ws)):
                        result += (
                            lim_dif
                            * 0.5
                            * (Ws[i] * f(lim_dif * 0.5 * xis[i] + lim_sum * 0.5))
                        )

                # Otherwise, pass the arguments on as well
                else:
                    funcres = f(lim_dif * 0.5 * xis[0] + lim_sum * 0.5, *args)

                    # =========================================================
                    # Depending on what shape the output function returns, we
                    # must take special precautions to ensure the product works
                    # =========================================================

                    # If the function output is the same size as its input
                    if len(funcres.shape) == len(lim_dif.shape):
                        result = lim_dif * 0.5 * (Ws[0] * funcres)

                        for i in np.arange(1, len(Ws)):
                            funcres = f(lim_dif * 0.5 * xis[i] + lim_sum * 0.5, *args)

                            result += lim_dif * 0.5 * (Ws[i] * funcres)

                    # If the function output has one dimension more than its
                    # corresponding input
                    elif len(funcres.shape) == len(lim_dif.shape) + 1:
                        result = np.einsum("i,ij->ij", lim_dif * 0.5 * Ws[0], funcres)

                        for i in np.arange(1, len(Ws)):
                            funcres = f(lim_dif * 0.5 * xis[i] + lim_sum * 0.5, *args)

                            result += np.einsum(
                                "i,ij->ij", lim_dif * 0.5 * Ws[i], funcres
                            )

                    # If the function output has one dimension more than its
                    # corresponding input
                    elif len(funcres.shape) == len(lim_dif.shape) + 2:
                        result = np.einsum("i,ijk->ijk", lim_dif * 0.5 * Ws[0], funcres)

                        for i in np.arange(1, len(Ws)):
                            funcres = f(lim_dif * 0.5 * xis[i] + lim_sum * 0.5, *args)

                            result += np.einsum(
                                "i,ijk->ijk", lim_dif * 0.5 * Ws[i], funcres
                            )

                    else:
                        msg = (
                            f"Shape of input dimension is {lim_sum.shape} "
                            f"and shape of output dimension is {funcres.shape}. "
                            "Currently, we have only implemented situations in "
                            "which input and output are the same shape, or where "
                            "output is one or two dimensions larger."
                        )
                        raise Exception(msg)

            else:
                # Now start the actual computation.

                # If no additional arguments were given, simply call the function
                if args is None:
                    result = (
                        (b - a)
                        * 0.5
                        * np.sum(Ws * f((b - a) * 0.5 * xis + (b + a) * 0.5))
                    )
                # Otherwise, pass the arguments on as well
                else:
                    result = (
                        (b - a)
                        * 0.5
                        * np.sum(Ws * f((b - a) * 0.5 * xis + (b + a) * 0.5, *args))
                    )

            # if adaptive, store results for next iteration
            if adaptive:
                # In the first iteration, just store the results
                if iteration == 1:
                    previous_result = copy.copy(result)

                # In later iterations, check integration process
                else:
                    # How much did the results change?
                    change = np.abs(result - previous_result)

                    # Check if the change in results was sufficient
                    if np.max(change) < threshold or iteration > 1000:
                        # Stop iterating
                        repeat = False

                        if iteration > 1000 and self.verbose:
                            print(
                                "WARNING: Adaptive integration stopped after "
                                + "1000 iteration cycles. Final change: "
                                + str(change)
                            )

                        # Print the final change if required
                        if verbose and self.verbose:
                            print(
                                "Final maximum change of Gauss Quadrature: "
                                + str(np.max(change))
                            )

                # If we must still continue repeating, increment order and store
                # current result for next iteration
                if repeat:
                    order += increment
                    previous_result = copy.copy(result)

            # If no adaptation is required, simply stop iterating
            else:
                repeat = False

        # If full output is desired
        if full_output:
            result = (result, order, xis, Ws)

        if verbose and self.verbose:
            print("Order: " + str(order))

        return result

    def projectedNewton(
        self,
        x0,
        fun,
        args=None,
        method="trueHessian",
        rtol_Jdef=1e-6,
        rtol_gdef=1e-6,
        itmaxdef=30,
        epsilon=0.01,
    ):
        import scipy

        def Armijo(xk, gk, pk, Jk, Ik, fun, args=None, itmax=15, sigma=1e-4, beta=2):
            """
            xk      : current iterate
            gk      : gradient at gk
            pk      : search direction at xk
            Jxk     : objective function at xk
            J       : objective function
            Ik      : set of locally optimal coordinates on the boundary
            """
            # Iteration counter
            it = 0

            Jxk_alpha_lin = Jk + 1

            alpha = beta

            while Jk < Jxk_alpha_lin and it < itmax:
                alpha /= beta
                alpha_pk = alpha * pk

                xk_alpha = np.maximum(0, xk - alpha_pk)

                Jxk_alpha = fun(
                    xk_alpha[:, 0], A=args[0], k=args[1], all_outputs=False
                )  # Does not compute jac and hess
                alpha_pk[Ik] = xk[Ik] - xk_alpha[Ik]

                Jxk_alpha_lin = Jxk_alpha + sigma * np.dot(gk.T, alpha_pk)
                # print('jxk: '+str(Jxk_alpha_lin)+' | '+str(Jk))

                it += 1

            if Jk < Jxk_alpha_lin:
                # print('Line Search reached max number of iterations.')

                xkh = xk
                xkh[Ik] = 0
                index = np.where(np.logical_and(xkh > 0, pk > 0))[0]
                # print(index)
                # print(xk)
                # print(pk)
                if len(index) == 0:
                    alpha = 1
                else:
                    alpha = np.min(xk[index] / pk[index])

                # raise Exception

            return alpha

        def projectGradient(xk, gk):
            Pgk = gk
            Pgk[np.where(np.logical_and(xk == 0, gk >= 0))] = 0

            return Pgk

        def is_symmetric(a, tol=1e-8):
            return np.all(np.abs(a - a.T) < tol)

        rtol_J = rtol_Jdef
        rtol_g = rtol_gdef
        itmax = itmaxdef

        # Initialize parameters
        if len(np.asarray(x0).shape) < 2:
            x0 = np.asarray(x0)[:, np.newaxis]
            # raise Exception("x0 must be a column vector.")
        if any(x0 < 0):
            raise Exception("Initial conditions must be positive.")

        xk = copy.copy(x0)
        Jk, gk, Hk = fun(xk[:, 0], A=args[0], k=args[1])
        gk = gk[:, np.newaxis]
        dim = len(x0)

        norm_Pg0 = np.linalg.norm(projectGradient(xk, gk))
        norm_Pgk = norm_Pg0

        tol_g = norm_Pg0 * rtol_g
        rdeltaJ = rtol_J + 1
        Jold = Jk
        it = 0

        # Start iterating
        while rdeltaJ > rtol_J and norm_Pgk > tol_g and it < itmax:
            # Define search direction
            wk = np.linalg.norm(xk - np.maximum(0, xk - gk))

            epsk = np.minimum(epsilon, wk)

            Ik = np.where(np.logical_and(xk[:, 0] <= epsk, gk[:, 0] > 0))[0]

            if not len(Ik) == 0:  # If Ik is not empty
                hxk = np.diag(Hk)  # Extract the Hessian's diagonal
                zk = np.zeros(dim)
                zk[Ik] = copy.copy(hxk[Ik])
                Hk[Ik, :] = 0
                Hk[:, Ik] = 0
                Hk += np.diag(zk)  # Buffer the Hessian's diagonal

            if method == "trueHessian":
                Lk = scipy.linalg.cholesky(Hk, lower=True)
                pk = scipy.linalg.solve(Lk.T, scipy.linalg.solve(Lk, gk))

            elif method == "modHessian":
                itmax = 100

                try:
                    Lk = scipy.linalg.cholesky(Hk, lower=True)
                    pk = scipy.linalg.solve(Lk.T, scipy.linalg.solve(Lk, gk))

                except scipy.linalg.LinAlgError:  # Hessian is not pd
                    if not is_symmetric(Hk):
                        Hk = Hk + Hk.T
                        Hk /= 2

                    print("Hessian is not pd.")

                    # Eigendecomposition
                    eigval, eigvec = scipy.linalg.eig(Hk)

                    eigval = np.abs(eigval)
                    eigval = np.maximum(eigval, 1e-8)[:, np.newaxis]

                    pk = np.dot(eigvec, ((1 / eigval) * np.dot(eigvec.T, gk)))

            elif method == "gradient":  # Revert to gradient descent
                itmax = 1000
                pk = gk

            else:
                raise Exception("method not implemented yet.")

            # Do a line search
            alphak = Armijo(xk=xk, gk=gk, pk=pk, Jk=Jk, Ik=Ik, fun=fun, args=args)

            # Update
            # xk      = np.maximum(0, xk - np.dot(alphak,pk))
            xk = np.maximum(0, xk - alphak * pk)

            # Evaluate objective function
            Jk, gk, Hk = fun(xk[:, 0], A=args[0], k=args[1])
            gk = gk[:, np.newaxis]

            # Convergence criteria
            rdeltaJ = np.abs(Jk - Jold) / np.abs(Jold)
            Jold = Jk
            norm_Pgk = np.linalg.norm(projectGradient(xk, gk))
            it += 1

        # Broken out of while loop
        xopt = copy.copy(xk)[:, 0]

        return xopt

    def adaptation_cross_terms(self, increment=1e-6, chronicle=False):
        """
        This function adapts a map with cross-terms.

        """
        from scipy.optimize import minimize

        def cell_to_term(cell):
            # Create basis function
            term = []
            for idx, order in enumerate(cell):
                term += [idx] * order

            # If term is nonlinear, add a Hermite function modifier
            if self.polynomial_type.lower() == "hermite function" and len(term) > 0:
                term += ["HF"]

            return term

        def construct_component_function(multi_index_matrix):
            # Find all active and proposed cells
            nonzero_cells = np.asarray(np.where(multi_index_matrix != 0)).T

            # Create couters for the monotone and nonmonotone terms
            term_counter = -1

            # Initiate a list for the indices of the proposed and original cells
            proposed_cells = []
            original_cells = []

            # Initiate monotone and nonmonotone lists
            monotone = []
            nonmonotone = []

            # Go through all cells
            for cell in nonzero_cells:
                # Increment the term counter
                term_counter += 1

                # Store this index, if this term was proposed
                if self.multi_index_matrix[tuple(cell)] < 0:
                    proposed_cells.append(term_counter)
                else:
                    original_cells.append(term_counter)

                # This cell depends on the last dimension
                if cell[-1] > 0:
                    term = cell_to_term(cell)

                    # Add it to the monotone term
                    monotone.append(copy.deepcopy(term))

                # This cell does not depend on the last dimension
                else:
                    term = cell_to_term(cell)

                    # Add it to the monotone term
                    nonmonotone.append(copy.deepcopy(term))

            return monotone, nonmonotone, proposed_cells, original_cells

        # If we are chronicling the results, create a dictionary for the outputs
        if chronicle:
            chronicle_dict = {}

        # So, let's adapt a map with cross-terms. Let's go through each of
        # the map component functions
        for k in range(self.D):
            # =============================================================
            # Initiation
            # =============================================================

            # If we chronicle, create a key for this map component
            if chronicle:
                chronicle_dict[k] = {}

            # Create a multi-index matrix
            multi_index_matrix = np.zeros(
                tuple([self.adaptation_max_order + 1] * (k + 1 + self.skip_dimensions)),
                dtype=int,
            )

            # The zero entry corresponds to a constant; activate it
            index = [0] * (k + 1 + self.skip_dimensions)
            multi_index_matrix[tuple(index)] = 1

            # One down in the last dimension corresponds to a marginal map;
            # activate that one too
            index = [0] * (k + self.skip_dimensions) + [1]
            multi_index_matrix[tuple(index)] = 1

            # Store this matrix
            self.multi_index_matrix = multi_index_matrix

            # Concatenate the coefficients, and define the divisor
            coeffs = np.asarray(
                list(copy.copy(self.coeffs_nonmon[k]))
                + list(copy.copy(self.coeffs_mon[k]))
            )
            div = len(self.coeffs_nonmon[k])

            # Minimize the objective function
            opt = minimize(
                method="BFGS",  #'L-BFGS-B',
                fun=self.objective_function,
                jac=self.objective_function_jacobian,
                x0=coeffs,
                args=(k, div),
            )

            # Retrieve the optimized coefficients
            coeffs = copy.copy(opt.x)

            # Save the optimized coefficients
            self.coeffs_nonmon[k] = copy.copy(coeffs[:div])
            self.coeffs_mon[k] = copy.copy(coeffs[div:])

            # =============================================================
            # Begin iteration
            # =============================================================

            repeat = True
            iterations = 0

            # If we chronicle, store results for this iteration
            if chronicle:
                chronicle_dict[k][iterations] = {
                    "monotone": copy.deepcopy(self.monotone[k]),
                    "nonmonotone": copy.deepcopy(self.nonmonotone[k]),
                    "coeffs_nonmon": copy.copy(self.coeffs_nonmon[k]),
                    "coeffs_mon": copy.copy(self.coeffs_mon[k]),
                    "multi_index_matrix": copy.copy(self.multi_index_matrix),
                }

            while repeat:
                # Increment the iteration counter
                iterations += 1

                # =========================================================
                # Find all candidate cells
                # =========================================================

                # Find all active cells
                nonzero_cells = np.asarray(np.where(self.multi_index_matrix > 0)).T

                # Go through all of these cells, check for zero-valued neighbours
                for cell in nonzero_cells:
                    # Go through all dimensions of this cell
                    for idx in range(k + 1 + self.skip_dimensions):
                        # Does the cell before exist?
                        if cell[idx] - 1 >= 0:
                            # Create index before
                            index = list(copy.copy(cell))
                            index[idx] -= 1

                            # Check multi_index_matrix
                            if self.multi_index_matrix[tuple(index)] <= 0:
                                self.multi_index_matrix[tuple(index)] -= 1

                        # Does the cell after exist?
                        if cell[idx] + 1 < self.adaptation_max_order + 1:
                            # Create index before
                            index = list(copy.copy(cell))
                            index[idx] += 1

                            # Check multi_index_matrix
                            if self.multi_index_matrix[tuple(index)] <= 0:
                                self.multi_index_matrix[tuple(index)] -= 1

                # Are there any cells which were proposed?
                proposed_cells = np.asarray(np.where(self.multi_index_matrix < 0)).T

                # If no cells have been proposed, end the while loop
                if len(proposed_cells) == 0:
                    break

                # Find all multi_index_matrix entries at the edges
                for cell in proposed_cells:
                    # Go through all coordinate indices
                    for val in cell:
                        # If this index is on the lower boundary
                        if val == 0:
                            # Reduce the multi_index_matrix entry further
                            self.multi_index_matrix[tuple(cell)] -= 1

                # Find the reduced set of indices - only those whose count
                # is equal to the dimension of the matrix
                proposed_cells = np.asarray(
                    np.where(self.multi_index_matrix <= -(k + 1 + self.skip_dimensions))
                ).T

                print(self.multi_index_matrix)

                # =========================================================
                # Iterate through all proposed cells
                # =========================================================

                # Extract initial coefficients
                coeffs = copy.copy(
                    np.asarray(
                        list(copy.copy(self.coeffs_nonmon[k]))
                        + list(copy.copy(self.coeffs_mon[k]))
                    )
                )

                # Calculate the reference objective function
                obj_ref = self.objective_function(coeffs=coeffs, k=k, div=div)

                # Pre-allocate space for the gradients
                grads = np.zeros(len(proposed_cells))

                # Iterate through all proposed cells
                for idx, cell in enumerate(proposed_cells):
                    # Reset the multi_index_matrix
                    self.multi_index_matrix[self.multi_index_matrix < 0] = 0

                    # Write in the cell
                    self.multi_index_matrix[tuple(cell)] = -1

                    # Update the map component specifications
                    monotone, nonmonotone, prop, orig = construct_component_function(
                        multi_index_matrix=self.multi_index_matrix
                    )

                    # =====================================================
                    # Update the stored maps with the candidate components
                    # =====================================================

                    # Store these map components
                    self.monotone[k] = copy.deepcopy(monotone)
                    self.nonmonotone[k] = copy.deepcopy(nonmonotone)

                    # Re-write the functions
                    self.function_constructor_alternative(k=k)

                    # Update the coefficients (after construction to get correct sizes)
                    coeffs_new = (
                        np.ones(len(self.fun_nonmon[k]) + len(self.fun_mon[k]))
                        * self.coeffs_init
                        + increment
                    )
                    coeffs_new[orig] = copy.copy(coeffs)

                    # Update the divisor
                    div = len(self.fun_nonmon[k])

                    # Update the basis functions
                    self.Psi_mon[k] = copy.copy(
                        self.fun_mon[k](copy.copy(self.X), self)
                    )
                    self.Psi_nonmon[k] = copy.copy(
                        self.fun_nonmon[k](copy.copy(self.X), self)
                    )

                    # =====================================================
                    # Determine the gradient
                    # =====================================================

                    # Evaluate the gradient through finite differences
                    obj_off = self.objective_function(coeffs=coeffs_new, k=k, div=div)

                    # Finite difference evaluation
                    grads[idx] = (obj_off - obj_ref) / increment

                # =========================================================
                # Add the cell with the strongest gradient
                # =========================================================

                # Find the strongest gradient
                minidx = np.where(np.abs(grads) == np.max(np.abs(grads)))[0][0]

                # Reset the multi_index_matrix
                self.multi_index_matrix[self.multi_index_matrix < 0] = 0

                # Find the entry we want to add
                added_cell = proposed_cells[minidx]

                # print(added_cell)

                # Set that entry to "proposed" in the multi_index_matrix
                self.multi_index_matrix[tuple(added_cell)] = -1

                # =========================================================
                # Construct the new map components
                # =========================================================

                # Find all active and proposed cells
                nonzero_cells = np.asarray(np.where(self.multi_index_matrix != 0)).T

                # Update the map component specifications
                monotone, nonmonotone, prop, orig = construct_component_function(
                    multi_index_matrix=self.multi_index_matrix
                )

                # Set that entry to "active" in the multi_index_matrix
                self.multi_index_matrix[tuple(added_cell)] = 1

                # =========================================================
                # Update the stored maps with the candidate components
                # =========================================================

                # Store these map components
                self.monotone[k] = copy.deepcopy(monotone)
                self.nonmonotone[k] = copy.deepcopy(nonmonotone)

                # Re-write the functions
                self.function_constructor_alternative(k=k)

                # Update the coefficients (after construction to get correct sizes)
                coeffs_new = (
                    np.ones(len(self.fun_nonmon[k]) + len(self.fun_mon[k]))
                    * self.coeffs_init
                )
                coeffs_new[orig] = copy.copy(coeffs)

                # Update the divisor
                div = len(self.fun_nonmon[k])

                # Update the basis functions
                self.Psi_mon[k] = copy.copy(self.fun_mon[k](copy.copy(self.X), self))
                self.Psi_nonmon[k] = copy.copy(
                    self.fun_nonmon[k](copy.copy(self.X), self)
                )

                # =========================================================
                # Update the coefficients
                # =========================================================

                # # Minimize the objective function
                # opt     = minimize(
                #     method  = 'BFGS',#'L-BFGS-B',
                #     fun     = self.objective_function,
                #     jac     = self.objective_function_jacobian,
                #     x0      = coeffs_new,
                #     args    = (k,div))

                # Minimize the objective function
                opt = minimize(
                    method="L-BFGS-B",
                    fun=self.objective_function,
                    x0=coeffs_new,
                    args=(k, div),
                )

                # Retrieve the optimized coefficients
                coeffs = copy.copy(opt.x)

                # Save the optimized coefficients
                self.coeffs_nonmon[k] = copy.copy(coeffs[:div])
                self.coeffs_mon[k] = copy.copy(coeffs[div:])

                # If we chronicle, store results for this iteration
                if chronicle:
                    chronicle_dict[k][iterations] = {
                        "monotone": copy.deepcopy(self.monotone[k]),
                        "nonmonotone": copy.deepcopy(self.nonmonotone[k]),
                        "coeffs_nonmon": copy.copy(self.coeffs_nonmon[k]),
                        "coeffs_mon": copy.copy(self.coeffs_mon[k]),
                        "multi_index_matrix": copy.copy(self.multi_index_matrix),
                    }

                # =========================================================
                # Should we stop?
                # =========================================================

                if iterations >= self.adaptation_max_iterations:
                    break

        # =================================================================
        # Are we done yet?
        # =================================================================

        if chronicle:
            import pickle

            # Pickle the adaptation dictionary
            pickle.dump(chronicle_dict, open("dictionary_adaptation_chronicle.p", "wb"))
