"""geometric_brownian.py"""

from typing import NoReturn, Protocol, Optional, Union

import numpy as np

import brownian_motion

# =============================================================================
# generate constant processes


class ConstantProcs:
    """
    Constant drift  matrix, each column of the matrix is a 1D process.
    - T is the sample size (rows of the matrix)
    - constants is either the constant for all processes (non random)
        or a tuple of constants one for each process
    - n_procs is the number of process (columns in the matix),
        if constants is tuple, this argument is ignored,
        if constants is a float then n_procs cannot be None,
        otherwise will raise ValueError
    The returned matrix columns are indexes in the same order as
    the constants tuple (in case it is a tuple).
    """

    def __init__(
        self,
        T: int,
        constants: Union[float, tuple[float, ...]],
        n_procs: Optional[int] = None,
    ) -> None:
        self.T = T
        self.constants = constants
        self._n_procs = n_procs

        self._n_procs_ = self._get_n_procs()

    @property
    def n_procs(self) -> int:
        """The number of drift processes."""
        return self._n_procs_

    def get_proc(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2D array, each column is a 1D process,
        the number of rows define the sample size,
        random_state not None to reproduce results.
        """
        if isinstance(self.constants, tuple):
            return (
                np.repeat(self.constants, self.T, axis=0).reshape(-1, self.T).T
            )
        return self.constants * np.ones((self.T, self._n_procs_))

    def _get_n_procs(self) -> int:
        """
        Check what is the correct n_proc, depending on params type and
        n_procs type.
        - if constants is float then n_procs cannot be None,
            the value of n_procs is then passed as the correct one.
        - if constants is tuple n_procs input argument is
            ignored and the correct n_procs is then the size of the tuple.
        """
        if isinstance(self.constants, tuple):
            return len(self.constants)
        elif self._n_procs is None:
            raise ValueError(
                "If constants is not tuple, n_procs cannot be None."
            )
        return self._n_procs


# =============================================================================
# mu, drift processes

# ---------------------------------------------------
# Protocol (interface)


class Drift(Protocol):
    """
    Base class for drift processes matrix.
    Each column of the drift matrix is a 1D process.
    """

    @property
    def sample_size(self) -> int:
        """The sample size T of the drift processes."""

    @property
    def n_procs(self) -> int:
        """The number of drift processes."""

    def get_mu(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2D array, each column is a 1D process,
        the number of rows define the sample size.,
        random_state not None to reproduce results.
        """


# ---------------------------------------------------
# Constant drift


class ConstantDrift(ConstantProcs):
    """
    Implements the Drift Protocol.
    Constant drift processes matrix, each column of the drift
    matrix is a 1D process.
    - T is the sample size (rows of the matrix)
    - mu_constants is either the constant for all processes (non random)
        or a tuple of constants one for each process
    - n_procs is the number of process (columns in the matix),
        if mu_constants is tuple, this argument is ignored,
        if mu_constants is a float then n_procs cannot be None,
        otherwise will raise ValueError
    The returned matrix columns are indexes in the same order as
    the mu_constants tuple (in case it is a tuple).
    """

    def __init__(
        self,
        T: int,
        mu_constants: Union[float, tuple[float, ...]],
        n_procs: Optional[int] = None,
    ) -> None:
        super().__init__(T, mu_constants, n_procs)
        self.mu_constants = mu_constants
        self.T = T
        self._n_procs = n_procs

    @property
    def sample_size(self) -> int:
        """The sample size T of the drift processes."""
        return self.T

    @property
    def n_procs(self) -> int:
        """The number of drift processes."""
        return super().n_procs

    def get_mu(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2D array, each column is a 1D process,
        the number of rows define the sample size,
        random_state not None to reproduce results.
        """
        return super().get_proc(random_state)


# ---------------------------------------------------
# drift constants from data


def estimate_drift_constants(proc_mat: np.ndarray) -> tuple[float, ...]:
    """
    Estimate drift constants from data (geometric Brownian motion paths).
    - proc_mat is a 2D array,  each column is a process
    Returns a tuple of floats indexed with the same order as proc_mat
    columns.
    """
    diffusion_increments = np.diff(proc_mat, axis=0) / proc_mat[:-1, :]
    increment_mus = np.mean(diffusion_increments, axis=0)
    return tuple([float(mu) for mu in increment_mus])


# =============================================================================
# sigma (standard deviation)

# ---------------------------------------------------
# Protocol (interface)


class Sigma(Protocol):
    """
    Base class for sigma processes matrix.
    Each column of the sigma matrix is a 1D process.
    """

    @property
    def sample_size(self) -> int:
        """The sample size T of the sigma processes."""

    @property
    def n_procs(self) -> int:
        """The number of sigma processes."""

    def get_sigma(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2D array, each column is a 1D process,
        the number of rows define the sample size.,
        random_state not None to reproduce results.
        """


# ---------------------------------------------------
# Constant sigma


class ConstantSigma(ConstantProcs):
    """
    Implements the Sigma Protocol.
    Constant sigma processes matrix, each column of the drift
    matrix is a 1D process.
    - T is the sample size (rows of the matrix)
    - sigma_constants is either the constant for all processes
        (non random) or a tuple of constants one for each process
    - n_procs is the number of process (columns in the matix),
        if sigma_constants is tuple, this argument is ignored,
        if sigma_constants is a float then n_procs cannot be None,
        otherwise will raise ValueError
    The returned matrix columns are indexes in the same order as
    the mu_constants tuple (in case it is a tuple).
    """

    def __init__(
        self,
        T: int,
        sigma_constants: Union[float, tuple[float, ...]],
        n_procs: Optional[int] = None,
    ) -> None:
        super().__init__(T, sigma_constants, n_procs)
        self.sigma_constants = sigma_constants
        self.T = T
        self._n_procs = n_procs

    @property
    def sample_size(self) -> int:
        """The sample size T of the sigma processes."""
        return self.T

    @property
    def n_procs(self) -> int:
        """The number of sigma processes."""
        return super().n_procs

    def get_sigma(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2D array, each column is a 1D process,
        the number of rows define the sample size,
        random_state not None to reproduce results.
        """
        return super().get_proc(random_state)


# ---------------------------------------------------
# sigma constants from data


def estimate_sigma_constants(proc_mat: np.ndarray) -> tuple[float, ...]:
    """
    Estimate sigma constants from data (geometric Brownian motion paths).
    - proc_mat is a 2D array,  each column is a process
    Returns a tuple of floats indexed with the same order as proc_mat
    columns.
    """
    diffusion_increments = np.diff(proc_mat, axis=0) / proc_mat[:-1, :]
    increment_sigmas = np.std(diffusion_increments, axis=0)
    return tuple([float(sigma) for sigma in increment_sigmas])


# =============================================================================
# Initial values for P

# ---------------------------------------------------
# Protocol (interface)


class InitP(Protocol):
    """
    Base for the vector of initial values for gen geometic
    Brownian motions (P_0).
    """

    @property
    def n_procs(self) -> int:
        """The number of init Ps (P_0's), length of P_0s vector."""

    def get_P_0(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 1D vector, random_state to reproduce
        results in case ther is something random, ignored
        if P_0 generation is non-random.
        """


# ---------------------------------------------------
# Random initial values


class RandomInitP:
    """
    Implements InitP.
    Random choice init P_0s.
    lower_bound is strictly less than upper_bound,
    both bounds are strictly positive.
    """

    def __init__(
        self, lower_bound: float, upper_bound: float, n_procs: int
    ) -> None:
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self._n_procs = n_procs

        self._validate_bounds()

    @property
    def n_procs(self) -> int:
        """The number of init Ps (P_0's), length of P_0s vector."""
        return self._n_procs

    def get_P_0(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 1D vector, random_state to reproduce results.
        """
        rng = np.random.default_rng(random_state)
        # random numbers in [0, 1)
        random_vec = rng.random(self._n_procs)
        # rescale [0, 1) interval to bounds
        return (
            self.upper_bound - self.lower_bound
        ) * random_vec + self.lower_bound

    def _validate_bounds(self) -> Optional[NoReturn]:
        """If price bounds are incorrect raise an exception."""
        if self.lower_bound <= 0 or self.upper_bound <= 0:
            raise ValueError("bounds have to be strictly positive ")
        if self.lower_bound >= self.upper_bound:
            raise ValueError("upper bound has to be larger than lower_bound.")
        return None


# ---------------------------------------------------
# Initial values from data


class DataInitP:
    """
    Implements InitP.
    P_0s from data.
    - P_data should be a 2D numpy array. Each column is a process.
    - last_P is a boolean flag, if true uses last sampled value from P_mat
        (last row), else, uses first value (first row).
    """

    def __init__(self, P_data: np.ndarray, last_P: bool = True) -> None:
        self.P_data = P_data
        self.last_P = last_P

    @property
    def n_procs(self) -> int:
        """The number of init Ps (P_0's), length of P_0s vector."""
        return self.P_data.shape[1]

    def get_P_0(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 1D vector, random_state to reproduce
        results in case ther is something random, ignored
        if P_0 generation is non-random.
        """
        if self.last_P:
            row_idx = -1
        else:
            row_idx = 0
        return self.P_data[row_idx, :]


# =============================================================================
# generalized geometric Brownian motion


class GenGeoBrownian:
    """
    Generalized Geometric Brownian motion simulator.
    Generates a matrix (2D numpy array) of processes,
    each column is a process.
    - drift is an concrete implementation of the Drift protocol
    - sigma is a concrete implementation of the Sigma protocol
    - init_P is a concrete implementation of the InitP protocol
    - rho is the correlation coefficient for the gen geo Brownian
        motion processes matrix,
    """

    def __init__(
        self,
        drift: Drift,
        sigma: Sigma,
        init_P: InitP,
        rho: Optional[float] = None,
    ) -> None:
        self.drift = drift
        self.sigma = sigma
        self.init_P = init_P
        self.rho = rho

        self._validate_drift_sigma_init_P()
        self.T, self.n_procs = self.drift.sample_size, self.drift.n_procs

    def get_P(self, random_state: Optional[int] = None) -> np.ndarray:
        """
        Returns a 2D array, each column is a process,
        random_state not None to reproduce results.
        """
        sigmas = self.sigma.get_sigma(random_state)
        time_integrals = self._get_time_integrals(sigmas, random_state)
        W_integrals = self._get_W_integrals(sigmas, random_state)
        P_0s = self.init_P.get_P_0(random_state)
        return P_0s[None, :] * np.exp(time_integrals + W_integrals)

    def _get_time_integrals(
        self, sigmas: np.ndarray, random_state: Optional[int]
    ) -> np.ndarray:
        """The integral with respect to time."""
        mus = self.drift.get_mu(random_state)
        integrals = np.cumsum(mus - sigmas ** 2 / 2, axis=0)
        return np.insert(integrals, 0, np.zeros(mus.shape[1]), axis=0)[:-1]

    def _get_W_integrals(
        self, sigmas: np.ndarray, random_state: Optional[int]
    ) -> np.ndarray:
        """Integral with respect to the Brownian motion (W)."""
        dWs = brownian_motion.get_corr_dW_matrix(
            self.T, self.n_procs, self.rho, random_state
        )
        integrals = np.cumsum(sigmas * dWs, axis=0)
        return np.insert(integrals, 0, np.zeros(dWs.shape[1]), axis=0)[:-1]

    def _validate_drift_sigma_init_P(self) -> Optional[NoReturn]:
        if (
            self.drift.n_procs != self.sigma.n_procs
            or self.drift.n_procs != self.init_P.n_procs
        ):
            raise ValueError(
                "n_procs for both drift, sigma and init_P has to be the same!"
            )
        elif self.drift.sample_size != self.sigma.sample_size:
            raise ValueError(
                "sample size T for both drift and sigma has to be the same!"
            )
        return None


# ---------------------------------------------------
# estimate gen geo Brownian correlation


def estimate_gBrownian_correlation(proc_mat: np.ndarray) -> float:
    """
    Estimate geometric Brownian motion mean correlation from data.
    - proc_mat is a 2D array,  each column is a process
    Returns a float, the correlation coefficient.
    """
    diffusion_increments = np.diff(proc_mat, axis=0) / proc_mat[:-1, :]
    corr_mat = np.corrcoef(diffusion_increments, rowvar=False)
    # put nan in correlation matrix diagonal to exclude it
    # when taking the mean (nanmean)
    np.fill_diagonal(corr_mat, np.nan)
    return float(np.nanmean(corr_mat))