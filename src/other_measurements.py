"""Additional phase-measurement models.

This module intentionally lives separately from :mod:`measurements` so that the
existing canonical and heterodyne implementations remain unchanged.  The
adaptive-homodyne phase matrix follows Appendix C of arXiv:2108.01009 and
describes the ideal Mark-II scheme with instantaneous local-oscillator feedback.
"""

import multiprocessing as mp_process
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mpmath as mp
import numpy as np
import qutip as qt
from scipy.special import gammaln

from measurements import POVM


_CACHE_VERSION = 2
_AUTOMATIC_CUTOFF = -1
_MAX_AUTOMATIC_REFINEMENTS = 100


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return int(value)


def _phase_matrix_diagnostics(phase_matrix: np.ndarray) -> Dict[str, float]:
    hermitian_part = (phase_matrix + phase_matrix.conjugate().T) / 2
    return {
        "max_diagonal_error": float(
            np.max(np.abs(np.diag(phase_matrix) - 1.0), initial=0.0)
        ),
        "hermiticity_error": float(
            np.max(
                np.abs(phase_matrix - phase_matrix.conjugate().T), initial=0.0
            )
        ),
        "min_eigenvalue": float(np.min(np.linalg.eigvalsh(hermitian_part))),
    }


def _validate_phase_matrix(
    phase_matrix: np.ndarray,
    dimension: int,
    tolerance: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    matrix = np.asarray(phase_matrix, dtype=complex)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            "phase_matrix must have shape "
            f"({dimension}, {dimension}), not {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("phase_matrix must contain only finite values")

    diagnostics = _phase_matrix_diagnostics(matrix)
    if diagnostics["hermiticity_error"] > tolerance:
        raise ValueError(
            "phase_matrix is not Hermitian within the requested tolerance: "
            f"error={diagnostics['hermiticity_error']:.3e}"
        )
    if diagnostics["max_diagonal_error"] > tolerance:
        raise ValueError(
            "phase_matrix does not have a unit diagonal within the requested "
            f"tolerance: error={diagnostics['max_diagonal_error']:.3e}"
        )
    if diagnostics["min_eigenvalue"] < -tolerance:
        raise ValueError(
            "phase_matrix is not positive semidefinite within the requested "
            f"tolerance: minimum eigenvalue={diagnostics['min_eigenvalue']:.3e}"
        )
    return matrix, diagnostics


class CovariantPhaseMeasurement(POVM):
    r"""A discretized covariant phase POVM specified by its phase matrix.

    The continuous POVM density is

    .. math::

        F(\phi)=\frac{1}{2\pi}\sum_{m,n}H_{mn}
        e^{i\phi(m-n)}|m\rangle\langle n|.

    Each discrete POVM element is the analytic integral of this density over
    the phase interval centered on the corresponding result.  A valid phase
    matrix is Hermitian and positive semidefinite and has ``H[n, n] = 1``.

    Parameters
    ----------
    phase_matrix:
        The phase matrix ``H`` in the Fock basis.
    result_resolution:
        Number of equal phase intervals spanning a full turn.
    num:
        Number operator defining the truncated Fock space.
    validation_tolerance:
        Absolute tolerance used for physicality and completeness checks.
    """

    def __init__(
        self,
        phase_matrix: np.ndarray,
        result_resolution: int,
        num: qt.Qobj,
        *,
        validation_tolerance: float = 1e-10,
    ):
        self.result_resolution = _positive_integer(
            result_resolution, "result_resolution"
        )
        if not np.isfinite(validation_tolerance) or validation_tolerance <= 0:
            raise ValueError("validation_tolerance must be finite and positive")
        if num.shape[0] != num.shape[1]:
            raise ValueError("num must be a square operator")

        self.num = num
        self.validation_tolerance = float(validation_tolerance)
        self.phase_matrix, self.phase_matrix_diagnostics = _validate_phase_matrix(
            phase_matrix, num.shape[0], self.validation_tolerance
        )
        super().__init__()
        self.completeness_error = self._validate_completeness()

    def get_possible_results(self) -> List[Union[Tuple[float, ...], float]]:
        return [
            2 * np.pi * index / self.result_resolution
            for index in range(self.result_resolution)
        ]

    def _operator_for_result(self, x: float) -> qt.Qobj:
        dimension = self.num.shape[0]
        row = np.arange(dimension)[:, np.newaxis]
        column = np.arange(dimension)[np.newaxis, :]
        number_difference = row - column

        # np.sinc(z) = sin(pi*z)/(pi*z), including the continuous value at z=0.
        # This is exactly (1/2pi) times the integral over the result interval.
        interval_kernel = (
            np.exp(1j * number_difference * x)
            * np.sinc(number_difference / self.result_resolution)
            / self.result_resolution
        )
        return qt.Qobj(self.phase_matrix * interval_kernel, dims=self.num.dims)

    def _validate_completeness(self) -> float:
        total = np.zeros(self.num.shape, dtype=complex)
        for element in self.POVM_elements.values():
            total += element.full()
        error = float(np.max(np.abs(total - np.eye(self.num.shape[0]))))
        roundoff_allowance = (
            100
            * np.finfo(float).eps
            * self.num.shape[0]
            * self.result_resolution
        )
        if error > max(self.validation_tolerance, roundoff_allowance):
            raise ValueError(
                "discretized phase POVM is not complete within the requested "
                f"tolerance: error={error:.3e}"
            )
        return error

    def effect_for_result(self, x: float) -> qt.Qobj:
        raise NotImplementedError()


def _moment_table(maximum_index: int) -> np.ndarray:
    """Return the recursively defined moment table in Appendix C."""
    moments = np.zeros(
        (maximum_index + 1, maximum_index + 1), dtype=np.longdouble
    )
    moments[0, 0] = 1.0

    for index in range(1, maximum_index + 1):
        moments[index, 0] = moments[index - 1, 0] / (2 * index + 1)
        moments[0, index] = moments[index, 0]

    for n in range(1, maximum_index + 1):
        for m in range(1, maximum_index + 1):
            denominator = 2 * (n - m) ** 2 + n + m
            moments[n, m] = (
                n * moments[n - 1, m] + m * moments[n, m - 1]
            ) / denominator
    return moments


def _gamma_coefficients(number: int) -> np.ndarray:
    factorial = np.longdouble(1.0)
    for factor in range(2, number + 1):
        factorial *= factor

    coefficients = np.empty(number // 2 + 1, dtype=np.longdouble)
    if np.isfinite(factorial):
        coefficients[0] = 1 / np.sqrt(factorial)
    else:
        # This fallback matters on platforms where longdouble has no wider
        # exponent range than float64.  The target N=220 environment uses the
        # finite-factorial branch, which retains additional precision.
        coefficients[0] = np.exp(
            np.longdouble(-0.5 * gammaln(number + 1))
        )
    for p in range(number // 2):
        coefficients[p + 1] = coefficients[p] * np.longdouble(
            (number - 2 * p) * (number - 2 * p - 1)
        ) / np.longdouble(2 * (p + 1))
    return coefficients


def _generalized_binomial_sequence(alpha: float, cutoff: int) -> np.ndarray:
    coefficients = np.empty(cutoff + 1, dtype=np.longdouble)
    coefficients[0] = 1.0
    for index in range(1, cutoff + 1):
        coefficients[index] = (
            coefficients[index - 1]
            * np.longdouble(alpha - index + 1)
            / np.longdouble(index)
        )
    return coefficients


def _adaptive_homodyne_element(
    m: int,
    n: int,
    moments: np.ndarray,
    gammas: List[np.ndarray],
    binomials: Dict[int, np.ndarray],
) -> float:
    r"""Evaluate one element using the two convolutions ``u^T M v``."""
    difference = n - m
    left = np.convolve(gammas[m], binomials[difference])
    right = np.convolve(gammas[n], binomials[-difference])
    moment_block = moments[: left.size, : right.size]
    products = left * np.dot(moment_block, right)
    value = np.sum(products, dtype=np.longdouble)
    if not np.isfinite(value):
        raise FloatingPointError(
            f"non-finite adaptive-homodyne matrix element H[{m}, {n}]"
        )
    return float(value)


def _series_data(
    dimension: int, cutoff: int, convergence_increment: int
) -> Tuple[np.ndarray, List[np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    maximum_moment_index = (
        (dimension - 1) // 2 + cutoff + convergence_increment
    )
    moments = _moment_table(maximum_moment_index)
    gammas = [_gamma_coefficients(number) for number in range(dimension)]
    binomials = {
        difference: _generalized_binomial_sequence(difference / 2, cutoff)
        for difference in range(-(dimension - 1), dimension)
    }
    extended_binomials = {
        difference: _generalized_binomial_sequence(
            difference / 2, cutoff + convergence_increment
        )
        for difference in range(-(dimension - 1), dimension)
    }
    return moments, gammas, binomials, extended_binomials


def _convergence_pairs(dimension: int) -> List[Tuple[int, int]]:
    if dimension < 2:
        return []
    middle = dimension // 2
    candidates = {
        (0, 1),
        (0, dimension - 1),
        (1, dimension - 1),
        (max(0, middle - 1), dimension - 1),
        (dimension - 2, dimension - 1),
        (middle, min(middle + 1, dimension - 1)),
    }
    return sorted((m, n) for m, n in candidates if m < n)


def _convergence_error(
    dimension: int,
    moments: np.ndarray,
    gammas: List[np.ndarray],
    binomials: Dict[int, np.ndarray],
    extended_binomials: Dict[int, np.ndarray],
) -> Tuple[float, Optional[Tuple[int, int]]]:
    maximum_error = 0.0
    worst_pair = None
    for m, n in _convergence_pairs(dimension):
        truncated = _adaptive_homodyne_element(
            m, n, moments, gammas, binomials
        )
        extended = _adaptive_homodyne_element(
            m, n, moments, gammas, extended_binomials
        )
        error = abs(truncated - extended)
        if error > maximum_error:
            maximum_error = error
            worst_pair = (m, n)
    return maximum_error, worst_pair


def _compute_adaptive_homodyne_phase_matrix(
    dimension: int,
    series_cutoff: Optional[int],
    convergence_increment: int,
    convergence_tolerance: float,
) -> Tuple[np.ndarray, int, float]:
    automatic_cutoff = series_cutoff is None
    cutoff = dimension - 1 if automatic_cutoff else series_cutoff
    physicality_refinements = 0

    for refinement in range(_MAX_AUTOMATIC_REFINEMENTS + 1):
        moments, gammas, binomials, extended_binomials = _series_data(
            dimension, cutoff, convergence_increment
        )
        convergence_error, worst_pair = _convergence_error(
            dimension, moments, gammas, binomials, extended_binomials
        )
        if convergence_error > convergence_tolerance:
            if not automatic_cutoff:
                raise RuntimeError(
                    "adaptive-homodyne series did not converge at "
                    f"series_cutoff={cutoff}: the change at H{worst_pair} is "
                    f"{convergence_error:.3e}, exceeding "
                    f"convergence_tolerance={convergence_tolerance:.3e}"
                )
        else:
            phase_matrix = np.eye(dimension, dtype=float)
            for m in range(dimension):
                for n in range(m + 1, dimension):
                    value = _adaptive_homodyne_element(
                        m, n, moments, gammas, binomials
                    )
                    phase_matrix[m, n] = value
                    phase_matrix[n, m] = value

            # H_nn = 1 follows analytically from POVM normalization.  Setting it
            # exactly avoids roundoff from an unnecessary series evaluation.
            try:
                _validate_phase_matrix(
                    phase_matrix, dimension, convergence_tolerance
                )
            except ValueError as error:
                if automatic_cutoff and physicality_refinements < 1:
                    physicality_refinements += 1
                else:
                    raise FloatingPointError(
                        "the adaptive-homodyne series converged in cutoff but "
                        "its evaluation did not produce a physical phase "
                        "matrix after an additional cutoff refinement. The "
                        "Appendix-C Maclaurin sums are numerically "
                        "ill-conditioned at this dimension; supply a "
                        "phase_matrix or cache computed with higher precision"
                    ) from error
            else:
                return phase_matrix, cutoff, convergence_error

        if refinement == _MAX_AUTOMATIC_REFINEMENTS:
            raise RuntimeError(
                "adaptive-homodyne series did not converge to a physical phase "
                f"matrix after {_MAX_AUTOMATIC_REFINEMENTS} automatic refinements"
            )
        cutoff += convergence_increment

    raise AssertionError("unreachable adaptive-homodyne cutoff loop")


_MP_MOMENTS = None
_MP_GAMMAS = None
_MP_BINOMIALS = None


def _mp_moment_table(maximum_index: int):
    moments = [
        [mp.mpf(0) for _ in range(maximum_index + 1)]
        for _ in range(maximum_index + 1)
    ]
    moments[0][0] = mp.mpf(1)
    for index in range(1, maximum_index + 1):
        moments[index][0] = moments[index - 1][0] / (2 * index + 1)
        moments[0][index] = moments[index][0]
    for n in range(1, maximum_index + 1):
        for m in range(1, maximum_index + 1):
            denominator = 2 * (n - m) ** 2 + n + m
            moments[n][m] = (
                n * moments[n - 1][m] + m * moments[n][m - 1]
            ) / denominator
    return moments


def _mp_gamma_coefficients(number: int):
    coefficients = [1 / mp.sqrt(mp.factorial(number))]
    for p in range(number // 2):
        coefficients.append(
            coefficients[-1]
            * (number - 2 * p)
            * (number - 2 * p - 1)
            / (2 * (p + 1))
        )
    return coefficients


def _mp_generalized_binomial_sequence(difference: int, cutoff: int):
    alpha = mp.mpf(difference) / 2
    coefficients = [mp.mpf(1)]
    for index in range(1, cutoff + 1):
        coefficient = coefficients[-1] * (alpha - index + 1) / index
        if coefficient == 0:
            break
        coefficients.append(coefficient)
    return coefficients


def _mp_convolve(left, right):
    result = [mp.mpf(0)] * (len(left) + len(right) - 1)
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            result[left_index + right_index] += left_value * right_value
    return result


def _initialize_mpmath_worker(
    dimension: int,
    cutoff: int,
    decimal_precision: int,
) -> None:
    global _MP_MOMENTS, _MP_GAMMAS, _MP_BINOMIALS
    mp.mp.dps = decimal_precision
    maximum_moment_index = (dimension - 1) // 2 + cutoff
    _MP_MOMENTS = _mp_moment_table(maximum_moment_index)
    _MP_GAMMAS = [
        _mp_gamma_coefficients(number) for number in range(dimension)
    ]
    _MP_BINOMIALS = {
        difference: _mp_generalized_binomial_sequence(difference, cutoff)
        for difference in range(-(dimension - 1), dimension)
    }


def _mp_adaptive_homodyne_element(m: int, n: int):
    if _MP_MOMENTS is None or _MP_GAMMAS is None or _MP_BINOMIALS is None:
        raise RuntimeError("mpmath adaptive-homodyne worker is not initialized")
    difference = n - m
    left = _mp_convolve(_MP_GAMMAS[m], _MP_BINOMIALS[difference])
    right = _mp_convolve(_MP_GAMMAS[n], _MP_BINOMIALS[-difference])
    transformed_right = [
        mp.fdot(row[: len(right)], right)
        for row in _MP_MOMENTS[: len(left)]
    ]
    return mp.fdot(left, transformed_right)


def _mp_adaptive_homodyne_worker(
    pair: Tuple[int, int]
) -> Tuple[int, int, float]:
    m, n = pair
    return m, n, float(_mp_adaptive_homodyne_element(m, n))


def _mp_sample_elements(
    dimension: int,
    cutoff: int,
    decimal_precision: int,
) -> Dict[Tuple[int, int], float]:
    _initialize_mpmath_worker(dimension, cutoff, decimal_precision)
    return {
        pair: float(_mp_adaptive_homodyne_element(*pair))
        for pair in _convergence_pairs(dimension)
    }


def _compute_mpmath_adaptive_homodyne_phase_matrix(
    dimension: int,
    series_cutoff: Optional[int],
    convergence_increment: int,
    convergence_tolerance: float,
    decimal_precision: int,
    workers: int,
) -> Tuple[np.ndarray, int, float]:
    automatic_cutoff = series_cutoff is None
    cutoff = dimension - 1 if automatic_cutoff else series_cutoff

    for refinement in range(_MAX_AUTOMATIC_REFINEMENTS + 1):
        truncated = _mp_sample_elements(
            dimension, cutoff, decimal_precision
        )
        extended = _mp_sample_elements(
            dimension,
            cutoff + convergence_increment,
            decimal_precision,
        )
        errors = {
            pair: abs(truncated[pair] - extended[pair])
            for pair in truncated
        }
        worst_pair = max(errors, key=errors.get) if errors else None
        convergence_error = errors[worst_pair] if worst_pair is not None else 0.0
        if convergence_error <= convergence_tolerance:
            break
        if not automatic_cutoff:
            raise RuntimeError(
                "adaptive-homodyne mpmath series did not converge at "
                f"series_cutoff={cutoff}: the change at H{worst_pair} is "
                f"{convergence_error:.3e}, exceeding "
                f"convergence_tolerance={convergence_tolerance:.3e}"
            )
        if refinement == _MAX_AUTOMATIC_REFINEMENTS:
            raise RuntimeError(
                "adaptive-homodyne mpmath series did not converge after "
                f"{_MAX_AUTOMATIC_REFINEMENTS} automatic refinements"
            )
        cutoff += convergence_increment

    pairs = [
        (m, n)
        for m in range(dimension)
        for n in range(m + 1, dimension)
    ]
    phase_matrix = np.eye(dimension, dtype=float)
    if workers == 1 or not pairs:
        _initialize_mpmath_worker(dimension, cutoff, decimal_precision)
        values = map(_mp_adaptive_homodyne_worker, pairs)
        for m, n, value in values:
            phase_matrix[m, n] = value
            phase_matrix[n, m] = value
    else:
        process_count = min(workers, len(pairs))
        chunksize = max(1, len(pairs) // (16 * process_count))
        with mp_process.Pool(
            processes=process_count,
            initializer=_initialize_mpmath_worker,
            initargs=(dimension, cutoff, decimal_precision),
        ) as pool:
            for m, n, value in pool.imap_unordered(
                _mp_adaptive_homodyne_worker,
                pairs,
                chunksize=chunksize,
            ):
                phase_matrix[m, n] = value
                phase_matrix[n, m] = value

    _validate_phase_matrix(
        phase_matrix, dimension, convergence_tolerance
    )
    return phase_matrix, cutoff, convergence_error


def _cache_scalar(cache: np.lib.npyio.NpzFile, name: str):
    if name not in cache.files:
        raise ValueError(f"adaptive-homodyne cache is missing metadata '{name}'")
    value = cache[name]
    if value.size != 1:
        raise ValueError(f"adaptive-homodyne cache metadata '{name}' is invalid")
    return value.item()


def _load_adaptive_homodyne_cache(
    cache_path: Path,
    dimension: int,
    requested_cutoff: int,
    convergence_increment: int,
    convergence_tolerance: float,
    requested_calculation_method: str,
    mp_dps: int,
) -> np.ndarray:
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            expected = {
                "cache_version": _CACHE_VERSION,
                "dimension": dimension,
                "requested_series_cutoff": requested_cutoff,
                "convergence_increment": convergence_increment,
            }
            for name, expected_value in expected.items():
                actual = _cache_scalar(cache, name)
                if actual != expected_value:
                    raise ValueError(
                        f"adaptive-homodyne cache metadata mismatch for {name}: "
                        f"expected {expected_value}, found {actual}"
                    )
            cached_tolerance = float(_cache_scalar(cache, "convergence_tolerance"))
            if cached_tolerance != convergence_tolerance:
                raise ValueError(
                    "adaptive-homodyne cache metadata mismatch for "
                    f"convergence_tolerance: expected {convergence_tolerance}, "
                    f"found {cached_tolerance}"
                )
            effective_cutoff = int(_cache_scalar(cache, "series_cutoff"))
            if effective_cutoff < 0:
                raise ValueError("adaptive-homodyne cache has an invalid series_cutoff")
            if requested_cutoff != _AUTOMATIC_CUTOFF and effective_cutoff != requested_cutoff:
                raise ValueError(
                    "adaptive-homodyne cache effective cutoff does not match the "
                    "requested explicit cutoff"
                )
            calculation_method = str(
                _cache_scalar(cache, "calculation_method")
            )
            if calculation_method not in {"numpy", "mpmath"}:
                raise ValueError(
                    "adaptive-homodyne cache has an invalid calculation_method"
                )
            cached_requested_method = str(
                _cache_scalar(cache, "requested_calculation_method")
            )
            if cached_requested_method not in {"auto", "numpy", "mpmath"}:
                raise ValueError(
                    "adaptive-homodyne cache has an invalid "
                    "requested_calculation_method"
                )
            requested_backend = requested_calculation_method
            if requested_backend == "auto":
                requested_backend = "numpy" if dimension <= 80 else "mpmath"
            if calculation_method != requested_backend:
                raise ValueError(
                    "adaptive-homodyne cache calculation method does not "
                    "match the requested backend"
                )
            cached_mp_dps = int(_cache_scalar(cache, "mp_dps"))
            if calculation_method == "mpmath" and cached_mp_dps != mp_dps:
                raise ValueError(
                    "adaptive-homodyne cache metadata mismatch for mp_dps: "
                    f"expected {mp_dps}, found {cached_mp_dps}"
                )
            phase_matrix = np.array(
                cache["phase_matrix"], dtype=complex, copy=True
            )
            stored_diagnostics = {
                name: float(_cache_scalar(cache, name))
                for name in (
                    "max_diagonal_error",
                    "hermiticity_error",
                    "min_eigenvalue",
                )
            }
    except (OSError, KeyError, ValueError) as error:
        raise ValueError(
            f"could not use adaptive-homodyne cache '{cache_path}': {error}"
        ) from error

    phase_matrix, diagnostics = _validate_phase_matrix(
        phase_matrix, dimension, convergence_tolerance
    )
    if np.max(np.abs(phase_matrix.imag)) > convergence_tolerance:
        raise ValueError("adaptive-homodyne cache contains a complex phase matrix")
    for name, stored_value in stored_diagnostics.items():
        if not np.isclose(
            diagnostics[name], stored_value, rtol=1e-12, atol=1e-14
        ):
            raise ValueError(
                f"adaptive-homodyne cache diagnostic '{name}' does not match "
                "the cached phase matrix"
            )
    return phase_matrix.real


def _write_adaptive_homodyne_cache(
    cache_path: Path,
    phase_matrix: np.ndarray,
    dimension: int,
    requested_cutoff: int,
    effective_cutoff: int,
    convergence_increment: int,
    convergence_tolerance: float,
    convergence_error: float,
    requested_calculation_method: str,
    calculation_method: str,
    mp_dps: int,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics = _phase_matrix_diagnostics(phase_matrix)
    with cache_path.open("wb") as cache_file:
        np.savez_compressed(
            cache_file,
            phase_matrix=phase_matrix,
            cache_version=_CACHE_VERSION,
            dimension=dimension,
            requested_series_cutoff=requested_cutoff,
            series_cutoff=effective_cutoff,
            convergence_increment=convergence_increment,
            convergence_tolerance=convergence_tolerance,
            convergence_error=convergence_error,
            requested_calculation_method=requested_calculation_method,
            calculation_method=calculation_method,
            mp_dps=mp_dps,
            **diagnostics,
        )


def adaptive_homodyne_phase_matrix(
    dimension: int,
    *,
    series_cutoff: Optional[int] = None,
    convergence_increment: int = 10,
    convergence_tolerance: float = 1e-10,
    cache_path: Optional[Union[str, Path]] = None,
    calculation_method: str = "auto",
    mp_dps: int = 50,
    mp_workers: int = 1,
) -> np.ndarray:
    r"""Construct the ideal Mark-II adaptive-homodyne phase matrix.

    The Appendix-C double series is reduced to ``u.T @ M @ v`` using two
    one-dimensional convolutions.  NumPy's extended floating-point precision
    is used for dimensions up to 80.
    Larger matrices use ``mpmath`` because the Maclaurin sums become
    ill-conditioned in fixed precision.  This selection can be overridden with
    ``calculation_method="numpy"`` or ``calculation_method="mpmath"``.  With
    automatic truncation, the initial global cutoff is ``dimension - 1`` (219
    for a 220-dimensional Fock space) and is increased, if needed, until
    representative low-, high-, and off-diagonal entries change by no more than
    ``convergence_tolerance`` when extended by ``convergence_increment``.

    A cache is read or written only when ``cache_path`` is supplied.  Existing
    caches are accepted only when their parameters and numerical diagnostics
    match the request.
    """
    dimension = _positive_integer(dimension, "dimension")
    convergence_increment = _positive_integer(
        convergence_increment, "convergence_increment"
    )
    if series_cutoff is not None:
        series_cutoff = _nonnegative_integer(series_cutoff, "series_cutoff")
    if not np.isfinite(convergence_tolerance) or convergence_tolerance <= 0:
        raise ValueError("convergence_tolerance must be finite and positive")
    convergence_tolerance = float(convergence_tolerance)
    if calculation_method not in {"auto", "numpy", "mpmath"}:
        raise ValueError(
            "calculation_method must be 'auto', 'numpy', or 'mpmath'"
        )
    mp_dps = _positive_integer(mp_dps, "mp_dps")
    if mp_dps < 20:
        raise ValueError("mp_dps must be at least 20")
    mp_workers = _positive_integer(mp_workers, "mp_workers")
    requested_cutoff = (
        _AUTOMATIC_CUTOFF if series_cutoff is None else series_cutoff
    )

    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.exists():
        return _load_adaptive_homodyne_cache(
            path,
            dimension,
            requested_cutoff,
            convergence_increment,
            convergence_tolerance,
            calculation_method,
            mp_dps,
        )

    resolved_method = calculation_method
    if resolved_method == "auto":
        resolved_method = "numpy" if dimension <= 80 else "mpmath"
    if resolved_method == "numpy":
        phase_matrix, effective_cutoff, convergence_error = (
            _compute_adaptive_homodyne_phase_matrix(
                dimension,
                series_cutoff,
                convergence_increment,
                convergence_tolerance,
            )
        )
    else:
        phase_matrix, effective_cutoff, convergence_error = (
            _compute_mpmath_adaptive_homodyne_phase_matrix(
                dimension,
                series_cutoff,
                convergence_increment,
                convergence_tolerance,
                mp_dps,
                mp_workers,
            )
        )
    if path is not None:
        _write_adaptive_homodyne_cache(
            path,
            phase_matrix,
            dimension,
            requested_cutoff,
            effective_cutoff,
            convergence_increment,
            convergence_tolerance,
            convergence_error,
            calculation_method,
            resolved_method,
            mp_dps,
        )
    return phase_matrix


class AdaptiveHomodynePhaseMeasurement(CovariantPhaseMeasurement):
    r"""Ideal Mark-II adaptive-homodyne phase measurement.

    This model uses the Appendix-C phase matrix for instantaneous adaptive
    local-oscillator feedback.  It therefore describes the idealized Mark-II
    limit, not finite feedback bandwidth or detector inefficiency.
    """

    def __init__(
        self,
        result_resolution: int,
        num: qt.Qobj,
        *,
        phase_matrix: Optional[np.ndarray] = None,
        series_cutoff: Optional[int] = None,
        convergence_increment: int = 10,
        convergence_tolerance: float = 1e-10,
        cache_path: Optional[Union[str, Path]] = None,
        calculation_method: str = "auto",
        mp_dps: int = 50,
        mp_workers: int = 1,
    ):
        if phase_matrix is None:
            phase_matrix = adaptive_homodyne_phase_matrix(
                num.shape[0],
                series_cutoff=series_cutoff,
                convergence_increment=convergence_increment,
                convergence_tolerance=convergence_tolerance,
                cache_path=cache_path,
                calculation_method=calculation_method,
                mp_dps=mp_dps,
                mp_workers=mp_workers,
            )
        super().__init__(
            phase_matrix,
            result_resolution,
            num,
            validation_tolerance=convergence_tolerance,
        )
