import tempfile
import unittest
from pathlib import Path

import numpy as np
import qutip as qt
from scipy.special import gammaln

from measurements import CanonicalPhaseMeasurement, IdealHeterodynePhaseMeasurement
from other_measurements import (
    AdaptiveHomodynePhaseMeasurement,
    CovariantPhaseMeasurement,
    adaptive_homodyne_phase_matrix,
)


class CovariantPhaseMeasurementTests(unittest.TestCase):
    def test_unit_phase_matrix_matches_canonical_measurement(self):
        dimension = 5
        resolution = 7
        num = qt.num(dimension)
        base_state = sum(qt.basis(dimension, n) for n in range(dimension))
        canonical = CanonicalPhaseMeasurement(base_state, resolution, num)
        covariant = CovariantPhaseMeasurement(
            np.ones((dimension, dimension)), resolution, num
        )

        for phase in canonical.get_possible_results():
            np.testing.assert_allclose(
                covariant.POVM_elements[phase].full(),
                canonical.POVM_elements[phase].full(),
                atol=1e-13,
                rtol=1e-13,
            )

    def test_heterodyne_phase_matrix_matches_existing_measurement(self):
        dimension = 6
        resolution = 9
        num = qt.num(dimension)
        row = np.arange(dimension)[:, np.newaxis]
        column = np.arange(dimension)[np.newaxis, :]
        heterodyne_phase_matrix = np.exp(
            gammaln((row + column) / 2 + 1)
            - 0.5 * gammaln(row + 1)
            - 0.5 * gammaln(column + 1)
        )
        existing = IdealHeterodynePhaseMeasurement(resolution, num)
        covariant = CovariantPhaseMeasurement(
            heterodyne_phase_matrix, resolution, num
        )

        for phase in existing.get_possible_results():
            np.testing.assert_allclose(
                covariant.POVM_elements[phase].full(),
                existing.POVM_elements[phase].full(),
                atol=2e-13,
                rtol=2e-13,
            )

    def test_rejects_nonphysical_phase_matrix(self):
        with self.assertRaisesRegex(ValueError, "unit diagonal"):
            CovariantPhaseMeasurement(np.eye(3) * 2, 5, qt.num(3))


class AdaptiveHomodyneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dimension = 6
        cls.phase_matrix = adaptive_homodyne_phase_matrix(
            cls.dimension, series_cutoff=80
        )

    def test_phase_matrix_is_normalized_hermitian_and_positive(self):
        np.testing.assert_allclose(
            self.phase_matrix, self.phase_matrix.conjugate().T, atol=1e-13
        )
        np.testing.assert_allclose(np.diag(self.phase_matrix), 1.0, atol=1e-13)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(self.phase_matrix))), -1e-12
        )

    def test_measurement_is_complete_and_covariant(self):
        resolution = 9
        num = qt.num(self.dimension)
        measurement = AdaptiveHomodynePhaseMeasurement(
            resolution, num, phase_matrix=self.phase_matrix
        )
        total = sum(measurement.POVM_elements.values())
        np.testing.assert_allclose(total.full(), np.eye(self.dimension), atol=1e-12)

        phases = measurement.get_possible_results()
        rotation = (1j * phases[1] * num).expm()
        rotated_zero = (
            rotation
            * measurement.POVM_elements[phases[0]]
            * rotation.dag()
        )
        np.testing.assert_allclose(
            rotated_zero.full(),
            measurement.POVM_elements[phases[1]].full(),
            atol=1e-12,
        )

    def test_explicit_unconverged_cutoff_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "did not converge"):
            adaptive_homodyne_phase_matrix(2, series_cutoff=1)

    def test_automatic_cutoff_produces_a_physical_matrix(self):
        phase_matrix = adaptive_homodyne_phase_matrix(10)
        self.assertGreaterEqual(
            float(np.min(np.linalg.eigvalsh(phase_matrix))), -1e-10
        )

    def test_cache_round_trip_and_metadata_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "adaptive_homodyne.npz"
            calculated = adaptive_homodyne_phase_matrix(
                5, series_cutoff=80, cache_path=cache_path
            )
            cached = adaptive_homodyne_phase_matrix(
                5, series_cutoff=80, cache_path=cache_path
            )
            np.testing.assert_array_equal(cached, calculated)

            with np.load(cache_path, allow_pickle=False) as cache:
                required_metadata = {
                    "phase_matrix",
                    "dimension",
                    "series_cutoff",
                    "convergence_increment",
                    "convergence_tolerance",
                    "max_diagonal_error",
                    "hermiticity_error",
                    "min_eigenvalue",
                }
                self.assertTrue(required_metadata.issubset(cache.files))

            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                adaptive_homodyne_phase_matrix(
                    5,
                    series_cutoff=80,
                    convergence_tolerance=1e-9,
                    cache_path=cache_path,
                )


if __name__ == "__main__":
    unittest.main()
