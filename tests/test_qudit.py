import unittest
from math import comb, factorial

import numpy as np
from qutip import coherent, ket2dm

from discrete_simulation import DiscreteSimulation
from qudit import (
    PhotonDistributionQudit,
    Qudit,
    photon_number_distribution_from_state,
)


class PhotonDistributionQuditTests(unittest.TestCase):
    def test_fixed_fock_state_has_binomial_loss_distribution(self):
        n0 = 5
        gamma_loss = 0.2
        photon_distribution = np.zeros(9)
        photon_distribution[n0] = 1
        qudit = PhotonDistributionQudit(9, photon_distribution)

        expected = np.array([
            comb(n0, loss) * gamma_loss**loss * (1 - gamma_loss) ** (n0 - loss)
            if loss <= n0 else 0
            for loss in range(9)
        ])
        actual = np.array([qudit.p_loss(gamma_loss, loss) for loss in range(9)])

        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_coherent_state_has_thinned_poisson_loss_distribution(self):
        mean_photons = 4
        gamma_loss = 0.15
        state = coherent(60, np.sqrt(mean_photons))
        photon_distribution = photon_number_distribution_from_state(state)
        qudit = PhotonDistributionQudit(60, photon_distribution)

        loss_counts = np.arange(60)
        thinned_mean = gamma_loss * mean_photons
        expected = np.exp(-thinned_mean) * thinned_mean**loss_counts / np.array(
            [factorial(int(loss)) for loss in loss_counts], dtype=float
        )
        actual = np.array([qudit.p_loss(gamma_loss, loss) for loss in loss_counts])

        np.testing.assert_allclose(actual, expected, atol=1e-13)
        self.assertTrue(np.all(actual >= 0))
        self.assertAlmostEqual(actual.sum(), 1, places=13)

    def test_materially_truncated_distribution_warns_and_is_normalized(self):
        with self.assertWarnsRegex(RuntimeWarning, "total weight"):
            qudit = PhotonDistributionQudit(2, [0.3, 0.2])

        probabilities = [qudit.p_loss(0.25, loss) for loss in range(2)]
        self.assertAlmostEqual(sum(probabilities), 1)

    def test_invalid_distribution_and_loss_probability_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            PhotonDistributionQudit(2, [1.1, -0.1])
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            PhotonDistributionQudit(2, [[1.0, 0.0]])

        qudit = PhotonDistributionQudit(2, [1.0, 0.0])
        for gamma_loss in (-0.1, 1.1, np.inf):
            with self.subTest(gamma_loss=gamma_loss):
                with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                    qudit.p_loss(gamma_loss, 0)

    def test_loss_probability_edges_and_modulo_aggregation(self):
        photon_distribution = np.zeros(6)
        photon_distribution[5] = 1
        qudit = PhotonDistributionQudit(3, photon_distribution)

        np.testing.assert_array_equal(
            [qudit.p_loss(0, loss) for loss in range(3)],
            [1, 0, 0],
        )
        np.testing.assert_array_equal(
            [qudit.p_loss(1, loss) for loss in range(3)],
            [0, 0, 1],
        )

        gamma_loss = 0.4
        physical = np.array([
            comb(5, loss) * gamma_loss**loss * (1 - gamma_loss) ** (5 - loss)
            for loss in range(6)
        ])
        expected = [
            physical[0] + physical[3],
            physical[1] + physical[4],
            physical[2] + physical[5],
        ]
        actual = [qudit.p_loss(gamma_loss, loss) for loss in range(3)]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertAlmostEqual(sum(actual), 1)

    def test_state_helper_supports_kets_and_density_matrices(self):
        state = coherent(20, 1.5)
        from_ket = photon_number_distribution_from_state(state)
        from_density_matrix = photon_number_distribution_from_state(ket2dm(state))

        np.testing.assert_allclose(from_ket, from_density_matrix, atol=1e-15)
        self.assertEqual(from_ket.ndim, 1)

    def test_input_is_copied_and_original_qudit_is_unchanged(self):
        supplied = np.array([1.0, 0.0])
        improved = PhotonDistributionQudit(2, supplied)
        supplied[:] = [0.0, 1.0]
        np.testing.assert_array_equal(improved.photon_number_distribution, [1.0, 0.0])

        gamma_loss = 0.03
        alpha = 2.5
        loss = 1
        original = Qudit(8)
        expected = (
            gamma_loss**loss
            / factorial(loss)
            * (1 - gamma_loss) ** (alpha**2 / 2)
            * alpha ** (2 * loss)
        )
        self.assertEqual(original.p_loss(gamma_loss, loss, alpha), expected)

        simulation = DiscreteSimulation(8, 8, 4)
        self.assertIs(type(simulation.enQudit.quditA), Qudit)


if __name__ == "__main__":
    unittest.main()
