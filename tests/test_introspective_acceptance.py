import unittest

from boltzgen.experimental.introspective_acceptance import (
    acceptance_rate,
    exact_final_distribution,
    l1_distance,
    run_simulation,
    toy_pocket_distributions,
)


class IntrospectiveAcceptanceTest(unittest.TestCase):
    def test_exact_final_distribution_matches_anchor(self) -> None:
        p, q = toy_pocket_distributions()

        final = exact_final_distribution(p, q)

        self.assertLess(l1_distance(final, p), 1e-12)

    def test_acceptance_rate_is_overlap(self) -> None:
        p, q = toy_pocket_distributions()

        expected = sum(min(p[key], q[key]) for key in p)

        self.assertEqual(acceptance_rate(p, q), expected)

    def test_simulation_tracks_anchor_distribution(self) -> None:
        p, q = toy_pocket_distributions()

        result = run_simulation(p, q, n_samples=50_000, seed=3)

        self.assertGreater(result.acceptance_rate, 0.70)
        self.assertLess(result.l1_to_anchor, 0.03)


if __name__ == "__main__":
    unittest.main()
