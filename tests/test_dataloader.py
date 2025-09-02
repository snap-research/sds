import unittest
from collections import Counter

from sds.dataloader import ScheduleType


class TestScheduleSampler(unittest.TestCase):

    def setUp(self):
        """Set up common test data."""
        self.counts = {0: 2, 1: 3, 2: 4}
        self.total_count = sum(self.counts.values()) # 9
        self.ratios = {k: v / self.total_count for k, v in self.counts.items()}
        self.shuffle_seed = 42

    def test_random_schedule_properties(self):
        """
        Tests RANDOM schedule for its behavioral properties.
        """
        num_trials = 10
        min_expected_differences = 3

        # Property 1: Reproducibility (Deterministic)
        # Calling with the exact same inputs must always yield the same output.
        res1 = ScheduleType.sample_iter_idx(ScheduleType.RANDOM, 5, self.counts, self.ratios, self.shuffle_seed)
        res2 = ScheduleType.sample_iter_idx(ScheduleType.RANDOM, 5, self.counts, self.ratios, self.shuffle_seed)
        self.assertEqual(res1, res2, "The same inputs should produce the same output.")

        # Property 2: Sensitivity to `num_batches_yielded` (Stochastic)
        # Using a different step should likely change the output. We test this statistically
        # to avoid flaky tests where the output might be the same by chance.
        base_result_for_step = res1
        different_step_results = [
            ScheduleType.sample_iter_idx(ScheduleType.RANDOM, 6 + i, self.counts, self.ratios, self.shuffle_seed)
            for i in range(num_trials)
        ]
        num_differences_step = sum(1 for r in different_step_results if r != base_result_for_step)
        self.assertGreaterEqual(
            num_differences_step,
            min_expected_differences,
            f"Changing num_batches_yielded should produce different results (got {num_differences_step}/{num_trials} differences)"
        )

        # Property 3: Sensitivity to `shuffle_seed` (Stochastic)
        # Using a different seed should likely change the output.
        base_result_for_seed = res1
        different_seed_results = [
            ScheduleType.sample_iter_idx(ScheduleType.RANDOM, 5, self.counts, self.ratios, self.shuffle_seed + 1 + i)
            for i in range(num_trials)
        ]
        num_differences_seed = sum(1 for r in different_seed_results if r != base_result_for_seed)
        self.assertGreaterEqual(
            num_differences_seed,
            min_expected_differences,
            f"Changing the seed should produce different results (got {num_differences_seed}/{num_trials} differences)"
        )

        # Property 4: Output Validity
        # The output must always be one of the keys in the counts dictionary.
        for i in range(20):
             result = ScheduleType.sample_iter_idx(ScheduleType.RANDOM, i, self.counts, self.ratios, self.shuffle_seed)
             self.assertIn(result, self.counts.keys())

    def test_consecutive_schedule(self):
        expected_sequence = [0, 0, 1, 1, 1, 2, 2, 2, 2]
        for i in range(self.total_count):
            result = ScheduleType.sample_iter_idx(ScheduleType.CONSECUTIVE, i, self.counts, self.ratios, self.shuffle_seed)
            self.assertEqual(result, expected_sequence[i])
        self.assertEqual(ScheduleType.sample_iter_idx(ScheduleType.CONSECUTIVE, self.total_count, self.counts, self.ratios, self.shuffle_seed), expected_sequence[0])

    def test_consecutive_interleaved_schedule(self):
        expected_sequence = [0, 1, 2, 0, 1, 2, 1, 2, 2]
        for i in range(self.total_count):
            result = ScheduleType.sample_iter_idx(ScheduleType.CONSECUTIVE_INTERLEAVED, i, self.counts, self.ratios, self.shuffle_seed)
            self.assertEqual(result, expected_sequence[i])
        self.assertEqual(ScheduleType.sample_iter_idx(ScheduleType.CONSECUTIVE_INTERLEAVED, self.total_count, self.counts, self.ratios, self.shuffle_seed), expected_sequence[0])

    def test_fixed_random_order_properties(self):
        seq1 = [ScheduleType.sample_iter_idx(ScheduleType.FIXED_RANDOM_ORDER, i, self.counts, self.ratios, self.shuffle_seed) for i in range(self.total_count)]
        seq2 = [ScheduleType.sample_iter_idx(ScheduleType.FIXED_RANDOM_ORDER, i + self.total_count, self.counts, self.ratios, self.shuffle_seed) for i in range(self.total_count)]
        self.assertEqual(Counter(seq1), self.counts)
        self.assertEqual(seq1, seq2)
        consecutive_order = sorted(seq1)
        self.assertNotEqual(seq1, consecutive_order, "The sequence should be shuffled")

    def test_random_order_properties(self):
        seq1 = [ScheduleType.sample_iter_idx(ScheduleType.RANDOM_ORDER, i, self.counts, self.ratios, self.shuffle_seed) for i in range(self.total_count)]
        seq2 = [ScheduleType.sample_iter_idx(ScheduleType.RANDOM_ORDER, i + self.total_count, self.counts, self.ratios, self.shuffle_seed) for i in range(self.total_count)]
        self.assertEqual(Counter(seq1), self.counts)
        self.assertEqual(Counter(seq2), self.counts)
        self.assertNotEqual(seq1, seq2)
        consecutive_order = sorted(seq1)
        self.assertNotEqual(seq1, consecutive_order, "The sequence should be shuffled")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
