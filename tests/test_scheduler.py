from __future__ import annotations

import unittest

from disttrain.pipeline.scheduler import PipelineScheduler, theoretical_bubble_ratio


class SchedulerTests(unittest.TestCase):
    def test_gpipe_schedule_length(self) -> None:
        scheduler = PipelineScheduler(
            schedule="gpipe",
            num_micro_batches=4,
            pipeline_depth=3,
            stage_index=1,
        )
        actions = scheduler.build()
        self.assertEqual(len(actions), 8)
        self.assertEqual(sum(1 for a in actions if a.kind == "F"), 4)
        self.assertEqual(sum(1 for a in actions if a.kind == "B"), 4)

    def test_1f1b_contains_interleaving(self) -> None:
        scheduler = PipelineScheduler(
            schedule="1f1b",
            num_micro_batches=4,
            pipeline_depth=3,
            stage_index=1,
        )
        actions = scheduler.build()
        kinds = [a.kind for a in actions]
        self.assertIn("F", kinds)
        self.assertIn("B", kinds)
        self.assertEqual(sum(1 for a in actions if a.kind == "F"), 4)
        self.assertEqual(sum(1 for a in actions if a.kind == "B"), 4)

    def test_bubble_ratio(self) -> None:
        ratio = theoretical_bubble_ratio(pipeline_depth=3, num_micro_batches=8)
        self.assertGreater(ratio, 0.0)
        self.assertLess(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
