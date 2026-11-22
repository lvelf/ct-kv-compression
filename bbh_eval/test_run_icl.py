"""Check demo selection without loading a tokenizer, model, or BBH data."""

import contextlib
import io
import unittest

from bbh_eval.prepare_splits import make_task_split
from bbh_eval.run_icl import make_parser, make_prefix, select_run_ids


class ShotSelectionTests(unittest.TestCase):
    def setUp(self):
        self.split = make_task_split(250)["seeds"]["0"]
        self.required_args = ["--task", "date_understanding", "--out", "unit-test"]

    def test_default_uses_the_full_ten_and_all_dev_questions(self):
        args = make_parser().parse_args(self.required_args)
        self.assertEqual(args.shots, 10)
        self.assertEqual(args.model, "openai-community/gpt2-large")
        demos, dev = select_run_ids(self.split, args.shots, args.limit)
        self.assertEqual(demos, self.split["demo_ids"])
        self.assertEqual(dev, self.split["dev_ids"])
        self.assertEqual((len(demos), len(dev)), (10, 40))

    def test_eight_shots_changes_only_demo_prefix(self):
        args = make_parser().parse_args(self.required_args + ["--shots", "8"])
        demos, dev = select_run_ids(self.split, args.shots, args.limit)
        self.assertEqual(demos, self.split["demo_ids"][:8])
        self.assertEqual(dev, self.split["dev_ids"])
        unused = self.split["demo_ids"][8:]
        self.assertTrue(set(unused).isdisjoint(demos + dev))
        rows = [{"input": f"Question {i}?", "target": f"Gold {i}."} for i in range(250)]
        prefix = make_prefix(rows, demos)
        self.assertEqual(prefix.count("Q: "), 8)
        for i in unused + dev:
            self.assertNotIn(rows[i]["target"], prefix)

    def test_unsupported_shot_count_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            make_parser().parse_args(self.required_args + ["--shots", "9"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
