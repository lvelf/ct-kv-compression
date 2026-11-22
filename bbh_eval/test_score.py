"""Run from the repository root: python3 -m unittest bbh_eval.test_score -v"""

import unittest

from bbh_eval.score import score_answer


class ExactMatchTests(unittest.TestCase):
    def test_boolean_correct_with_outer_whitespace(self):
        result = score_answer("boolean_expressions", " True\n", "True", "True and True is")
        self.assertTrue(result["exact_match"])

    def test_wrong_but_valid_boolean(self):
        result = score_answer("boolean_expressions", "False", "True", "True and True is")
        self.assertTrue(result["valid_format"])
        self.assertFalse(result["exact_match"])

    def test_boolean_ambiguous_lowercase_empty_or_prose_is_invalid(self):
        for answer in ("", "TRUE", "True or False", "True, because both are true."):
            with self.subTest(answer=answer):
                result = score_answer("boolean_expressions", answer, "True", "True and True is")
                self.assertFalse(result["valid_format"])
                self.assertFalse(result["exact_match"])

    def test_date_uses_offered_label_including_parentheses(self):
        question = "Which date is earlier?\nOptions:\n(A) 03/02/2020\n(B) 03/01/2020"
        self.assertTrue(score_answer("date_understanding", " (B) ", "(B)", question)["exact_match"])
        self.assertTrue(score_answer("date_understanding", "(A)", "(B)", question)["valid_format"])
        for answer in ("B", "(C)", "03/01/2020", "(A) or (B)"):
            with self.subTest(answer=answer):
                self.assertFalse(score_answer("date_understanding", answer, "(B)", question)["valid_format"])

    def test_bad_gold_is_a_data_error(self):
        with self.assertRaises(ValueError):
            score_answer("date_understanding", "(A)", "(C)", "(A) First\n(B) Second")


if __name__ == "__main__":
    unittest.main()
