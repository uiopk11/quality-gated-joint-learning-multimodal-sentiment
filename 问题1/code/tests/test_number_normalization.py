from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q1lib.number_normalization import build_comparison_units
from q1lib.quality import ASRWord, match_word_sequences


class NumberNormalizationTests(unittest.TestCase):
    def values(self, words: list[str]) -> list[str]:
        return [item.value for item in build_comparison_units(words)]

    def test_number_words_and_digits_are_equivalent(self) -> None:
        self.assertEqual(self.values(["twelve", "hundred"]), ["num:1200"])
        self.assertEqual(self.values(["1200"]), ["num:1200"])
        self.assertEqual(self.values(["one", "thousand", "two", "hundred"]), ["num:1200"])
        self.assertEqual(self.values(["one", "hundred", "and", "five"]), ["num:105"])

    def test_percent_keeps_its_ordinary_word(self) -> None:
        self.assertEqual(
            self.values(["twenty", "percent"]),
            ["num:20", "percent"],
        )

    def test_different_numbers_do_not_match(self) -> None:
        self.assertNotEqual(self.values(["twelve", "hundred"]), self.values(["1300"]))

    def test_multiword_number_timestamp_is_subdivided(self) -> None:
        asr = [
            ASRWord(0, "we", 0.0, 0.2),
            ASRWord(1, "sold", 0.2, 0.4),
            ASRWord(2, "1200", 0.4, 0.8),
            ASRWord(3, "units", 0.8, 1.0),
        ]
        mapping, mask, starts, ends, metrics = match_word_sequences(
            ["we", "sold", "twelve", "hundred", "units"], asr, minimum_contiguous_matches=2
        )
        self.assertEqual(mask, [1, 1, 1, 1, 1])
        self.assertEqual(mapping[2:4], [2, 2])
        self.assertAlmostEqual(starts[2] or -1, 0.4)
        self.assertAlmostEqual(ends[2] or -1, 0.6)
        self.assertAlmostEqual(starts[3] or -1, 0.6)
        self.assertAlmostEqual(ends[3] or -1, 0.8)
        self.assertEqual(metrics["numeric_equivalence_count"], 1.0)


if __name__ == "__main__":
    unittest.main()
