from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Sequence


_SMALL = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}
_SCALES = {
    "hundred": 100,
    "hundreds": 100,
    "thousand": 1_000,
    "thousands": 1_000,
    "million": 1_000_000,
    "millions": 1_000_000,
    "billion": 1_000_000_000,
    "billions": 1_000_000_000,
}
_NUMBER_WORDS = set(_SMALL) | set(_TENS) | set(_SCALES) | {"and", "point"}
_DIGIT_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class ComparisonUnit:
    value: str
    source_indices: tuple[int, ...]
    surface: str
    is_number: bool


def _canonical_decimal(value: str | int | Decimal) -> str:
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return str(value)
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f").rstrip("0").rstrip(".")


def _parse_integer_words(words: Sequence[str]) -> int | None:
    if not words:
        return None
    total = 0
    current = 0
    saw_value = False
    for word in words:
        if word == "and":
            continue
        if word in _SMALL:
            current += _SMALL[word]
            saw_value = True
        elif word in _TENS:
            current += _TENS[word]
            saw_value = True
        elif word in {"hundred", "hundreds"}:
            current = max(1, current) * 100
            saw_value = True
        elif word in {"thousand", "thousands", "million", "millions", "billion", "billions"}:
            scale = _SCALES[word]
            total += max(1, current) * scale
            current = 0
            saw_value = True
        else:
            return None
    return total + current if saw_value else None


def _parse_number_words(words: Sequence[str]) -> str | None:
    cleaned = [word for word in words if word != "and"]
    if not cleaned:
        return None
    if cleaned.count("point") > 1:
        return None
    if "point" not in cleaned:
        value = _parse_integer_words(cleaned)
        return None if value is None else _canonical_decimal(value)
    point_index = cleaned.index("point")
    integer_words = cleaned[:point_index]
    decimal_words = cleaned[point_index + 1 :]
    integer_value = _parse_integer_words(integer_words) if integer_words else 0
    if integer_value is None or not decimal_words or any(word not in _SMALL for word in decimal_words):
        return None
    decimal_digits = "".join(str(_SMALL[word]) for word in decimal_words)
    return _canonical_decimal(f"{integer_value}.{decimal_digits}")


def build_comparison_units(words: Sequence[str]) -> list[ComparisonUnit]:
    """Build matching-only units without modifying the original transcript tokens."""
    normalized = [str(word).casefold() for word in words]
    units: list[ComparisonUnit] = []
    index = 0
    while index < len(normalized):
        word = normalized[index]
        if _DIGIT_RE.fullmatch(word):
            units.append(
                ComparisonUnit(
                    value=f"num:{_canonical_decimal(word)}",
                    source_indices=(index,),
                    surface=word,
                    is_number=True,
                )
            )
            index += 1
            continue
        if word in _NUMBER_WORDS and word not in {"and", "point"}:
            end = index + 1
            while end < len(normalized):
                candidate = normalized[end]
                if candidate in _NUMBER_WORDS and candidate not in {"and", "point"}:
                    end += 1
                    continue
                if candidate == "point" and end + 1 < len(normalized) and normalized[end + 1] in _SMALL:
                    end += 1
                    continue
                if (
                    candidate == "and"
                    and end + 1 < len(normalized)
                    and normalized[end + 1] in (_NUMBER_WORDS - {"and", "point"})
                ):
                    end += 1
                    continue
                break
            phrase = normalized[index:end]
            parsed = _parse_number_words(phrase)
            if parsed is not None:
                units.append(
                    ComparisonUnit(
                        value=f"num:{parsed}",
                        source_indices=tuple(range(index, end)),
                        surface=" ".join(phrase),
                        is_number=True,
                    )
                )
                index = end
                continue
        units.append(
            ComparisonUnit(value=word, source_indices=(index,), surface=word, is_number=False)
        )
        index += 1
    return units
