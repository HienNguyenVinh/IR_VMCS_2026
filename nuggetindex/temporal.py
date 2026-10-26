"""Temporal expression extraction (START, END, POINT) for Algorithm 1.

Paper reference: Section 3.2.1 (Stage 3: Validity Inference).
- Identifies date expressions in text and classifies them as:
  - START: 'since 2019', 'from 2015', 'started in 2018', 'appointed in 2020', 'as of 2017'
  - END: 'until 2023', 'through 2022', 'stepped down in 2021', 'ended in 2020'
  - POINT: standalone date/year references ('in 2020', '2019-05-12', 'in July 2018')
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import List, Optional, Tuple

from .models import parse_iso_date


class TempType(str, Enum):
    START = "START"
    END = "END"
    POINT = "POINT"


@dataclass
class TemporalExpression:
    raw_text: str
    date_val: date
    expr_type: TempType

    def __repr__(self) -> str:
        return f"TemporalExpression({self.raw_text!r}, {self.date_val}, {self.expr_type.value})"


# Regex patterns for dates
MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9, "october": 10,
    "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12
}


def _parse_date_components(year_s: str, month_s: Optional[str] = None, day_s: Optional[str] = None) -> Optional[date]:
    try:
        y = int(year_s)
        m = 1
        if month_s:
            m_norm = month_s.strip().lower()
            if m_norm in MONTHS:
                m = MONTHS[m_norm]
            else:
                m = int(month_s)
        d = 1
        if day_s:
            d = int(day_s)
        return date(y, m, d)
    except Exception:
        return None


def extract_temporal_expressions(text: str) -> List[TemporalExpression]:
    """Extract temporal expressions classified into START, END, or POINT."""
    results: List[TemporalExpression] = []

    # 1. Range pattern: "from <date1> to/until <date2>"
    range_pattern = re.compile(
        r'\bfrom\s+(\d{4}(?:-\d{2}(?:-\d{2})?)?|[a-zA-Z]+\s+\d{4})\s+(?:to|until|till|-)\s+(\d{4}(?:-\d{2}(?:-\d{2})?)?|[a-zA-Z]+\s+\d{4})\b',
        re.IGNORECASE
    )
    for m in range_pattern.finditer(text):
        d1 = parse_iso_date(m.group(1)) or _parse_fuzzy(m.group(1))
        d2 = parse_iso_date(m.group(2)) or _parse_fuzzy(m.group(2))
        if d1:
            results.append(TemporalExpression(m.group(1), d1, TempType.START))
        if d2:
            results.append(TemporalExpression(m.group(2), d2, TempType.END))

    # 2. Start patterns: "since 2019", "appointed CEO ... in January 2015", "became ... in 2020"
    start_pattern = re.compile(
        r'\b(?:since|starting in|began in|beginning in|appointed\s+(?:[\w\s]+?\s+)?in|became\s+[\w\s]+?\s+in|took over(?:\s+[\w\s]+?)?\s+in|as of)\s+([a-zA-Z]+\s+\d{4}|\d{4}(?:-\d{2}(?:-\d{2})?)?)\b',
        re.IGNORECASE
    )
    for m in start_pattern.finditer(text):
        d = parse_iso_date(m.group(1)) or _parse_fuzzy(m.group(1))
        if d:
            results.append(TemporalExpression(m.group(0), d, TempType.START))

    # 3. End patterns: "until 2023", "till 2021", "stepped down in 2021", "left in 2022", "ended in 2020", "through 2022"
    end_pattern = re.compile(
        r'\b(?:until|till|through|stepped down in|left in|ended in|resigned in|served until)\s+([a-zA-Z]+\s+\d{4}|\d{4}(?:-\d{2}(?:-\d{2})?)?)\b',
        re.IGNORECASE
    )
    for m in end_pattern.finditer(text):
        d = parse_iso_date(m.group(1)) or _parse_fuzzy(m.group(1))
        if d:
            results.append(TemporalExpression(m.group(0), d, TempType.END))

    # 4. Point-in-time / standalone dates: "in 2021", "in March 2020", ISO dates
    iso_pattern = re.compile(r'\b(\d{4}-\d{2}-\d{2})\b')
    for m in iso_pattern.finditer(text):
        d = parse_iso_date(m.group(1))
        if d and not any(r.date_val == d for r in results):
            results.append(TemporalExpression(m.group(1), d, TempType.POINT))

    year_pattern = re.compile(r'\b(?:in|during|on|by)?\s*([a-zA-Z]+\s+)?(\d{4})\b')
    for m in year_pattern.finditer(text):
        month_str = m.group(1)
        year_str = m.group(2)
        d = _parse_date_components(year_str, month_str)
        if d and not any(r.date_val == d for r in results):
            # If preceding word was "in" / "during"
            raw = m.group(0).strip()
            results.append(TemporalExpression(raw, d, TempType.POINT))

    return results


def _parse_fuzzy(val: str) -> Optional[date]:
    parts = val.strip().split()
    if len(parts) == 2 and parts[0].isalpha() and parts[1].isdigit():
        return _parse_date_components(parts[1], parts[0])
    return parse_iso_date(val)
