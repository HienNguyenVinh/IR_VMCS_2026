"""Unit tests for temporal expression extraction."""

from datetime import date
from nuggetindex.temporal import TempType, extract_temporal_expressions


def test_start_expressions():
    text = "Alice Smith was appointed CEO of Acme Robotics in January 2015."
    exprs = extract_temporal_expressions(text)
    assert any(e.expr_type == TempType.START and e.date_val == date(2015, 1, 1) for e in exprs)

    text2 = "Elena Rostova is a board member since May 2018."
    exprs2 = extract_temporal_expressions(text2)
    assert any(e.expr_type == TempType.START and e.date_val == date(2018, 5, 1) for e in exprs2)


def test_end_expressions():
    text = "Alice Smith served until January 2019."
    exprs = extract_temporal_expressions(text)
    assert any(e.expr_type == TempType.END and e.date_val == date(2019, 1, 1) for e in exprs)

    text2 = "Bob Johnson stepped down in January 2023."
    exprs2 = extract_temporal_expressions(text2)
    assert any(e.expr_type == TempType.END and e.date_val == date(2023, 1, 1) for e in exprs2)


def test_point_expressions():
    text = "The treaty was signed on 2021-08-15 in Geneva."
    exprs = extract_temporal_expressions(text)
    assert any(e.expr_type == TempType.POINT and e.date_val == date(2021, 8, 15) for e in exprs)


def test_range_expressions():
    text = "The program ran from 2016 to 2020."
    exprs = extract_temporal_expressions(text)
    assert any(e.expr_type == TempType.START and e.date_val == date(2016, 1, 1) for e in exprs)
    assert any(e.expr_type == TempType.END and e.date_val == date(2020, 1, 1) for e in exprs)

