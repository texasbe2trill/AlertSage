"""Tests for the Hunt tab Lucene-style query DSL."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from triage.hunt_query import (
    AndExpr,
    FieldExpr,
    NotExpr,
    OrExpr,
    ParseError,
    RangeExpr,
    Term,
    compile_query,
    evaluate,
    parse,
    parse_duration,
    tokenize,
)


# ---------------------------------------------------------------------------
# Sample rows
# ---------------------------------------------------------------------------


NOW = datetime(2026, 5, 3, 12, 0, 0, tzinfo=timezone.utc)


def _row(**overrides):
    base = {
        "id": 1,
        "incident_text": "User uploaded a file to dropbox.",
        "final_label": "data_exfiltration",
        "max_prob": 0.92,
        "timestamp": "2026-05-03 11:00:00",
        "raw_result": {"mitre_techniques": ["T1567", "T1041"]},
        "_severity": "critical",
        "_status": "new",
        "_anomaly": 88,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_basic_field_value():
    toks = [t[:2] for t in tokenize("label:malware")]
    assert toks == [("WORD", "label"), ("COLON", ":"), ("WORD", "malware"), ("EOF", "")]


def test_tokenize_quoted_with_spaces():
    toks = [t[:2] for t in tokenize('narrative:"data leak"')]
    assert toks == [
        ("WORD", "narrative"),
        ("COLON", ":"),
        ("QUOTED", "data leak"),
        ("EOF", ""),
    ]


def test_tokenize_operators():
    toks = [t[:2] for t in tokenize("confidence:>=0.8")]
    assert toks == [
        ("WORD", "confidence"),
        ("COLON", ":"),
        ("OP", ">="),
        ("WORD", "0.8"),
        ("EOF", ""),
    ]


def test_tokenize_keywords():
    kinds = [t[0] for t in tokenize("label:a AND label:b OR NOT label:c")]
    assert "AND" in kinds and "OR" in kinds and "NOT" in kinds


def test_tokenize_minus_as_not():
    toks = [t[:2] for t in tokenize("-label:benign")]
    assert toks[0] == ("MINUS", "-")


def test_tokenize_unterminated_quote():
    with pytest.raises(ParseError):
        tokenize('narrative:"unterminated')


def test_tokenize_range_brackets():
    kinds = [t[0] for t in tokenize("confidence:[0.5 TO 1.0]")]
    assert "LBRACK" in kinds and "TO" in kinds and "RBRACK" in kinds


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_empty_query_matches_all():
    node = parse("")
    assert isinstance(node, AndExpr) and node.items == ()
    assert evaluate(node, _row(), now=NOW) is True


def test_parse_bare_term_is_narrative_substring():
    node = parse("dropbox")
    assert isinstance(node, Term) and node.text == "dropbox"


def test_parse_field_value():
    node = parse("label:malware")
    assert node == FieldExpr(name="label", op=":", value="malware")


def test_parse_field_alias_canonicalizes():
    node = parse("class:malware")
    assert node == FieldExpr(name="label", op=":", value="malware")


def test_parse_unknown_field_raises():
    with pytest.raises(ParseError):
        parse("nonsense:foo")


def test_parse_implicit_and():
    node = parse("label:malware confidence:>0.8")
    assert isinstance(node, AndExpr)
    assert len(node.items) == 2


def test_parse_explicit_or_and_grouping():
    node = parse("(label:malware OR label:phishing) AND confidence:>=0.8")
    assert isinstance(node, AndExpr)
    assert isinstance(node.items[0], OrExpr)


def test_parse_not_keyword_and_minus_equivalent():
    a = parse("NOT label:benign_activity")
    b = parse("-label:benign_activity")
    assert isinstance(a, NotExpr) and isinstance(b, NotExpr)
    assert a.child == b.child


def test_parse_range():
    node = parse("confidence:[0.5 TO 0.9]")
    assert node == RangeExpr(name="confidence", lo="0.5", hi="0.9")


def test_parse_quoted_value():
    node = parse('narrative:"transferred to dropbox"')
    assert node == FieldExpr(
        name="narrative", op=":", value="transferred to dropbox"
    )


def test_parse_unterminated_paren():
    with pytest.raises(ParseError):
        parse("(label:malware")


def test_parse_trailing_garbage():
    with pytest.raises(ParseError):
        parse("label:malware ) extra")


# ---------------------------------------------------------------------------
# Evaluator: string / substring / list
# ---------------------------------------------------------------------------


def test_eval_label_exact_match():
    assert evaluate(parse("label:data_exfiltration"), _row(), now=NOW) is True
    assert evaluate(parse("label:malware"), _row(), now=NOW) is False


def test_eval_label_alias():
    assert evaluate(parse("class:data_exfiltration"), _row(), now=NOW) is True


def test_eval_narrative_substring_case_insensitive():
    row = _row(incident_text="User uploaded to DROPBOX.")
    assert evaluate(parse("narrative:dropbox"), row, now=NOW) is True


def test_eval_bare_term_matches_narrative():
    assert evaluate(parse("dropbox"), _row(), now=NOW) is True
    assert evaluate(parse("nopenope"), _row(), now=NOW) is False


def test_eval_mitre_membership():
    assert evaluate(parse("mitre:T1041"), _row(), now=NOW) is True
    assert evaluate(parse("mitre:T9999"), _row(), now=NOW) is False


def test_eval_severity_and_status():
    row = _row(_severity="high", _status="contained")
    assert evaluate(parse("severity:high status:contained"), row, now=NOW) is True


# ---------------------------------------------------------------------------
# Evaluator: numbers + ranges
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("confidence:>0.8", True),
        ("confidence:>=0.92", True),
        ("confidence:>0.92", False),
        ("confidence:<0.5", False),
        ("confidence:<=0.92", True),
        ("confidence:[0.9 TO 1.0]", True),
        ("confidence:[0.0 TO 0.5]", False),
    ],
)
def test_eval_numeric_comparisons(expr, expected):
    assert evaluate(parse(expr), _row(max_prob=0.92), now=NOW) is expected


def test_eval_anomaly_threshold():
    assert evaluate(parse("anomaly:>=80"), _row(_anomaly=88), now=NOW) is True
    assert evaluate(parse("anomaly:>=80"), _row(_anomaly=10), now=NOW) is False


def test_eval_string_field_rejects_op():
    with pytest.raises(ParseError):
        evaluate(parse("label:>malware"), _row(), now=NOW)


def test_eval_number_field_rejects_non_numeric():
    with pytest.raises(ParseError):
        evaluate(parse("confidence:high"), _row(), now=NOW)


# ---------------------------------------------------------------------------
# Evaluator: time + duration
# ---------------------------------------------------------------------------


def test_eval_last_duration_within_window():
    row = _row(timestamp=(NOW - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"))
    assert evaluate(parse("last:24h"), row, now=NOW) is True
    assert evaluate(parse("last:1h"), row, now=NOW) is False


def test_eval_after_before():
    row = _row(timestamp="2026-05-03 11:00:00")
    assert evaluate(parse("after:2026-05-01"), row, now=NOW) is True
    assert evaluate(parse("before:2026-05-04"), row, now=NOW) is True
    assert evaluate(parse("before:2026-05-02"), row, now=NOW) is False


def test_eval_invalid_duration_raises():
    with pytest.raises(ParseError):
        parse_duration("forever")


def test_eval_last_with_naive_decorated_dt_does_not_crash():
    """Regression: app.py's _decorate_history_row stores a NAIVE _dt
    (from datetime.fromisoformat on a SQLite string). Comparing that
    against the tz-aware `now` previously raised
    `TypeError: can't compare offset-naive and offset-aware datetimes`."""
    naive_dt = (NOW - timedelta(hours=2)).replace(tzinfo=None)
    row = _row(timestamp=None)
    row["_dt"] = naive_dt
    assert evaluate(parse("last:24h"), row, now=NOW) is True
    assert evaluate(parse("last:1h"), row, now=NOW) is False


def test_eval_after_with_naive_decorated_dt():
    naive_dt = datetime(2026, 5, 3, 11, 0, 0)  # no tzinfo
    row = _row(timestamp=None)
    row["_dt"] = naive_dt
    assert evaluate(parse("after:2026-05-01"), row, now=NOW) is True
    assert evaluate(parse("before:2026-05-04"), row, now=NOW) is True


# ---------------------------------------------------------------------------
# Boolean composition
# ---------------------------------------------------------------------------


def test_eval_and_or_not_combined():
    row = _row(final_label="malware", max_prob=0.95)
    expr = "(label:malware OR label:phishing) AND confidence:>=0.9 AND NOT status:closed"
    assert evaluate(parse(expr), row, now=NOW) is True


def test_eval_minus_excludes():
    rows = [
        _row(id=1, final_label="malware"),
        _row(id=2, final_label="benign_activity"),
    ]
    pred = compile_query("-label:benign_activity")
    kept = [r for r in rows if pred(r)]
    assert [r["id"] for r in kept] == [1]


def test_eval_compile_query_predicate_runs():
    pred = compile_query("label:data_exfiltration AND mitre:T1041")
    assert pred(_row()) is True


def test_eval_empty_query_matches_everything():
    assert evaluate(parse("   "), _row(), now=NOW) is True
