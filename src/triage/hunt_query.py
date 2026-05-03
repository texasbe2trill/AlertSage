"""Lucene-style query DSL for hunting AlertSage triage history.

Grammar (informal):

    expr     := or_expr
    or_expr  := and_expr ( ('OR' | '||') and_expr )*
    and_expr := unary    ( ('AND' | '&&')? unary )*       # implicit AND
    unary    := ('NOT' | '-') unary | primary
    primary  := '(' or_expr ')'
              | clause
    clause   := QUOTED                                    # bare quoted -> narrative substring
              | WORD ( ':' value )?                        # WORD or field:value
    value    := QUOTED
              | WORD
              | OP (QUOTED | WORD)                         # confidence:>0.8
              | '[' (WORD|QUOTED) 'TO' (WORD|QUOTED) ']'   # confidence:[0.5 TO 1.0]

Examples:

    label:malware
    severity:critical AND last:24h
    confidence:>0.8 narrative:"dropbox"
    mitre:T1566 OR mitre:T1190
    NOT status:closed
    -label:benign_activity

Bare tokens (no field) match against the incident narrative as
case-insensitive substrings, so `dropbox` is shorthand for
`narrative:"dropbox"`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable

# ---------------------------------------------------------------------------
# AST + errors
# ---------------------------------------------------------------------------


class ParseError(ValueError):
    """Raised on malformed DSL queries.

    Carries `pos` (column in the input string) so the UI can render a
    helpful caret. Always include the original query text in the
    formatted message for clarity in tracebacks.
    """

    def __init__(self, msg: str, pos: int = 0):
        super().__init__(msg)
        self.pos = pos
        self.msg = msg

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.msg} (at column {self.pos + 1})"


@dataclass(frozen=True)
class Term:
    """Bare term -- matches narrative substring."""

    text: str


@dataclass(frozen=True)
class FieldExpr:
    """Single field comparison: name OP value."""

    name: str  # canonical
    op: str    # ":", ">", ">=", "<", "<="
    value: str


@dataclass(frozen=True)
class RangeExpr:
    """Inclusive range: name:[lo TO hi]."""

    name: str
    lo: str
    hi: str


@dataclass(frozen=True)
class NotExpr:
    child: Any


@dataclass(frozen=True)
class AndExpr:
    items: tuple


@dataclass(frozen=True)
class OrExpr:
    items: tuple


Node = Term | FieldExpr | RangeExpr | NotExpr | AndExpr | OrExpr


# ---------------------------------------------------------------------------
# Field registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """Describes one queryable field.

    `kind` selects how `value` is interpreted:
      - "string":     case-insensitive equality
      - "substring":  case-insensitive substring containment
      - "number":     numeric (supports >, >=, <, <=, [a TO b])
      - "list":       extracted value is a list; match if value is a
                      member (case-insensitive equality)
      - "duration":   value is a duration like "24h"; matches when the
                      timestamp is within `now - duration`
      - "time":       value is an ISO date; supports >, <, >=, <=, :,
                      with `:` defaulting to `>=` for `after` and `<=`
                      for `before`
    """

    name: str
    aliases: tuple[str, ...]
    kind: str
    extract: Callable[[dict], Any]
    suggestions: tuple[str, ...] = ()
    help: str = ""


def _row_label(r: dict) -> str:
    return str(r.get("final_label") or "").lower()


def _row_text(r: dict) -> str:
    return str(r.get("incident_text") or "").lower()


def _row_conf(r: dict) -> float:
    try:
        return float(r.get("max_prob") or 0)
    except (TypeError, ValueError):
        return 0.0


def _row_anomaly(r: dict) -> int:
    try:
        return int(r.get("_anomaly") or 0)
    except (TypeError, ValueError):
        return 0


def _row_status(r: dict) -> str:
    return str(r.get("_status") or "new").lower()


def _row_severity(r: dict) -> str:
    return str(r.get("_severity") or "").lower()


def _row_mitre(r: dict) -> list[str]:
    raw = r.get("raw_result") or {}
    techs = raw.get("mitre_techniques") or []
    return [str(t).upper() for t in techs]


def _row_id(r: dict) -> int:
    for key in ("id", "analysis_id"):
        v = r.get(key)
        if v is None:
            continue
        try:
            return int(v)
        except (TypeError, ValueError):
            continue
    return 0


def _ensure_aware(dt: datetime | None) -> datetime | None:
    """Return `dt` with a UTC tzinfo if it was naive.

    The caller (Streamlit UI, CLI, tests) hands us datetimes from many
    sources -- SQLite ISO strings, ``datetime.now()`` without a tz, etc.
    Comparing a naive datetime against ``now`` (which we always build
    tz-aware) raises ``TypeError``, so we normalize at the read edge.
    """
    if dt is None:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _row_dt(r: dict) -> datetime | None:
    """Return the row timestamp as a tz-aware datetime, or None."""
    cached = r.get("_dt")
    if isinstance(cached, datetime):
        return _ensure_aware(cached)
    raw = r.get("timestamp")
    if not raw:
        return None
    if isinstance(raw, datetime):
        return _ensure_aware(raw)
    s = str(raw).strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec(
        name="label",
        aliases=("class", "classification"),
        kind="string",
        extract=_row_label,
        suggestions=(
            "malware",
            "phishing",
            "data_exfiltration",
            "access_abuse",
            "web_attack",
            "policy_violation",
            "benign_activity",
            "uncertain",
        ),
        help="Final triage label.",
    ),
    FieldSpec(
        name="severity",
        aliases=("sev",),
        kind="string",
        extract=_row_severity,
        suggestions=("critical", "high", "medium", "low", "info"),
        help="Severity tier derived from label.",
    ),
    FieldSpec(
        name="status",
        aliases=(),
        kind="string",
        extract=_row_status,
        suggestions=("new", "triaging", "contained", "closed"),
        help="Case management status.",
    ),
    FieldSpec(
        name="confidence",
        aliases=("conf",),
        kind="number",
        extract=_row_conf,
        suggestions=("0.5", "0.8", "0.9"),
        help="Classifier confidence 0.0-1.0. Use >, <, >=, <=, [a TO b].",
    ),
    FieldSpec(
        name="anomaly",
        aliases=(),
        kind="number",
        extract=_row_anomaly,
        suggestions=("50", "75", "90"),
        help="Anomaly score 0-100. Use >, <, >=, <=, [a TO b].",
    ),
    FieldSpec(
        name="narrative",
        aliases=("text", "body", "desc", "description"),
        kind="substring",
        extract=_row_text,
        help="Substring match against the incident narrative.",
    ),
    FieldSpec(
        name="mitre",
        aliases=("technique", "attack"),
        kind="list",
        extract=_row_mitre,
        suggestions=("T1566", "T1059", "T1486", "T1190", "T1078", "T1041"),
        help="MITRE ATT&CK technique id (e.g. T1566).",
    ),
    FieldSpec(
        name="id",
        aliases=("analysis_id", "aid"),
        kind="number",
        extract=_row_id,
        help="Analysis ID (integer).",
    ),
    FieldSpec(
        name="last",
        aliases=(),
        kind="duration",
        extract=_row_dt,
        suggestions=("1h", "24h", "7d", "30d"),
        help="Events within the past duration. Example: last:24h.",
    ),
    FieldSpec(
        name="after",
        aliases=("since",),
        kind="time",
        extract=_row_dt,
        suggestions=("2026-04-01", "2026-04-15"),
        help="Events on or after this timestamp (YYYY-MM-DD).",
    ),
    FieldSpec(
        name="before",
        aliases=("until",),
        kind="time",
        extract=_row_dt,
        suggestions=("2026-04-15",),
        help="Events on or before this timestamp (YYYY-MM-DD).",
    ),
)

_FIELD_INDEX: dict[str, FieldSpec] = {}
for _spec in FIELDS:
    _FIELD_INDEX[_spec.name] = _spec
    for _alias in _spec.aliases:
        _FIELD_INDEX[_alias] = _spec


def field_names() -> list[str]:
    """All canonical field names, in registry order. Used by the UI."""
    return [s.name for s in FIELDS]


def field_spec(name: str) -> FieldSpec | None:
    return _FIELD_INDEX.get(name.lower())


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_KEYWORDS = {"AND", "OR", "NOT", "TO"}


def tokenize(s: str) -> list[tuple[str, str, int]]:
    """Return a list of (kind, value, position) triples plus an EOF.

    The tokenizer splits words at colons so that ``label:malware``
    becomes three tokens (WORD, COLON, WORD), which keeps the parser
    grammar simple.
    """
    tokens: list[tuple[str, str, int]] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            i += 1
            continue
        if c == "(":
            tokens.append(("LPAREN", "(", i))
            i += 1
            continue
        if c == ")":
            tokens.append(("RPAREN", ")", i))
            i += 1
            continue
        if c == "[":
            tokens.append(("LBRACK", "[", i))
            i += 1
            continue
        if c == "]":
            tokens.append(("RBRACK", "]", i))
            i += 1
            continue
        if c == ":":
            tokens.append(("COLON", ":", i))
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and s[j] != '"':
                j += 1
            if j >= n:
                raise ParseError("Unterminated quoted string", i)
            tokens.append(("QUOTED", s[i + 1 : j], i))
            i = j + 1
            continue
        if c in "<>":
            if i + 1 < n and s[i + 1] == "=":
                tokens.append(("OP", c + "=", i))
                i += 2
            else:
                tokens.append(("OP", c, i))
                i += 1
            continue
        if c == "-":
            # Leading minus only acts as NOT when at the start of a
            # clause. A minus mid-token (e.g. inside "T1190-x") is part
            # of the word.
            prev = tokens[-1][0] if tokens else None
            if prev in (None, "LPAREN", "AND", "OR", "NOT", "MINUS"):
                if i + 1 < n and not s[i + 1].isspace() and s[i + 1] not in "()[]":
                    tokens.append(("MINUS", "-", i))
                    i += 1
                    continue
        # Word: read until whitespace, structural char, or colon
        j = i
        while j < n and not s[j].isspace() and s[j] not in '()[]:"' and s[j] not in "<>":
            j += 1
        if j == i:
            # Should not happen given the branches above, but guard
            # against an infinite loop on stray characters.
            raise ParseError(f"Unexpected character: {c!r}", i)
        word = s[i:j]
        upper = word.upper()
        if upper in _KEYWORDS:
            tokens.append((upper, upper, i))
        else:
            tokens.append(("WORD", word, i))
        i = j
    tokens.append(("EOF", "", n))
    return tokens


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[tuple[str, str, int]]):
        self.toks = tokens
        self.i = 0

    def _peek(self) -> tuple[str, str, int]:
        return self.toks[self.i]

    def _advance(self) -> tuple[str, str, int]:
        t = self.toks[self.i]
        self.i += 1
        return t

    def _expect(self, kind: str) -> tuple[str, str, int]:
        t = self._advance()
        if t[0] != kind:
            raise ParseError(f"Expected {kind}, got {t[1]!r}", t[2])
        return t

    def parse(self) -> Node:
        if self._peek()[0] == "EOF":
            # Empty query matches everything.
            return AndExpr(items=())
        node = self._or()
        t = self._peek()
        if t[0] != "EOF":
            raise ParseError(f"Unexpected token {t[1]!r}", t[2])
        return node

    def _or(self) -> Node:
        left = self._and()
        items = [left]
        while self._peek()[0] == "OR":
            self._advance()
            items.append(self._and())
        if len(items) == 1:
            return items[0]
        return OrExpr(items=tuple(items))

    def _and(self) -> Node:
        left = self._unary()
        items = [left]
        while True:
            t = self._peek()
            if t[0] in ("EOF", "RPAREN", "OR"):
                break
            if t[0] == "AND":
                self._advance()
            items.append(self._unary())
        if len(items) == 1:
            return items[0]
        return AndExpr(items=tuple(items))

    def _unary(self) -> Node:
        t = self._peek()
        if t[0] in ("NOT", "MINUS"):
            self._advance()
            return NotExpr(child=self._unary())
        return self._primary()

    def _primary(self) -> Node:
        t = self._peek()
        if t[0] == "LPAREN":
            self._advance()
            inner = self._or()
            self._expect("RPAREN")
            return inner
        if t[0] == "QUOTED":
            self._advance()
            return Term(text=t[1])
        if t[0] == "WORD":
            self._advance()
            if self._peek()[0] == "COLON":
                self._advance()
                return self._field_value(t[1], t[2])
            return Term(text=t[1])
        raise ParseError(f"Unexpected token {t[1]!r}", t[2])

    def _field_value(self, name: str, name_pos: int) -> Node:
        spec = field_spec(name)
        if spec is None:
            raise ParseError(
                f"Unknown field {name!r}. Try one of: {', '.join(field_names())}",
                name_pos,
            )

        t = self._peek()
        if t[0] == "OP":
            self._advance()
            v = self._advance()
            if v[0] not in ("WORD", "QUOTED"):
                raise ParseError(f"Expected value after {t[1]!r}", v[2])
            return FieldExpr(name=spec.name, op=t[1], value=v[1])
        if t[0] == "LBRACK":
            self._advance()
            lo = self._advance()
            if lo[0] not in ("WORD", "QUOTED"):
                raise ParseError("Expected lower bound after '['", lo[2])
            to = self._advance()
            if to[0] != "TO":
                raise ParseError("Expected TO in range", to[2])
            hi = self._advance()
            if hi[0] not in ("WORD", "QUOTED"):
                raise ParseError("Expected upper bound after TO", hi[2])
            self._expect("RBRACK")
            return RangeExpr(name=spec.name, lo=lo[1], hi=hi[1])
        if t[0] in ("WORD", "QUOTED"):
            self._advance()
            return FieldExpr(name=spec.name, op=":", value=t[1])
        raise ParseError(f"Expected value after {name!r}:", t[2])


def parse(query: str) -> Node:
    """Parse a DSL string into an AST."""
    return _Parser(tokenize(query)).parse()


# ---------------------------------------------------------------------------
# Duration / datetime helpers
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(r"^\s*(\d+)\s*(s|m|h|d|w)\s*$", re.I)
_DURATION_UNITS = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def parse_duration(s: str) -> timedelta:
    m = _DURATION_RE.match(s)
    if not m:
        raise ParseError(f"Invalid duration {s!r}; expected like 24h, 7d, 30m")
    n = int(m.group(1))
    unit = _DURATION_UNITS[m.group(2).lower()]
    return timedelta(**{unit: n})


def parse_datetime(s: str) -> datetime:
    s = s.strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ParseError(f"Invalid date {s!r}; use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


def _num_compare(actual: float, op: str, target: float) -> bool:
    if op == ":":
        return abs(actual - target) < 1e-9
    if op == ">":
        return actual > target
    if op == ">=":
        return actual >= target
    if op == "<":
        return actual < target
    if op == "<=":
        return actual <= target
    return False


def _eval_field(node: FieldExpr, row: dict, now: datetime) -> bool:
    spec = _FIELD_INDEX[node.name]
    extracted = spec.extract(row)

    if spec.kind == "duration":
        if not isinstance(extracted, datetime):
            return False
        delta = parse_duration(node.value)
        return extracted >= now - delta

    if spec.kind == "time":
        if not isinstance(extracted, datetime):
            return False
        target = parse_datetime(node.value)
        if node.op == ":":
            if spec.name == "after":
                return extracted >= target
            if spec.name == "before":
                return extracted <= target
            return extracted >= target
        return _num_compare(extracted.timestamp(), node.op, target.timestamp())

    if spec.kind == "number":
        try:
            target = float(node.value)
        except ValueError as exc:
            raise ParseError(
                f"{spec.name!r} expects a number, got {node.value!r}"
            ) from exc
        try:
            actual = float(extracted) if extracted is not None else 0.0
        except (TypeError, ValueError):
            actual = 0.0
        return _num_compare(actual, node.op, target)

    if spec.kind == "string":
        if node.op != ":":
            raise ParseError(
                f"{spec.name!r} only supports ':' equality, not {node.op!r}"
            )
        return str(extracted).lower() == node.value.lower()

    if spec.kind == "substring":
        if node.op != ":":
            raise ParseError(
                f"{spec.name!r} only supports ':' substring, not {node.op!r}"
            )
        return node.value.lower() in str(extracted).lower()

    if spec.kind == "list":
        if node.op != ":":
            raise ParseError(
                f"{spec.name!r} only supports ':' membership, not {node.op!r}"
            )
        target = node.value.upper()
        return any(target == str(it).upper() for it in (extracted or []))

    return False


def _eval_range(node: RangeExpr, row: dict, now: datetime) -> bool:
    spec = _FIELD_INDEX[node.name]
    if spec.kind == "number":
        try:
            lo = float(node.lo)
            hi = float(node.hi)
        except ValueError as exc:
            raise ParseError(
                f"{spec.name!r} range expects numbers, got [{node.lo} TO {node.hi}]"
            ) from exc
        try:
            actual = float(spec.extract(row) or 0)
        except (TypeError, ValueError):
            actual = 0.0
        return lo <= actual <= hi
    if spec.kind == "time":
        extracted = spec.extract(row)
        if not isinstance(extracted, datetime):
            return False
        return parse_datetime(node.lo) <= extracted <= parse_datetime(node.hi)
    raise ParseError(f"{spec.name!r} does not support [a TO b] ranges")


def evaluate(node: Node, row: dict, *, now: datetime | None = None) -> bool:
    """Return True when `row` matches the parsed query."""
    now = now or datetime.now(timezone.utc)
    if isinstance(node, AndExpr):
        return all(evaluate(c, row, now=now) for c in node.items)
    if isinstance(node, OrExpr):
        return any(evaluate(c, row, now=now) for c in node.items)
    if isinstance(node, NotExpr):
        return not evaluate(node.child, row, now=now)
    if isinstance(node, Term):
        return node.text.lower() in _row_text(row)
    if isinstance(node, FieldExpr):
        return _eval_field(node, row, now)
    if isinstance(node, RangeExpr):
        return _eval_range(node, row, now)
    raise TypeError(f"Unknown node type: {type(node).__name__}")


def compile_query(query: str) -> Callable[[dict], bool]:
    """Parse `query` once, then return a row-predicate.

    Captures `now` at compile time so a single hunt run uses a stable
    "as of" instant for relative duration filters like ``last:24h``.
    """
    node = parse(query)
    now = datetime.now(timezone.utc)
    return lambda row: evaluate(node, row, now=now)


def filter_rows(query: str, rows: Iterable[dict]) -> list[dict]:
    """Convenience: parse + evaluate against an iterable of rows."""
    predicate = compile_query(query)
    return [r for r in rows if predicate(r)]


__all__ = [
    "ParseError",
    "Term",
    "FieldExpr",
    "RangeExpr",
    "NotExpr",
    "AndExpr",
    "OrExpr",
    "FieldSpec",
    "FIELDS",
    "field_names",
    "field_spec",
    "tokenize",
    "parse",
    "evaluate",
    "compile_query",
    "filter_rows",
    "parse_duration",
    "parse_datetime",
]
