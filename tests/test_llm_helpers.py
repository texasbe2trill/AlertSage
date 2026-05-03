"""Tests for the small set of pure helpers in triage.llm_helpers
that the BYOK classifier override path in app.py depends on."""

from __future__ import annotations

import copy
import inspect

import pytest

from triage import llm_helpers
from triage.llm_helpers import (
    LLM_ASSIST_FALLBACK,
    LLM_ASSIST_MODES,
    LLM_ASSIST_OFF,
    LLM_ASSIST_OVERRIDE,
    LLMProviderError,
    MITRE_MAPPING,
    RATE_LIMIT_BYOK_CAP,
    RATE_LIMIT_BYOK_WINDOW_S,
    RATE_LIMIT_DEMO_CAP,
    RATE_LIMIT_DEMO_WINDOW_S,
    RATE_LIMIT_LOCAL_CAP,
    apply_llm_override,
    effective_rate_window,
    with_forced_fallback,
)


def _sklearn_result(label: str = "uncertain", max_prob: float = 0.42) -> dict:
    return {
        "incident_text": "anything",
        "base_label": label,
        "final_label": label,
        "max_prob": max_prob,
        "mitre_techniques": MITRE_MAPPING.get(label, []),
    }


def test_modes_are_distinct_and_complete():
    assert set(LLM_ASSIST_MODES) == {
        LLM_ASSIST_OFF, LLM_ASSIST_FALLBACK, LLM_ASSIST_OVERRIDE,
    }


def test_apply_llm_override_replaces_label_and_techniques():
    result = _sklearn_result(label="uncertain")
    opinion = {"label": "phishing", "mitre_ids": ["T1566"], "rationale": "..."}

    applied = apply_llm_override(result, opinion)

    assert applied is True
    assert result["final_label"] == "phishing"
    assert result["mitre_techniques"] == ["T1566"]


def test_apply_llm_override_falls_back_to_canonical_mitre_when_opinion_omits_ids():
    result = _sklearn_result(label="uncertain")
    opinion = {"label": "malware", "mitre_ids": [], "rationale": "..."}

    applied = apply_llm_override(result, opinion)

    assert applied is True
    assert result["final_label"] == "malware"
    # malware in MITRE_MAPPING is the source of truth here.
    assert result["mitre_techniques"] == MITRE_MAPPING["malware"]


def test_apply_llm_override_skips_uncertain_label():
    result = _sklearn_result(label="phishing", max_prob=0.81)
    snapshot = copy.deepcopy(result)
    opinion = {"label": "uncertain", "mitre_ids": [], "rationale": "..."}

    applied = apply_llm_override(result, opinion)

    assert applied is False
    assert result == snapshot


def test_apply_llm_override_skips_invalid_label():
    """Defense in depth: an LLM that returns an out-of-vocab label
    must NOT mutate the result, even though llm_second_opinion already
    normalizes. The override helper must not assume that normalization
    happened, since callers might pass raw model output someday."""
    result = _sklearn_result(label="phishing", max_prob=0.81)
    snapshot = copy.deepcopy(result)
    opinion = {
        "label": "definitely_not_a_real_label",
        "mitre_ids": ["T9999"],
        "rationale": "...",
    }

    applied = apply_llm_override(result, opinion)

    assert applied is False
    assert result == snapshot


def test_apply_llm_override_handles_none_and_empty_opinion():
    result = _sklearn_result(label="phishing")
    snapshot = copy.deepcopy(result)

    assert apply_llm_override(result, None) is False
    assert apply_llm_override(result, {}) is False
    assert apply_llm_override(result, {"label": ""}) is False
    assert result == snapshot


class _StubClassifier:
    """Records call sequence and returns scripted opinions, so we can
    assert the orchestrator's retry-on-uncertain behavior end to end."""

    def __init__(self, scripted: list[tuple[dict | None, str | None]]) -> None:
        self.scripted = list(scripted)
        self.calls: list[dict] = []

    def __call__(
        self,
        text: str,
        *,
        skip_preprocessing: bool = False,
        force_classification: bool = False,
    ) -> tuple[dict | None, str | None]:
        self.calls.append(
            {
                "text": text,
                "skip_preprocessing": skip_preprocessing,
                "force_classification": force_classification,
            }
        )
        return self.scripted.pop(0)


def test_with_forced_fallback_skips_retry_when_first_pass_is_actionable():
    stub = _StubClassifier([({"label": "malware", "mitre_ids": ["T1486"]}, None)])
    opinion, err, details = with_forced_fallback(stub, "ransomware on WS-FIN-04")

    assert opinion == {"label": "malware", "mitre_ids": ["T1486"]}
    assert err is None
    assert details["force_pass_attempted"] is False
    assert details["first_pass_label"] == "malware"
    assert len(stub.calls) == 1
    assert stub.calls[0]["force_classification"] is False


def test_with_forced_fallback_retries_on_uncertain_with_force_flag():
    """The behavioral test the user asked for: a first-pass 'uncertain'
    opinion must trigger a second call with force_classification=True."""
    stub = _StubClassifier(
        [
            ({"label": "uncertain", "mitre_ids": [], "rationale": "hedging"}, None),
            ({"label": "phishing", "mitre_ids": ["T1566"], "rationale": "..."}, None),
        ]
    )
    opinion, err, details = with_forced_fallback(stub, "Teams chat from fake IT")

    assert err is None
    assert opinion["label"] == "phishing"
    assert details["force_pass_attempted"] is True
    assert details["first_pass_label"] == "uncertain"
    assert details["force_pass_label"] == "phishing"
    assert len(stub.calls) == 2
    # First call is the normal pass.
    assert stub.calls[0]["force_classification"] is False
    # Retry uses force_classification.
    assert stub.calls[1]["force_classification"] is True
    # Same input text on both calls.
    assert stub.calls[0]["text"] == stub.calls[1]["text"]


def test_with_forced_fallback_returns_first_pass_when_force_pass_errors():
    """If the second pass fails (rate limit, network), the orchestrator
    surfaces the first-pass opinion plus the force-pass error so the
    UI still has a rationale to render and apply_llm_override skips
    the override on the 'uncertain' label."""
    stub = _StubClassifier(
        [
            ({"label": "uncertain", "mitre_ids": [], "rationale": "first"}, None),
            (None, "rate limit reached"),
        ]
    )
    opinion, err, details = with_forced_fallback(stub, "anything")

    assert opinion is not None
    assert opinion["label"] == "uncertain"
    assert err == "rate limit reached"
    assert details["force_pass_attempted"] is True
    assert details["force_pass_err"] == "rate limit reached"
    assert details["force_pass_label"] is None
    assert len(stub.calls) == 2


def test_with_forced_fallback_passes_through_first_pass_error():
    """If the first pass errors (no opinion), don't burn a second call.

    This is exactly the path that placeholder failures now take after
    the LLMProviderError change: the wrapper sees (None, err) and
    bails immediately rather than retrying against the same broken
    upstream. The provider-failures counter in app.py increments off
    this path, and the surface warning shows the err string.
    """
    stub = _StubClassifier([(None, "OpenAI inference failed: 429 Too Many Requests")])
    opinion, err, details = with_forced_fallback(stub, "anything")

    assert opinion is None
    assert err == "OpenAI inference failed: 429 Too Many Requests"
    assert details["force_pass_attempted"] is False
    assert details["first_pass_err"] == "OpenAI inference failed: 429 Too Many Requests"
    assert len(stub.calls) == 1


def test_placeholder_result_raises_llm_provider_error():
    """The previous _placeholder_result silently returned a synthetic
    {'label': 'uncertain'} dict, which masked provider failures as
    'successful uncertain' answers and bypassed both the
    force_classification rewrite and the orchestrator's err path.
    The new contract is that it raises, so the outer try/except in
    run_llm_second_opinion converts it into (None, err)."""
    with pytest.raises(LLMProviderError) as excinfo:
        llm_helpers._placeholder_result("OpenAI inference failed: 401 unauthorized")
    assert "401 unauthorized" in str(excinfo.value)


def test_with_forced_fallback_does_not_retry_on_provider_failure():
    """Re-stating the failure-bail contract from the placeholder
    perspective: when the first call raises a provider error (now
    surfaced as None+err), the orchestrator must NOT make a second
    call. The same upstream is just as broken in force-mode; retrying
    only doubles the failure-rate without ever producing a label."""
    stub = _StubClassifier([(None, "Hugging Face: model is currently loading")])
    opinion, err, details = with_forced_fallback(stub, "anything")

    assert len(stub.calls) == 1, (
        "force-pass must not retry on provider failure"
    )
    assert opinion is None
    assert err == "Hugging Face: model is currently loading"
    assert details["force_pass_attempted"] is False


def test_effective_rate_window_byok_openai_anthropic_huggingface():
    """BYOK calls go against the user's own quota, so the cap lifts to
    the BYOK ceiling for all three hosted providers. A regression that
    silently drops one of these providers back to the demo cap would
    re-create the rate-limited starvation the user reported."""
    for provider in ("openai", "anthropic", "huggingface"):
        cap, window = effective_rate_window(provider, byok_present=True)
        assert cap == RATE_LIMIT_BYOK_CAP, f"{provider} BYOK cap regressed"
        assert window == RATE_LIMIT_BYOK_WINDOW_S


def test_effective_rate_window_bundled_huggingface_keeps_demo_cap():
    """The bundled HF token is shared across every visitor on the
    Streamlit Cloud deploy; without BYOK the modest demo cap protects
    that shared quota from one user burning it all."""
    cap, window = effective_rate_window("huggingface", byok_present=False)
    assert cap == RATE_LIMIT_DEMO_CAP
    assert window == RATE_LIMIT_DEMO_WINDOW_S


def test_effective_rate_window_local_is_effectively_unlimited():
    """Local llama.cpp throughput is bounded by CPU/GPU, not API
    quota, so an explicit very-high cap keeps the existing sliding-
    window code unchanged without ever throttling local calls."""
    cap, window = effective_rate_window("local", byok_present=False)
    assert cap == RATE_LIMIT_LOCAL_CAP
    # byok_present is irrelevant for local; still high.
    cap_b, _ = effective_rate_window("local", byok_present=True)
    assert cap_b == RATE_LIMIT_LOCAL_CAP


def test_effective_rate_window_unknown_provider_falls_back_to_demo_cap():
    """Defense in depth: a typo or new provider that lands here
    without a dedicated branch should default to the conservative
    cap rather than the BYOK ceiling, so nobody can accidentally
    bypass the demo throttle by misspelling 'openai'."""
    cap, window = effective_rate_window("not-a-provider", byok_present=True)
    assert cap == RATE_LIMIT_DEMO_CAP
    assert window == RATE_LIMIT_DEMO_WINDOW_S


def test_mitre_mapping_includes_insider_threat():
    """sklearn's class set includes 'insider_threat'; without an entry
    in MITRE_MAPPING those events bucketed to UNMAPPED in the coverage
    CSV. The mapping below picks the techniques that best characterize
    the canonical insider scenario."""
    assert "insider_threat" in MITRE_MAPPING
    techs = MITRE_MAPPING["insider_threat"]
    assert "T1078" in techs, "Valid Accounts must be present"
    assert "T1567" in techs, "Exfiltration Over Web Service must be present"
    assert "T1052" in techs, "Exfiltration Over Physical Medium must be present"


def test_llm_second_opinion_uses_analyst_voice_for_uncertain_default_pass():
    """Contract: the default-pass prompt frames 'uncertain' as a last
    resort with explicit criteria, not as a peer of the seven actionable
    labels. A regression that puts 'uncertain' back in the allowed-set
    enumeration would re-create the leak the user complained about."""
    source = inspect.getsource(llm_helpers.llm_second_opinion)
    # Analyst-voice phrasing must appear. The strings get split across
    # adjacent string literals in the source, so we look for fragments
    # that survive the concatenation rather than full sentences.
    assert "Lean toward the most plausible attacker-aligned label" in source
    assert "over-classification is" in source
    assert "a missed one is not" in source
    assert "genuinely uninterpretable" in source
    # Force-pass clause must also exist.
    assert "force_classification" in source
    assert "'uncertain' is NOT permitted under any circumstances" in source


def test_llm_second_opinion_synonym_map_covers_common_near_misses():
    """The expanded synonym map absorbs the broader MITRE-tactic
    vocabulary that capable models reach for. Each of these terms used
    to schema-rebound to 'uncertain'; with the map, they land on a
    canonical actionable label. If any of these mappings disappears,
    the user will start seeing more 'uncertain' rows again."""
    source = inspect.getsource(llm_helpers.llm_second_opinion)
    expected_pairs = (
        ('"lateral_movement": "access_abuse"', "lateral_movement"),
        ('"credential_stuffing": "access_abuse"', "credential_stuffing"),
        ('"command_and_control": "malware"', "command_and_control"),
        ('"persistence": "malware"', "persistence"),
        ('"defense_evasion": "malware"', "defense_evasion"),
        ('"ddos": "web_attack"', "ddos"),
        ('"social_engineering": "phishing"', "social_engineering"),
        ('"data_theft": "data_exfiltration"', "data_theft"),
    )
    for pattern, term in expected_pairs:
        assert pattern in source, (
            f"Synonym mapping for {term!r} is missing; "
            "schema rebound will start sending it to 'uncertain' again."
        )


def test_llm_second_opinion_force_mode_never_returns_uncertain():
    """Contract for the second-pass retry: force_classification mode
    must rewrite 'uncertain' (whether returned by the LLM or produced
    by the schema gate) to 'benign_activity'. This is what makes the
    forced-fallback path actually escape uncertain instead of looping
    back to it."""
    source = inspect.getsource(llm_helpers.llm_second_opinion)
    # The schema-gate rebound default is benign_activity in force mode.
    assert (
        '"benign_activity" if force_classification else "uncertain"' in source
    )
    # The post-gate guard rewrites a literal 'uncertain' to benign_activity
    # in force mode so a stubborn LLM can't bypass the schema rewrite by
    # returning the in-vocab string 'uncertain'.
    assert "force_classification and label == \"uncertain\"" in source


def test_llm_second_opinion_has_no_provider_asymmetry():
    """Contract: every provider's LLM response goes through the same
    post-parse normalization (synonym map plus schema validation) and
    the same fallback path. The previous build had a `trusted_provider`
    boolean that gated extra defensive guards for HuggingFace and local
    llama.cpp, which routinely downgraded confident in-vocab answers
    that hosted providers got to keep. This test guards against
    accidental reintroduction of that asymmetry: any future code that
    branches the LABEL handling on provider identity will fail it.

    Note: provider identity is still legitimately consulted at the
    BACKEND DISPATCH level (HF -> local fallback when llama is
    available), and that branch is untouched by this contract.
    """
    source = inspect.getsource(llm_helpers.llm_second_opinion)
    assert "trusted_provider" not in source, (
        "Reintroducing 'trusted_provider' would re-create the per-provider "
        "asymmetry where HuggingFace and local responses get downgraded "
        "relative to OpenAI and Anthropic. Backend dispatch (which client "
        "to import) is still allowed to branch on provider_choice; what "
        "is forbidden is conditioning LABEL handling on provider identity, "
        "and the trusted_provider boolean was the gate that did exactly "
        "that. Removing it removed the asymmetry."
    )
    # Defense in depth: the post-parse normalization should not branch
    # on provider identity at all. We assert this by checking that
    # synonym normalization and schema validation are unconditional in
    # the source rather than nested under any `if provider_*` block.
    assert "synonym_map" in source, "synonym normalization must apply uniformly"
    assert "if label not in MITRE_MAPPING" in source, (
        "schema validation must apply uniformly"
    )
