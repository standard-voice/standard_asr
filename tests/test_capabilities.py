# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hierarchical capability system."""

from __future__ import annotations

import pytest

from standard_asr.capabilities import (
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    DiarizationConstraints,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PhraseHintsConstraints,
    PromptCap,
    PromptConstraints,
    ReconnectCap,
    StreamingCapabilities,
    StreamTimestampsCap,
    WordTimestampsCap,
    _children,  # pyright: ignore[reportPrivateUsage]
    _get_child,  # pyright: ignore[reportPrivateUsage]
    _read_attr,  # pyright: ignore[reportPrivateUsage]
    granularity_offers_all,
)


def _rich() -> DeclaredCapabilities:
    return DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                runtime_override=FlagCap(supported=True),
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=3),
                ),
            ),
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word", "segment"]),
            guidance=GuidanceCaps(prompt=PromptCap(supported=True)),
        ),
        streaming=StreamingCapabilities(
            emits_partials=FlagCap(supported=True),
            reconnect=ReconnectCap(mode="lossy"),
        ),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )


def test_supports_leaf_flags() -> None:
    caps = _rich()
    assert caps.supports("batch.language.runtime_override") is True
    assert caps.supports("batch.word_timestamps") is True
    assert caps.supports("streaming.emits_partials") is True


def test_supports_top_level_orthogonal() -> None:
    caps = _rich()
    assert caps.supports("streaming_input") is True
    assert caps.supports("streaming_output") is True


def test_supports_fail_closed_missing_key() -> None:
    caps = _rich()
    # Not declared under streaming guidance -> False.
    assert caps.supports("streaming.guidance.phrase_hints") is False
    # batch guidance phrase_hints default supported=False.
    assert caps.supports("batch.guidance.phrase_hints") is False
    assert caps.supports("batch.totally.unknown.path") is False


def test_supports_mode_node() -> None:
    caps = _rich()
    assert caps.supports("streaming.reconnect") is True  # lossy != unsupported
    caps2 = DeclaredCapabilities(
        streaming=StreamingCapabilities(reconnect=ReconnectCap(mode="unsupported"))
    )
    assert caps2.supports("streaming.reconnect") is False


def test_omitted_streaming_domain_unsupported() -> None:
    caps = DeclaredCapabilities(batch=BatchCapabilities())
    assert caps.supports("streaming") is False
    assert caps.supports("streaming.emits_partials") is False
    assert caps.supports("batch") is True


def test_default_is_fail_closed() -> None:
    caps = DeclaredCapabilities()
    assert caps.supports("batch") is False
    assert caps.supports("streaming_input") is False


def test_covers_subset_invariant() -> None:
    declared = _rich()
    # Effective narrows: drop word_timestamps support, the streaming domain and
    # (per fail-closed consistency) the streaming axis flags along with it.
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        ),
    )
    assert declared.covers(effective) is True


def test_covers_rejects_widening() -> None:
    declared = DeclaredCapabilities(batch=BatchCapabilities())
    # Effective claims more than declared -> not covered.
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word"])
        )
    )
    assert declared.covers(effective) is False


def test_covers_rejects_constraint_widening() -> None:
    # H1: declared max=2; effective claims max=999 -> widening, must be rejected
    # even though the SET of supported paths is identical.
    declared = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=2),
                )
            )
        )
    )
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=999),
                )
            )
        )
    )
    assert declared.covers(effective) is False


def test_covers_allows_constraint_narrowing() -> None:
    declared = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=5),
                )
            )
        )
    )
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=2),
                )
            )
        )
    )
    assert declared.covers(effective) is True
    # Equal limits are fine too.
    assert declared.covers(declared) is True


def test_numeric_constraints_reject_non_positive_values() -> None:
    # A numeric guidance limit (max count) MUST be a positive integer: a negative
    # or zero limit is nonsensical and would drive bogus slicing in param_gating
    # (a negative max silently truncates everything and reports a fake degrade), so
    # it is rejected at construction. None stays valid (unbounded).
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PromptConstraints(max_tokens=-1)
    with pytest.raises(ValidationError):
        PromptConstraints(max_tokens=0)
    with pytest.raises(ValidationError):
        PhraseHintsConstraints(max_terms=0)
    with pytest.raises(ValidationError):
        PhraseHintsConstraints(max_chars_per_term=-5)
    with pytest.raises(ValidationError):
        PhraseHintsConstraints(max_words_per_term=0)
    with pytest.raises(ValidationError):
        DiarizationConstraints(max_speakers=-2)
    with pytest.raises(ValidationError):
        CandidateLanguagesConstraints(max=0)

    # A valid positive limit (and an unconstrained None) still construct fine.
    assert PromptConstraints(max_tokens=5).max_tokens == 5
    assert PromptConstraints().max_tokens is None
    assert PhraseHintsConstraints(max_terms=3, max_chars_per_term=10).max_terms == 3
    assert DiarizationConstraints(max_speakers=4).max_speakers == 4


def test_covers_rejects_granularity_widening() -> None:
    declared = DeclaredCapabilities(
        batch=BatchCapabilities(
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word", "segment"])
        )
    )
    # Effective adds "char" -> widening the offered granularity set.
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(
            word_timestamps=WordTimestampsCap(
                supported=True, granularities=["word", "segment", "char"]
            )
        )
    )
    assert declared.covers(effective) is False
    # Narrowing to a subset is allowed.
    narrowed = DeclaredCapabilities(
        batch=BatchCapabilities(
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word"])
        )
    )
    assert declared.covers(narrowed) is True


def test_covers_empty_declared_granularities_is_unbounded() -> None:
    # CAPA-1: an empty declared granularities list means "unbounded (all)", so a
    # narrowing from it to any concrete subset is a valid effective ⊆ declared
    # (must NOT false-fail). Mirrors param_gating's empty="offers all" semantics.
    # Typed WordTimestampsCap can't be empty+supported (RUNT-6 validator), so the
    # empty-declared case is exercised via an x_* dict node carrying the field.
    assert granularity_offers_all([]) is True
    assert granularity_offers_all(["word"]) is False
    declared = _x_caps({"x_wt": {"supported": True, "granularities": []}})
    effective = _x_caps({"x_wt": {"supported": True, "granularities": ["word"]}})
    assert declared.covers(effective) is True
    # And empty -> empty is trivially covered.
    assert declared.covers(declared) is True


def test_word_timestamps_supported_requires_granularities() -> None:
    # RUNT-6: a typed WordTimestampsCap that declares supported=True MUST
    # enumerate at least one granularity (no "supported but unenumerated"
    # ambiguity). Unsupported keeps the empty default.
    with pytest.raises(ValueError, match="non-empty"):
        WordTimestampsCap(supported=True)
    # supported=False with empty granularities is fine.
    assert WordTimestampsCap(supported=False).granularities == []
    # supported=True with a granularity is fine.
    assert WordTimestampsCap(supported=True, granularities=["word"]).granularities == ["word"]


def test_covers_rejects_mode_widening() -> None:
    declared = DeclaredCapabilities(
        streaming=StreamingCapabilities(reconnect=ReconnectCap(mode="lossy"))
    )
    # Effective claims the stronger "seamless" mode -> widening.
    effective = DeclaredCapabilities(
        streaming=StreamingCapabilities(reconnect=ReconnectCap(mode="seamless"))
    )
    assert declared.covers(effective) is False
    # Reducing lossy -> unsupported is a valid close (handled by set logic, and
    # the reduction map permits it).
    reduced = DeclaredCapabilities(
        streaming=StreamingCapabilities(reconnect=ReconnectCap(mode="unsupported"))
    )
    assert declared.covers(reduced) is True


def test_covers_rejects_timestamps_mode_widening() -> None:
    declared = DeclaredCapabilities(
        streaming=StreamingCapabilities(timestamps=StreamTimestampsCap(mode="post_align"))
    )
    effective = DeclaredCapabilities(
        streaming=StreamingCapabilities(timestamps=StreamTimestampsCap(mode="native_frame_aligned"))
    )
    assert declared.covers(effective) is False


def test_covers_allows_standard_mode_reduction_between_supported_modes() -> None:
    # A change to a provably-weaker (but still supported) standard mode is a
    # legal narrowing: the fail-closed guard must fall through, not reject it.
    declared = DeclaredCapabilities(
        streaming=StreamingCapabilities(reconnect=ReconnectCap(mode="seamless"))
    )
    effective = DeclaredCapabilities(
        streaming=StreamingCapabilities(reconnect=ReconnectCap(mode="lossy"))
    )
    assert declared.covers(effective) is True


def test_covers_unknown_mode_change_is_fail_closed() -> None:
    # R3-CAPS-03: tokens outside the standard reduction map (an x_* experimental
    # enum node) have no provable strength order, so a mode CHANGE between them
    # MUST NOT pass as a legal narrowing (fail-closed); an identical mode is a
    # trivially-valid non-widening.
    declared = _x_caps({"x_acme_decode": {"mode": "alpha"}})
    changed = _x_caps({"x_acme_decode": {"mode": "beta"}})
    unchanged = _x_caps({"x_acme_decode": {"mode": "alpha"}})
    assert declared.covers(changed) is False
    assert declared.covers(unchanged) is True


def test_streaming_flags_require_streaming_domain() -> None:
    # R3-CAPS-04: a supported streaming axis flag with an omitted streaming
    # domain is self-contradictory (an omitted domain means streaming is
    # unsupported, fail-closed) -> rejected at construction.
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="streaming_output"):
        DeclaredCapabilities(streaming_output=FlagCap(supported=True))
    with pytest.raises(ValidationError, match="streaming_input"):
        DeclaredCapabilities(
            streaming_input=FlagCap(supported=True),
            streaming_output=FlagCap(supported=True),
        )
    # Flags + domain is the legitimate streaming declaration.
    ok = DeclaredCapabilities(
        streaming=StreamingCapabilities(),
        streaming_input=FlagCap(supported=True),
        streaming_output=FlagCap(supported=True),
    )
    assert ok.supports("streaming_input") is True
    # No flags + no domain (batch-only) stays valid.
    batch_only = DeclaredCapabilities(batch=BatchCapabilities())
    assert batch_only.supports("streaming") is False


def test_unsupported_feature_constraints_not_in_supported_paths() -> None:
    # H1: an unsupported feature's constraint sub-container MUST NOT appear as a
    # supported path (constraints is a default-factory, never None).
    caps = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(candidate_languages=CandidateLanguagesCap(supported=False)),
            guidance=GuidanceCaps(prompt=PromptCap(supported=False)),
        )
    )
    paths = set(caps.iter_supported_paths())
    assert "batch.language.candidate_languages" not in paths
    assert "batch.language.candidate_languages.constraints" not in paths
    assert "batch.guidance.prompt.constraints" not in paths


def test_supports_fail_closed_on_constraints_subpaths() -> None:
    # CAPA-2: a constraints submodel is NOT a capability node -- supports() on a
    # constraints sub-path is fail-CLOSED, even when the feature IS supported,
    # and obviously when it is not.
    caps = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=3),
                )
            ),
            guidance=GuidanceCaps(prompt=PromptCap(supported=False)),
        )
    )
    # Real capability paths still answer True.
    assert caps.supports("batch.language.candidate_languages") is True
    # Constraints sub-paths are fail-closed even under a SUPPORTED feature.
    assert caps.supports("batch.language.candidate_languages.constraints") is False
    # ... and under an UNSUPPORTED feature.
    assert caps.supports("batch.guidance.prompt.constraints") is False


def test_node_at_returns_typed_node() -> None:
    caps = DeclaredCapabilities(
        batch=BatchCapabilities(
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word"])
        )
    )
    node = caps.node_at("batch.word_timestamps")
    assert isinstance(node, WordTimestampsCap)
    assert node.granularities == ["word"]
    # Containers and missing paths return None (not a leaf node).
    assert caps.node_at("batch") is None
    assert caps.node_at("batch.nope") is None


def test_unknown_x_namespace_key_tolerated() -> None:
    # Forward-compat: extra keys parse without error and are queryable.
    caps = DeclaredCapabilities.model_validate(
        {
            "batch": {"x_acme_beamsearch": {"supported": True}},
            "streaming_input": {"supported": False},
        }
    )
    assert caps.supports("batch.x_acme_beamsearch") is True
    assert caps.supports("batch.x_acme_unknown") is False


def test_supports_fail_closed_on_unknown_non_extension_segment() -> None:
    # A path segment that is neither a known standard field nor an x_* extension
    # key is fail-closed (returns False), even when the typo'd key was tolerated
    # on parse (containers use extra="allow" for forward compat). A typo of a real
    # capability MUST NOT read as supported and weaken the gating contract.
    assert DeclaredCapabilities().supports("streamnig") is False
    # A typo'd top-level key that landed in model_extra still fails closed.
    top_typo = DeclaredCapabilities.model_validate({"streamnig": {"supported": True}})
    assert top_typo.supports("streamnig") is False
    assert "streamnig" not in set(top_typo.iter_supported_paths())
    # A typo'd field inside a typed container (batch.word_timestmaps) too.
    nested_typo = DeclaredCapabilities.model_validate(
        {"batch": {"word_timestmaps": {"supported": True}}}
    )
    assert nested_typo.supports("batch.word_timestmaps") is False
    assert "batch.word_timestmaps" not in set(nested_typo.iter_supported_paths())
    # A real declared standard path and a real x_* path still resolve correctly.
    declared = DeclaredCapabilities.model_validate(
        {
            "batch": {
                "word_timestamps": {"supported": True, "granularities": ["word"]},
                "x_acme_beam": {"supported": True},
            }
        }
    )
    assert declared.supports("batch.word_timestamps") is True
    assert declared.supports("batch.x_acme_beam") is True
    paths = set(declared.iter_supported_paths())
    assert {"batch.word_timestamps", "batch.x_acme_beam"} <= paths


def test_supports_descends_non_extension_keys_inside_x_star_subtree() -> None:
    # The x_* filter applies only to extra keys on typed standard nodes. Inside a
    # raw x_* subtree the vendor owns the whole structure, so non-x_* child keys
    # (e.g. "nested") still resolve and appear as supported paths.
    caps = DeclaredCapabilities.model_validate(
        {"batch": {"x_container": {"nested": {"supported": True}}}}
    )
    assert caps.supports("batch.x_container.nested") is True
    paths = set(caps.iter_supported_paths())
    assert {"batch.x_container", "batch.x_container.nested"} <= paths


# --------------------------------------------------------------------------- #
# x_* extension subtrees parse to plain dict nodes; the traversal/derivation
# helpers (_get_child, _read_attr, _node_items, _derive_supported) must handle
# dicts as well as typed models. These exercise those dict branches.
# --------------------------------------------------------------------------- #
def _x_caps(batch_extra: dict[str, object]) -> DeclaredCapabilities:
    return DeclaredCapabilities.model_validate({"batch": batch_extra})


def test_dict_node_supports_via_mode_and_supported_keys() -> None:
    # A dict node with a "mode" key is supported unless the mode is an off-mode;
    # a dict with only a "supported" key follows that flag; a bare container dict
    # (no recognised keys) counts as present/supported.
    caps = _x_caps(
        {
            "x_mode_on": {"mode": "lossy"},
            "x_mode_off": {"mode": "unsupported"},
            "x_flag_true": {"supported": True},
            "x_flag_false": {"supported": False},
            "x_container": {"nested": {"supported": True}},
        }
    )
    assert caps.supports("batch.x_mode_on") is True
    assert caps.supports("batch.x_mode_off") is False
    assert caps.supports("batch.x_flag_true") is True
    assert caps.supports("batch.x_flag_false") is False
    # Present container dict + descend into its nested supported child.
    paths = set(caps.iter_supported_paths())
    assert "batch.x_container" in paths
    assert "batch.x_container.nested" in paths


def test_dict_node_supported_is_strict_boolean() -> None:
    # A raw x_* dict node's `supported` is read as a STRICT boolean: only a real
    # bool True counts. A non-bool (the STRING "false" -- truthy in Python -- or a
    # number) is a malformed declaration and is fail-closed to False, never
    # silently promoted to supported.
    caps = _x_caps(
        {
            "x_str_false": {"supported": "false"},
            "x_str_true": {"supported": "true"},
            "x_num_one": {"supported": 1},
            "x_real_true": {"supported": True},
            "x_real_false": {"supported": False},
        }
    )
    assert caps.supports("batch.x_str_false") is False
    assert caps.supports("batch.x_str_true") is False
    assert caps.supports("batch.x_num_one") is False
    assert caps.supports("batch.x_real_true") is True
    assert caps.supports("batch.x_real_false") is False
    # canonical_json injects the same fail-closed value for cross-language clients.
    cj = caps.canonical_json()
    assert cj["batch"]["x_str_false"]["supported"] is False
    assert cj["batch"]["x_real_true"]["supported"] is True


def test_dict_node_mode_does_not_override_explicit_supported_false() -> None:
    # An explicit `supported: false` is authoritative: a `mode` sub-key on the
    # same node MUST NOT raise it back to true (fail-closed, spec §C R6). A node
    # carrying only `mode` (no `supported`) still derives from the mode.
    caps = _x_caps(
        {
            "x_off_with_mode": {"supported": False, "mode": "seamless"},
            "x_on_with_mode": {"supported": True, "mode": "unsupported"},
            "x_mode_only": {"mode": "seamless"},
        }
    )
    assert caps.supports("batch.x_off_with_mode") is False
    # An explicit supported=True is likewise authoritative over the mode key.
    assert caps.supports("batch.x_on_with_mode") is True
    # No `supported` key -> mode governs.
    assert caps.supports("batch.x_mode_only") is True


def test_dict_node_mode_is_strict_string() -> None:
    # A raw x_* dict node's `mode` is read as a STRICT string archetype token: a
    # non-string (bool, number, None) is a malformed declaration and is
    # fail-CLOSED to False, never silently promoted to supported. Without the type
    # guard, `True`/`1` would pass the `not in {off-modes}` frozenset check (the
    # off-modes are strings) and be wrongly reported as supported (spec §C R1).
    caps = _x_caps(
        {
            "x_mode_true": {"mode": True},
            "x_mode_one": {"mode": 1},
            "x_mode_none": {"mode": None},
            "x_mode_str": {"mode": "lossy"},
        }
    )
    assert caps.supports("batch.x_mode_true") is False
    assert caps.supports("batch.x_mode_one") is False
    assert caps.supports("batch.x_mode_none") is False
    # A real string mode (not an off-mode) still derives as supported.
    assert caps.supports("batch.x_mode_str") is True
    # canonical_json injects the same fail-closed value for cross-language clients.
    cj = caps.canonical_json()
    assert cj["batch"]["x_mode_true"]["supported"] is False
    assert cj["batch"]["x_mode_str"]["supported"] is True


def test_covers_with_dict_nodes_constraint_narrowing() -> None:
    # Both declared and effective carry an x_* dict node with a numeric upper
    # bound; widening it must be rejected, narrowing/equal accepted (the dict
    # branch of the constraint comparison).
    declared = _x_caps({"x_feat": {"supported": True, "constraints": {"max": 5}}})
    narrowed = _x_caps({"x_feat": {"supported": True, "constraints": {"max": 2}}})
    widened = _x_caps({"x_feat": {"supported": True, "constraints": {"max": 9}}})
    assert declared.covers(narrowed) is True
    assert declared.covers(declared) is True
    assert declared.covers(widened) is False


def test_covers_rejects_dict_node_unbounding() -> None:
    # Declared has a finite bound; effective drops it to None (unbounded) -> a
    # widening of an open bound, which must be rejected.
    declared = _x_caps({"x_feat": {"supported": True, "constraints": {"max": 5}}})
    unbounded = _x_caps({"x_feat": {"supported": True, "constraints": {"max": None}}})
    assert declared.covers(unbounded) is False


def test_traversal_helpers_handle_non_container_nodes() -> None:
    # A primitive (non-model, non-dict) value can appear as a "node" when a path
    # descends through a leaf scalar; the traversal helpers must degrade to a safe
    # empty/None result rather than raise.
    assert _get_child(42, "anything") is None
    assert _get_child("a string", "x") is None
    assert _read_attr(42, "max") is None
    assert _read_attr(True, "mode") is None
    assert _children(42) == []
    assert _children("scalar") == []


def test_canonical_json_injects_derived_supported() -> None:
    import json

    from standard_asr.capabilities import FinalityCap

    caps = DeclaredCapabilities(
        batch=BatchCapabilities(),
        streaming=StreamingCapabilities(
            reconnect=ReconnectCap(mode="unsupported"),
            timestamps=StreamTimestampsCap(mode="native_frame_aligned"),
            finality_level=FinalityCap(mode="closed"),
        ),
        streaming_input=FlagCap(supported=True),
    )
    cj = caps.canonical_json()

    # The root is the container of all modes, not a capability: no supported key.
    assert "supported" not in cj
    # A present mode domain is supported.
    assert cj["batch"]["supported"] is True
    assert cj["streaming"]["supported"] is True
    # Flag nodes keep their real supported field.
    assert cj["streaming_input"]["supported"] is True
    # enum/mode nodes now expose a uniform derived supported alongside mode.
    assert cj["streaming"]["reconnect"] == {"mode": "unsupported", "supported": False}
    assert cj["streaming"]["timestamps"]["supported"] is True
    assert cj["streaming"]["finality_level"]["supported"] is True
    # The whole tree is JSON-serializable.
    json.dumps(cj)


def test_canonical_json_preserves_unknown_extra_keys() -> None:
    # Containers tolerate unknown keys (R6 forward-compat); canonical JSON passes
    # them through, including nested dict values, without injecting supported.
    caps = DeclaredCapabilities.model_validate(
        {"batch": {"x_vendor": {"flavor": "fast"}, "streaming_input": True}}
    )
    cj = caps.canonical_json()
    assert cj["batch"]["x_vendor"] == {"flavor": "fast"}


def test_canonical_json_injects_supported_on_json_sourced_x_star_dicts() -> None:
    # CAPA-3: an x_* capability parsed from JSON lands in model_extra as a raw
    # dict. canonical_json MUST inject the derived `supported` so cross-language
    # clients get the same uniform probe as typed nodes. A bare constraints-like
    # dict (no mode/supported) stays untouched.
    caps = DeclaredCapabilities.model_validate(
        {
            "batch": {
                "x_acme_flag": {"supported": True},
                "x_acme_mode": {"mode": "lossy"},
                "x_acme_off": {"mode": "unsupported"},
                "x_acme_bounded": {"supported": True, "constraints": {"max": 5}},
            }
        }
    )
    cj = caps.canonical_json()
    assert cj["batch"]["x_acme_flag"]["supported"] is True
    # mode dict gains a derived supported alongside its mode.
    assert cj["batch"]["x_acme_mode"] == {"mode": "lossy", "supported": True}
    assert cj["batch"]["x_acme_off"]["supported"] is False
    # bounded x_* dict keeps its constraints; constraints dict itself gets NO
    # supported (not a capability node).
    assert cj["batch"]["x_acme_bounded"]["supported"] is True
    assert cj["batch"]["x_acme_bounded"]["constraints"] == {"max": 5}
    assert "supported" not in cj["batch"]["x_acme_bounded"]["constraints"]


def test_self_resamples_is_declarable_engine_global_flag() -> None:
    # X-AU-1: `self_resamples` is the one behavioural capability the spec places
    # in Capabilities (spec §AI 3.2, §C R7). It is engine-global, queried via
    # `supports("self_resamples")` like streaming_input/streaming_output, and is
    # informational only -- it does not change any resampling decision.
    declared = DeclaredCapabilities(self_resamples=FlagCap(supported=True))
    assert declared.supports("self_resamples") is True
    # Appears in canonical JSON with its real supported field.
    assert declared.canonical_json()["self_resamples"] == {"supported": True}
    # Absent => False (fail-closed), and the node defaults to not-supported.
    default = DeclaredCapabilities()
    assert default.supports("self_resamples") is False
    assert default.canonical_json()["self_resamples"] == {"supported": False}


def test_canonical_json_absent_domain_is_null() -> None:
    # An unsupported mode domain serializes as null (fail-closed), never as a
    # present-but-empty container.
    caps = DeclaredCapabilities(streaming=StreamingCapabilities())
    cj = caps.canonical_json()
    assert cj["batch"] is None
    assert cj["streaming"]["supported"] is True


def test_canonical_json_preserves_constraints_without_supported() -> None:
    # Constraint submodels are not capabilities: they keep their fields but get
    # no injected supported key.
    caps = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(
                candidate_languages=CandidateLanguagesCap(
                    supported=True,
                    constraints=CandidateLanguagesConstraints(max=3),
                ),
            )
        )
    )
    cj = caps.canonical_json()
    cand = cj["batch"]["language"]["candidate_languages"]
    assert cand["supported"] is True
    assert cand["constraints"] == {"max": 3}


def test_covers_continues_past_unresolvable_node() -> None:
    # When a supported path in `other` does not resolve to a node in this tree
    # (declared_node is None), the per-node narrowing check is skipped (continue)
    # but set-containment still governs the result.
    declared = _x_caps({"x_feat": {"supported": True}})
    # `other` supports the same path; declared._resolve finds it, but we also add
    # a present container whose effective node differs in shape.
    same = _x_caps({"x_feat": {"supported": True}})
    assert declared.covers(same) is True
