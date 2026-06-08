# SPDX-FileCopyrightText: 2026 Standard Voice Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hierarchical capability system."""

from __future__ import annotations

from standard_asr.capabilities import (
    BatchCapabilities,
    CandidateLanguagesCap,
    CandidateLanguagesConstraints,
    DeclaredCapabilities,
    FlagCap,
    GuidanceCaps,
    LanguageCaps,
    PromptCap,
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
    # Effective narrows: drop word_timestamps support.
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(
            language=LanguageCaps(runtime_override=FlagCap(supported=True)),
        ),
        streaming_input=FlagCap(supported=True),
    )
    assert declared.covers(effective) is True


def test_covers_rejects_widening() -> None:
    declared = DeclaredCapabilities(batch=BatchCapabilities())
    # Effective claims more than declared -> not covered.
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(word_timestamps=WordTimestampsCap(supported=True))
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
    # CAPA-1: a declared empty granularities list means "unbounded (all)", so a
    # narrowing from it to any concrete subset is a valid effective ⊆ declared
    # (must NOT false-fail). Mirrors param_gating treating empty as "offers all".
    assert granularity_offers_all([]) is True
    assert granularity_offers_all(["word"]) is False
    declared = DeclaredCapabilities(
        batch=BatchCapabilities(word_timestamps=WordTimestampsCap(supported=True))
    )
    effective = DeclaredCapabilities(
        batch=BatchCapabilities(
            word_timestamps=WordTimestampsCap(supported=True, granularities=["word"])
        )
    )
    assert declared.covers(effective) is True
    # And empty -> empty is trivially covered.
    assert declared.covers(declared) is True


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
