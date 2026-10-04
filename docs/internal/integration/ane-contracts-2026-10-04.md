# ANE integration candidate

This branch combines Standard ASR main `97bfdb2` with the local-engine contracts from `5f6eef2`. It is a new integration candidate; it does not change the disposition of PR 106. Nothing has been pushed.

## Contract decisions

Main owns partial stability: events use `stable_text`, capabilities use `partial_stability`, and `supersede` withdraws the retired segments together with their stable text. The removed `stable_until`, `word_stability`, and cross-supersede frozen-prefix obligations are not compatibility aliases. Their obsolete tests were removed where automatic merging retained them. Main's stable-text fixtures and lifecycle checks remain.

Exact text composition remains explicit through `Segment.text_separator`, event separators, `compose_segment_text`, and `compose_reduced_text`. The reducer retains finalized word and segment evidence. A separator established with a nonempty stable prefix cannot change on a later partial or plain final; closed formatting and supersede withdrawal follow main's existing exemptions. The specification states this rule alongside the separator field.

Audio progress is independent of aligned transcript timestamps. The unified capability checker now checks `audio_processed_until` against `audio_progress` and checks `start`/`end` against `timestamps`. It retains all of main's other capability checks. An undeclared field remains visible with a diagnostic in ordinary mode; strict mode terminates with `engine_error`. This replaces PR 106's separate cursor-removal guard. The two checks prevent an adapter from claiming alignment merely to report processed audio.

Strict final results, explicit `partial_result()` snapshots, `SessionStatus`, measured input duration, native-bulk prepare/finalize hooks, configured duration limits, typed provider options, the server engine pool, and deterministic engine close remain. The shared pipeline and transport implementation retain their previous tests. No plugin-specific implementation is added to Standard ASR.

`runtime.engine_pool` remains an internal server implementation; it is not added to the public API reference. Its operator-facing behavior is documented in the server specification. No new API reference page is required by the integration itself.

## Merge conflicts

| File | Resolution |
| --- | --- |
| `src/standard_asr/runtime/streaming.py` | Keep main's stable-text validation, Unicode boundary repair, supersede withdrawal, and capability guard. Retain exact separators, reduced evidence, input duration, strict result/status methods, and sync bridge methods. Fold independent progress into the capability guard. Remove the old private cursor configuration path and supersede-obligation machinery. |
| `src/standard_asr/contract/capabilities.py` | Keep `partial_stability`; add independent `audio_progress`. Do not restore `word_stability`. |
| `src/standard_asr/compliance.py` | Use main's shared guard for capability checks. Retain engine close as a protocol obligation and document the separate progress/timestamp checks. |
| `src/standard_asr/__init__.py` | Export both exact-composition helpers and `bind_session_capabilities`, alongside status/error additions. |
| `src/standard_asr/toolchain/server.py` | Retain engine leases and typed provider validation, and bind effective capabilities for structural engines. |
| `tests/test_asr_interface.py` | Retain both duration/progress fixtures and main's effective-capability fixtures. Tests inspecting a deliberately failed session request its explicit partial snapshot. |
| `tests/test_compliance.py` | Keep main's stable-text and supersede semantics, plus independent audio-progress coverage. |
| `tests/test_streaming.py` | Keep main's stable-text/capability cases and PR 106's exact text, evidence, duration, status, and terminal-result cases. Remove tests asserting retired cross-supersede obligations. Verify progress diagnostics preserve the event. Add a distinct segment-timestamp capability case. |
| `docs/content/engine-authors/adapt-an-asr-system.md` | Use main's stable-text guidance and unified guard behavior. Document exact composition and independent audio progress. |
| `docs/content/specification/protocol.md` | Use main's stability and whole-segment withdrawal semantics; retain exact separators, status/result, duration, and local-engine contracts. Specify the independent cursor/timestamp capabilities and their diagnostics. |

The remaining files merged without textual conflicts. Their integrated behavior was checked with the full suite. Server fixtures that emit measured spans now declare timestamp support. The structural server fixture provides the close method required by the retained pool contract.

## Verification

- `uv sync --offline --all-extras` created an isolated environment from the existing package cache.
- `uv run pytest`: 2,824 passed; statement and branch coverage 100%.
- `prek run --all-files`: the first run passed lint, strict type checking, Vale, and the Actions audit; the formatter changed two test files. The second run passed every hook on the formatted tree.
- The plugin's separate integration run passed 720 tests, according to the parent task. Native inference evidence is owned by that task.

These checks establish the exercised behavior, not a substitute for independent review. An independent review of HTTP cancellation and engine-lease lifetime is proceeding separately; any confirmed defect belongs in a subsequent focused commit.
