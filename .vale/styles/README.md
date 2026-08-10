# Vale styles

- `Google/` — the [errata-ai/Google](https://github.com/errata-ai/Google)
  package (v0.7.1; MIT license, Copyright (c) 2018 - 2019 Joseph Kato — full
  text in `Google/LICENSE`, attribution recorded in the repository `NOTICE`):
  a Vale implementation of the
  [Google developer documentation style guide](https://developers.google.com/style/),
  whose guide text is CC BY 4.0 and is the source the rule messages derive
  from. Vendored, not fetched at run time, so CI needs no
  network and the ruleset cannot drift under us. To upgrade: `vale sync` with
  `Packages = Google`, diff, and record the new version here. Do not edit the
  vendored files; tune rules from `.vale.ini`.
- `StandardASR/` — this repo's own rules: the mechanizable subset of
  [`TERMINOLOGY.md`](../../TERMINOLOGY.md) (forbidden synonyms, American
  spelling, the emoji ban), plus `Spelling.yml`, which replaces
  `Vale.Spelling` with token-level filters for identifier-shaped words
  (snake_case, CamelCase, `OSError`-style leading caps, dotted paths). The
  filters exist because the `TokenIgnores` alternative rewrites text before
  parsing and corrupts code-span shielding and position mapping.
- `config/vocabularies/StandardASR/` — the accept list: domain terms and
  code-adjacent words the spelling rules must not flag. Entries are
  case-insensitive regexes (`(?i)word`); keep them sorted.
