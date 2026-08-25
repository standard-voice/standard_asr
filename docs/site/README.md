# Documentation site

This directory holds the documentation site: a [Next.js](https://nextjs.org)
application built with [Fumadocs](https://fumadocs.dev), exported as a fully
static site and deployed to GitHub Pages. It renders the Markdown
content in `docs/content/`, and that directory is the whole rule: every
Markdown file under it is published, and nothing outside it is. Internal
material (design notes, research, working notes, historical pages) lives in
`docs/internal/`, which the site never sees. Adding a page is adding a file.

Only the docs toolchain uses Node. Regular Python development never needs
this directory.

## Develop

```bash
cd docs/site
pnpm install
pnpm dev
```

The dev server runs at `http://localhost:3000` with no base path.

`pnpm dev` and `pnpm build` first regenerate `.generated/api.json`, the
neutral API data the reference pages render. The generator
(`scripts/dump_api.py`) reads the Python source with Griffe, so the
repository's `uv` environment must be synced first
(`uv sync --locked --all-extras --all-groups` at the repository root).

## Build

```bash
NEXT_PUBLIC_BASE_PATH=/standard_asr pnpm build
```

The static site lands in `out/`. `NEXT_PUBLIC_BASE_PATH` carries the GitHub
Pages project prefix; leave it unset to build for a root domain. The
post-build step (`scripts/postbuild.mjs`) writes `.nojekyll` and redirect
stubs for the previous site's URLs.

To preview the export the way GitHub Pages serves it, the base path has to
be part of the URL, so serve a directory that contains it:

```bash
mkdir -p /tmp/preview && ln -sfn "$PWD/out" /tmp/preview/standard_asr
python3 -m http.server --directory /tmp/preview 8000
# then open http://localhost:8000/standard_asr/
```

Serving `out/` directly at the root would load, but every base-pathed link
and the search index would resolve differently than in production.

## Checks

`pnpm lint` runs [oxlint](https://oxc.rs) with the React hooks rules
(`rules-of-hooks`, `exhaustive-deps`, `set-state-in-effect`), configured in
`.oxlintrc.json`. `pnpm types:check` runs the TypeScript compiler in strict
mode, and `next build` enforces the App Router rules that matter for
correctness (server/client boundaries, route exports). The linter is oxlint
rather than ESLint because ESLint plus `eslint-config-next` pulled 273
transitive packages into a tree of 502 to lint about a dozen files, and
three of those packages were the only ones the repository's license gate
rejected. oxlint ships the same hooks checks as one MIT-licensed binary.
`pnpm format` runs [oxfmt](https://oxc.rs), the Oxc formatter, with
Prettier-compatible output, mirroring `ruff format` on the Python side;
CI enforces it with `pnpm format:check`. Linter and formatter come from
the same Oxc toolchain.

CI also link-checks the export with [lychee](https://github.com/lycheeverse/lychee):
every internal link and `#fragment` target in `out/` must resolve, which
guards the specification's stable anchors. The check runs offline, so
external links are excluded and the result is deterministic. To reproduce
locally, run lychee against a directory that mirrors the Pages prefix, the
same shape the preview uses:

```bash
mkdir -p /tmp/linkroot && ln -sfn "$PWD/out" /tmp/linkroot/standard_asr
lychee --offline --include-fragments --index-files index.html \
  --root-dir /tmp/linkroot 'out/**/*.html'
```

## Content rules

- Published prose lives in `docs/content/` as plain Markdown, governed by
  `STYLE.md` and the Vale gate. This app directory is exempt from Vale
  (`scripts/vale.sh`); keep governed prose out of it.
- `docs/content/meta.json` orders the sidebar. An unlisted page still
  publishes and appends at the end (the `...` entry); a stale entry fails
  the build (`scripts/check-content.mjs`).
- Link a sibling page with an explicit relative path (`./other-page.md`,
  `../section/page.md`). A bare `page.md` is not resolved and ships as a
  broken link.
- Markdown files cannot use JSX. Raw HTML fails the build loudly
  (`source.config.ts`, `remarkHtmlGuard`) instead of being dropped
  silently; `<br>` is converted.
- Explicit heading anchors use the mkdocs form `{#id}` in the source and
  are preserved by `remarkMkdocsAnchors`.
- API reference pages declare `api_module` in their frontmatter; the
  generated reference renders after the page's own prose.
