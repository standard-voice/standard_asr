// Post-export fixups for GitHub Pages:
// - .nojekyll: insurance against Jekyll processing if the Pages source is
//   ever switched back to branch deployment.
// - Redirect stubs: the previous mkdocs site (offline plugin, so no
//   directory URLs) served every page at /<path>.html; the same content
//   now lives under /docs/. Each old URL gets a static stub at its exact
//   old path (snapshot of the final gh-pages deploy) that forwards and
//   keeps the fragment. Reference stubs also translate the old dotted
//   mkdocstrings anchors (#standard_asr.engine.EngineBase.run) to the new
//   bare ids (#EngineBase.run) by stripping their module prefix, and
//   prose stubs translate the heading ids the slugifier change renamed
//   (legacy_fragments.json: every heading id from the final gh-pages
//   deploy that github-slugger derives differently, mapped to the id the
//   new page renders).
import { access, mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const outDir = fileURLToPath(new URL('../out/', import.meta.url));
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? '';
const legacyFragments = JSON.parse(
  await readFile(new URL('./legacy_fragments.json', import.meta.url), 'utf-8'),
);

// Old mkdocs URLs (site-root relative) -> new locations. index.html is the
// new home page itself, and 404.html is Next's own, so neither gets a stub.
const redirects = {
  'quickstart.html': { to: 'docs/quickstart/' },
  'installation.html': { to: 'docs/installation/' },
  'mission.html': { to: 'docs/mission/' },
  'goals.html': { to: 'docs/goals/' },
  'for_app_dev/discover_and_use.html': { to: 'docs/app-developers/discover-and-use/' },
  'for_app_dev/streaming.html': { to: 'docs/app-developers/streaming/' },
  'for_app_dev/errors.html': { to: 'docs/app-developers/errors/' },
  'for_asr_dev/adapting_engine.html': { to: 'docs/engine-authors/adapt-an-asr-system/' },
  'for_asr_dev/plugin_entrypoints.html': { to: 'docs/engine-authors/plugin-entry-points/' },
  'spec/specification.html': { to: 'docs/specification/protocol/' },
  'spec/server.html': { to: 'docs/specification/server-api/' },
  'spec/cli.html': { to: 'docs/specification/cli/' },
  'spec/download-policy.html': { to: 'docs/specification/download-policy/' },
  'reference/index.html': { to: 'docs/reference/', module: 'standard_asr' },
  'reference/engine.html': { to: 'docs/reference/engine/', module: 'standard_asr.engine' },
  'reference/compliance.html': {
    to: 'docs/reference/compliance/',
    module: 'standard_asr.compliance',
  },
  'reference/streaming.html': {
    to: 'docs/reference/streaming/',
    module: 'standard_asr.runtime.streaming',
  },
  'reference/results.html': {
    to: 'docs/reference/results/',
    module: 'standard_asr.contract.results',
  },
  'reference/capabilities.html': {
    to: 'docs/reference/capabilities/',
    module: 'standard_asr.contract.capabilities',
  },
  'reference/exceptions.html': {
    to: 'docs/reference/exceptions/',
    module: 'standard_asr.contract.exceptions',
  },
  'reference/wire.html': { to: 'docs/reference/wire/', module: 'standard_asr.audio.wire' },
};

function stub(target, module, fragments) {
  const url = `${basePath}/${target}`;
  const strip = module
    ? `const p=${JSON.stringify(`#${module}.`)};if(h.startsWith(p))h='#'+h.slice(p.length);`
    : '';
  const translate = fragments
    ? `const m=${JSON.stringify(fragments)};const k=m[h.slice(1)];if(k)h='#'+k;`
    : '';
  return `<!doctype html>
<meta charset="utf-8">
<title>Redirecting\u2026</title>
<link rel="canonical" href="${url}">
<script>let h=location.hash;${strip}${translate}location.replace(${JSON.stringify(url)} + h);</script>
<meta http-equiv="refresh" content="0; url=${url}">
<p>This page moved to <a href="${url}">${url}</a>.</p>
`;
}

await writeFile(path.join(outDir, '.nojekyll'), '');
for (const [from, { to, module }] of Object.entries(redirects)) {
  const file = path.join(outDir, from);
  // A stub must never shadow a real exported page.
  if (
    await access(file).then(
      () => true,
      () => false,
    )
  ) {
    throw new Error(`redirect stub would overwrite an exported file: ${from}`);
  }
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, stub(to, module, legacyFragments[from]));
}
console.log(`postbuild: .nojekyll + ${Object.keys(redirects).length} redirect stubs`);
